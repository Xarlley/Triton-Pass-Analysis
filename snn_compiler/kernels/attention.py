"""脉冲注意力融合 kernel。

脉冲注意力（Spikingformer SSA / SDT-V2 MS_Attention 共有形态）核心：
    q,k,v 都是脉冲 {0,1}；  kv = kᵀ @ v ;  a = (q @ kv) * scale ;  s = attn_lif(a)
无 softmax → matmul 可结合律重排成线性序、操作数保持脉冲/小整数。

本模块提供两类「脉冲感知」kernel：

1. ``spike_av_lif`` —— 把 ``a=(q@kv)*scale`` 的第二个 matmul 与 ``*scale`` 与 ``attn_lif``
   **融成一个 kernel**：对每个 (batch·head) 的一块 token，沿时间维 T 在 **寄存器**里维持膜
   电位 v[BLOCK_N, D]，每个 t 现算 a[t]=q[t]@kv[t]、更新膜电位、直接写脉冲。
   **从不把 [T,B,heads,N,d] 的注意力图落显存**（v0 用 bmm 会落一次 ~46MB 再读回）。
   因 q∈{0,1}、kv 为 ≤N 的小整数（KᵀV of 二值），q@kv 全程是精确整数 → **与 bmm+LIF 逐位一致**。

2. ``spike_ktv`` —— ``kv = kᵀ@v`` 的脉冲×脉冲 matmul（二值），fp32 累加（小整数精确）。
   小尺寸 [d,d]，留作与 bmm 对比；默认仍可用 torch.bmm。

只把 triton 当库用（@triton.jit），不改 triton 源码 —— snn_compiler 保持独立。
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# popcount device 函数（triton libdevice）；不可用则 spike_ktv_popcount 退回报错
try:
    from triton.language.extra import libdevice
    HAS_POPC = True
except Exception:
    try:
        from triton.language.extra.cuda import libdevice
        HAS_POPC = True
    except Exception:
        libdevice = None
        HAS_POPC = False


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


# ============================================================
#   融合 kernel：a=(q@kv)*scale → LIF（膜电位寄存器跨 T）
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32}, num_warps=2),
        triton.Config({"BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_N": 128}, num_warps=8),
    ],
    key=["T", "N", "D", "BLOCK_D"],
)
@triton.jit
def _spike_av_lif_kernel(
    q_ptr,          # [T, BH, N, D]  fp（脉冲 0/1）
    kv_ptr,         # [T, BH, D, D]  fp（小整数）
    spike_ptr,      # [T, BH, N, D]  输出脉冲
    BH,
    T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    decay: tl.constexpr, input_scale: tl.constexpr,
    v_th: tl.constexpr, v_reset: tl.constexpr, RESET_HARD: tl.constexpr,
):
    pid_bh = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    d_off = tl.arange(0, BLOCK_D)
    n_mask = n_off < N
    d_mask = d_off < D

    N_i64 = tl.full([], N, tl.int64)
    D_i64 = tl.full([], D, tl.int64)
    ND = N_i64 * D_i64
    DD = D_i64 * D_i64
    BHND = tl.full([], BH, tl.int64) * ND
    BHDD = tl.full([], BH, tl.int64) * DD

    # q/spike 基址：[t, bh, n, d]  →  t*BH*N*D + bh*N*D + n*D + d
    qn = n_off.to(tl.int64)[:, None] * D_i64 + d_off.to(tl.int64)[None, :]   # [BLOCK_N, BLOCK_D]
    qmask = n_mask[:, None] & d_mask[None, :]
    # kv 基址：[t, bh, k, j] → t*BH*D*D + bh*D*D + k*D + j
    kvkj = d_off.to(tl.int64)[:, None] * D_i64 + d_off.to(tl.int64)[None, :]  # [BLOCK_D, BLOCK_D]
    kvmask = d_mask[:, None] & d_mask[None, :]

    v_mem = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    for t in tl.static_range(0, T):
        t_i64 = tl.full([], t, tl.int64)
        q_base = t_i64 * BHND + pid_bh * ND
        kv_base = t_i64 * BHDD + pid_bh * DD
        q_t = tl.load(q_ptr + q_base + qn, mask=qmask, other=0.0).to(tl.float32)     # [BN, BD]
        kv_t = tl.load(kv_ptr + kv_base + kvkj, mask=kvmask, other=0.0).to(tl.float32)  # [BD, BD]
        a = tl.dot(q_t, kv_t)                                  # [BN, BD]  (= q@kv, contract over D)
        v_mem = decay * v_mem + input_scale * a                # LIF 充电（scale 已折进 input_scale）
        spike = (v_mem >= v_th).to(tl.float32)
        if RESET_HARD:
            v_mem = v_mem * (1.0 - spike) + spike * v_reset
        else:
            v_mem = v_mem - spike * v_th
        tl.store(spike_ptr + q_base + qn, spike, mask=qmask)


def spike_av_lif(q, kv, *, scale, tau=2.0, decay=None, decay_input=True,
                 v_threshold=0.5, v_reset=0.0, soft_reset=False):
    """融合 (q@kv)*scale → LIF。

    q : [T, B, heads, N, d] 脉冲；kv : [T, B, heads, d, d]。返回脉冲 [T, B, heads, N, d]。
    LIF：decay/input_scale 同 fused_bias_if_lif；scale 折进 input_scale。
    """
    assert q.is_cuda and kv.is_cuda
    T, B, H, N, D = q.shape
    BH = B * H
    qf = q.reshape(T, BH, N, D).contiguous()
    kvf = kv.reshape(T, BH, D, D).contiguous()
    spike = torch.empty_like(qf)

    if decay is None:
        dec = 1.0 if tau is None else (1.0 - 1.0 / tau)
    else:
        dec = float(decay)
    base_scale = (1.0 / tau) if (decay_input and tau is not None) else 1.0
    in_scale = base_scale * float(scale)

    BLOCK_D = max(16, _next_pow2(D))               # tl.dot 要求收缩维 >= 16
    grid = lambda meta: (BH, triton.cdiv(N, meta["BLOCK_N"]))
    _spike_av_lif_kernel[grid](
        qf, kvf, spike, BH,
        T=T, N=N, D=D, BLOCK_D=BLOCK_D,
        decay=dec, input_scale=in_scale,
        v_th=float(v_threshold), v_reset=float(v_reset),
        RESET_HARD=(not soft_reset),
    )
    return spike.reshape(T, B, H, N, D)


# ============================================================
#   脉冲×脉冲 KᵀV（二值 matmul，fp32 精确累加）
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 64}, num_warps=2),
        triton.Config({"BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_N": 256}, num_warps=4),
    ],
    key=["N", "D", "BLOCK_D"],
)
@triton.jit
def _spike_ktv_kernel(
    k_ptr, v_ptr, out_ptr,          # k,v: [TBH, N, D]; out: [TBH, D, D]
    N: tl.constexpr, D: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)          # one (t,b,head)
    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < D
    N_i64 = tl.full([], N, tl.int64)
    D_i64 = tl.full([], D, tl.int64)
    base = pid * N_i64 * D_i64
    acc = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)   # [i, j] = sum_n K[n,i]V[n,j]
    for n0 in range(0, N, BLOCK_N):
        n_off = n0 + tl.arange(0, BLOCK_N)
        n_mask = n_off < N
        kk = n_off.to(tl.int64)[:, None] * D_i64 + d_off.to(tl.int64)[None, :]
        m = n_mask[:, None] & d_mask[None, :]
        k_blk = tl.load(k_ptr + base + kk, mask=m, other=0.0).to(tl.float32)   # [BN, BD]
        v_blk = tl.load(v_ptr + base + kk, mask=m, other=0.0).to(tl.float32)   # [BN, BD]
        acc += tl.dot(tl.trans(k_blk), v_blk)      # [BD,BN]@[BN,BD] -> [BD,BD]
    oj = d_off.to(tl.int64)[:, None] * D_i64 + d_off.to(tl.int64)[None, :]
    om = d_mask[:, None] & d_mask[None, :]
    tl.store(out_ptr + pid * D_i64 * D_i64 + oj, acc, mask=om)


def spike_ktv(k, v):
    """kv = kᵀ@v，k,v: [T,B,heads,N,d] 脉冲 → [T,B,heads,d,d]。"""
    T, B, H, N, D = k.shape
    TBH = T * B * H
    kf = k.reshape(TBH, N, D).contiguous()
    vf = v.reshape(TBH, N, D).contiguous()
    out = torch.empty((TBH, D, D), device=k.device, dtype=torch.float32)
    BLOCK_D = max(16, _next_pow2(D))               # tl.dot 要求收缩维 >= 16
    grid = (TBH,)
    _spike_ktv_kernel[grid](kf, vf, out, N=N, D=D, BLOCK_D=BLOCK_D)
    return out.reshape(T, B, H, D, D)


# ============================================================
#   bit-pack + popcount KᵀV（脉冲二值性 → 真实算力节省，逐位精确）
#   把 token 维每 32 个脉冲打包进 1 个 int32，KᵀV[i,j]=Σ_w popcount(Kp[i,w]&Vp[j,w])。
#   实测含 Triton pack 的全路径净胜 cutlass bmm 1.4–2.2×（随 batch 增大），且 max|Δ|=0。
# ============================================================
@triton.jit
def _pack_N_kernel(k_ptr, out_ptr, N, D, W, BLOCK_I: tl.constexpr):
    """k:[TBH,N,D] {0,1} -> out:[TBH,D,W] int32（沿 token 维打包，每 32 token→1 word）。"""
    pid_tbh = tl.program_id(0).to(tl.int64)
    pid_i = tl.program_id(1)
    i_off = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    i_mask = i_off < D
    N_i64 = N.to(tl.int64); D_i64 = D.to(tl.int64); W_i64 = W.to(tl.int64)
    base = pid_tbh * N_i64 * D_i64
    t_off = tl.arange(0, 32)
    shifts = (tl.full([32], 1, tl.int64) << t_off.to(tl.int64))
    for w in range(0, W):
        n = w * 32 + t_off
        n_mask = n < N
        addr = base + n.to(tl.int64)[None, :] * D_i64 + i_off.to(tl.int64)[:, None]
        m = i_mask[:, None] & n_mask[None, :]
        bit = tl.load(k_ptr + addr, mask=m, other=0).to(tl.int64)
        word = tl.sum(bit * shifts[None, :], axis=1)
        tl.store(out_ptr + pid_tbh * D_i64 * W_i64 + i_off.to(tl.int64) * W_i64 + w,
                 word.to(tl.int32), mask=i_mask)


@triton.jit
def _ktv_popcount_kernel(kpack_ptr, vpack_ptr, out_ptr, W,
                         D: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    i_off = tl.arange(0, BLOCK_D); j_off = tl.arange(0, BLOCK_D)
    i_mask = i_off < D; j_mask = j_off < D
    D_i64 = tl.full([], D, tl.int64); W_i64 = W.to(tl.int64)
    base = pid * D_i64 * W_i64
    acc = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.int32)
    for w in range(0, W):
        wi = tl.full([], w, tl.int64)
        ki = tl.load(kpack_ptr + base + i_off.to(tl.int64) * W_i64 + wi, mask=i_mask, other=0)
        vj = tl.load(vpack_ptr + base + j_off.to(tl.int64) * W_i64 + wi, mask=j_mask, other=0)
        acc += libdevice.popc(ki[:, None] & vj[None, :])
    oj = i_off.to(tl.int64)[:, None] * D_i64 + j_off.to(tl.int64)[None, :]
    om = i_mask[:, None] & j_mask[None, :]
    tl.store(out_ptr + pid * D_i64 * D_i64 + oj, acc.to(tl.float32), mask=om)


def _pack_N(xf):
    """xf:[TBH,N,D] {0,1} -> [TBH,D,W] int32。"""
    TBH, N, D = xf.shape
    W = (N + 31) // 32
    out = torch.empty((TBH, D, W), device=xf.device, dtype=torch.int32)
    _pack_N_kernel[(TBH, triton.cdiv(D, 64))](xf.contiguous(), out, N, D, W, BLOCK_I=64)
    return out


def spike_ktv_popcount(k, v):
    """bit-exact 的 KᵀV，用 bit-pack + popcount。k,v:[T,B,heads,N,d] 脉冲 → [T,B,heads,d,d]。

    实测净胜 cutlass bmm 1.4–2.2×（随 batch 增大）。要求 libdevice.popc 可用（triton 3.x）。
    """
    if not HAS_POPC:
        raise RuntimeError("libdevice.popc unavailable in this triton build; use ktv_mode='bmm'")
    T, B, H, N, D = k.shape
    TBH = T * B * H
    kp = _pack_N(k.reshape(TBH, N, D))
    vp = _pack_N(v.reshape(TBH, N, D))
    out = torch.empty((TBH, D, D), device=k.device, dtype=torch.float32)
    _ktv_popcount_kernel[(TBH,)](kp, vp, out, kp.shape[-1], D=D, BLOCK_D=max(16, _next_pow2(D)))
    return out.reshape(T, B, H, D, D)
