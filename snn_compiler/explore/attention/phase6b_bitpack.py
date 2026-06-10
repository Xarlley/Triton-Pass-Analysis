"""P6b 探索：bit-pack + popcount 的 KᵀV（二值 GEMM）能否打过 cutlass bmm？

KᵀV[i,j] = Σ_n K[n,i]·V[n,j] = Σ_w popcount(Kpack[i,w] & Vpack[j,w])，
把 N(token) 维打包进 int32 字（W=ceil(N/32)≈7）→ 收缩从 N=196 降到 W=7（~28× 少的内积步）。
但用标量 AND+popcount（无 tensor core）vs cutlass（tensor core）—— 实测谁快是开放问题。

诚实记录正/负结果。跑法：cd ~/charlley/snn_compiler_attn && python snn_compiler/explore/attention/phase6b_bitpack.py
"""
import os, sys
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
import _bench_util as BU
from snn_compiler.kernels.attention import spike_ktv

dev = "cuda"

# popcount device 函数（triton libdevice）
try:
    from triton.language.extra import libdevice
    _POPC = libdevice.popc
except Exception:
    try:
        from triton.language.extra.cuda import libdevice
        _POPC = libdevice.popc
    except Exception:
        _POPC = None


def bitpack_N(x):
    """x:[TBH,N,d] {0,1} -> packed:[TBH,d,W] int32（按 token 维打包），W=ceil(N/32)。"""
    TBH, N, d = x.shape
    W = (N + 31) // 32
    xt = x.transpose(1, 2).contiguous()           # [TBH,d,N]
    pad = W * 32 - N
    if pad:
        xt = F.pad(xt, (0, pad))
    xt = xt.reshape(TBH, d, W, 32).to(torch.int64)
    shifts = torch.arange(32, device=x.device, dtype=torch.int64)
    packed = (xt << shifts).sum(-1)               # int64，低 32 位是打包字
    return packed.to(torch.int32).contiguous()    # [TBH,d,W]


@triton.jit
def _ktv_popcount_kernel(kpack_ptr, vpack_ptr, out_ptr, W,
                         D: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    i_off = tl.arange(0, BLOCK_D)
    j_off = tl.arange(0, BLOCK_D)
    i_mask = i_off < D
    j_mask = j_off < D
    D_i64 = tl.full([], D, tl.int64)
    W_i64 = W.to(tl.int64)
    base = pid * D_i64 * W_i64
    acc = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.int32)
    for w in range(0, W):
        wi = tl.full([], w, tl.int64)
        ki = tl.load(kpack_ptr + base + i_off.to(tl.int64) * W_i64 + wi, mask=i_mask, other=0)  # [BD]
        vj = tl.load(vpack_ptr + base + j_off.to(tl.int64) * W_i64 + wi, mask=j_mask, other=0)  # [BD]
        andij = (ki[:, None] & vj[None, :])        # [BD,BD] int32
        acc += libdevice.popc(andij)
    oj = i_off.to(tl.int64)[:, None] * D_i64 + j_off.to(tl.int64)[None, :]
    om = i_mask[:, None] & j_mask[None, :]
    tl.store(out_ptr + pid * D_i64 * D_i64 + oj, acc.to(tl.float32), mask=om)


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


def ktv_popcount(k, v):
    """k,v:[T,B,heads,N,d] {0,1} -> kv:[T,B,heads,d,d]（bit-pack+popcount）。"""
    T, B, H, N, D = k.shape
    TBH = T * B * H
    kp = bitpack_N(k.reshape(TBH, N, D))           # [TBH,d,W]
    vp = bitpack_N(v.reshape(TBH, N, D))
    W = kp.shape[-1]
    out = torch.empty((TBH, D, D), device=k.device, dtype=torch.float32)
    BLOCK_D = _next_pow2(D)
    _ktv_popcount_kernel[(TBH,)](kp, vp, out, W, D=D, BLOCK_D=BLOCK_D)
    return out.reshape(T, B, H, D, D), (kp, vp)


def spikes(shape, rate):
    return (torch.rand(*shape, device=dev) < rate).float()


def run(tag, T, B, heads, N, d, rk, rv):
    k = spikes((T, B, heads, N, d), rk)
    v = spikes((T, B, heads, N, d), rv)
    kv_bmm = k.transpose(-2, -1) @ v

    if _POPC is None:
        print(f"[{tag}] libdevice.popc unavailable -> skip popcount kernel")
        return
    kv_pc, (kp, vp) = ktv_popcount(k, v)
    err = (kv_bmm - kv_pc).abs().max().item()

    r_bmm = BU.bench(lambda: k.transpose(-2, -1) @ v, warmup=25, iters=100)
    r_ktv = BU.bench(lambda: spike_ktv(k, v), warmup=25, iters=100)
    # popcount：含打包成本 vs 不含（打包可在上一层 LIF 时顺带产出，故也测纯 kernel）
    r_pc_full = BU.bench(lambda: ktv_popcount(k, v)[0], warmup=25, iters=100)
    W = kp.shape[-1]
    out = torch.empty((T*B*heads, d, d), device=dev, dtype=torch.float32)
    BD = _next_pow2(d)
    r_pc_kernel = BU.bench(lambda: _ktv_popcount_kernel[(T*B*heads,)](kp, vp, out, W, D=d, BLOCK_D=BD),
                           warmup=25, iters=100)

    print(f"\n[{tag}] T={T} B={B} heads={heads} N={N} d={d} rk/rv={rk}/{rv}  W={W}")
    print(f"   correctness popcount vs bmm: max|Δ|={err:.3e}")
    print(f"   bmm(cutlass)        : {BU.fmt(r_bmm)}")
    print(f"   spike_ktv(triton)   : {BU.fmt(r_ktv)}")
    print(f"   popcount full(+pack): {BU.fmt(r_pc_full)}")
    print(f"   popcount kernel only: {BU.fmt(r_pc_kernel)}")
    print(f"   => popcount-kernel / bmm = {r_bmm['median_ms']/r_pc_kernel['median_ms']:.2f}x ; "
          f"full / bmm = {r_bmm['median_ms']/r_pc_full['median_ms']:.2f}x")


def main():
    BU.gpu_guard(tag="P6b-start")
    print(f"libdevice.popc available: {_POPC is not None}")
    run("SF-like",     T=4, B=16, heads=8, N=196, d=96, rk=0.026, rv=0.047)
    run("SDT-like",    T=4, B=16, heads=8, N=196, d=64, rk=0.034, rv=0.078)
    run("SF-like B64", T=4, B=64, heads=8, N=196, d=96, rk=0.026, rv=0.047)
    BU.gpu_guard(tag="P6b-end")
    print("\nP6b_DONE")


if __name__ == "__main__":
    main()
