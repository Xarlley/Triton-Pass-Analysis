"""prototype: Conv-BN-LIF-AvgPool2x2 融合 kernel。

观察：VGG-16 中 5 个 Pool layer 跟在 LIF 后面。当前路径：
  Conv → BN → LIF → AvgPool2x2 → next Conv
  ----  ----   --                  ----
  y     spike  spike    pool_out   conv input

  spike 写一次完整 [T, B, C, H, W] (= 2×NCL×T bytes bf16)
  pool 读一次完整 + 写 1/4 的 pool_out

如果把 Pool fuse 进 LIF kernel：每个 (B, C, h, w) 输出位置 = 2×2 = 4 个 (B, C, H, W) 位置的均值。
LIF 在 t loop 内对 4 个 spike 求平均，直接写 pool_out。
- 不写 spike full buffer，省 2×NCL×T bytes write
- 不读 spike full buffer，省 2×NCL×T bytes read
- 共省 4×NCL×T bytes = 4 倍原本写入量

对 5 个 Pool layer 的 VGG-16，在 T=128, B=16 时省下约 10 GB 内存搬运。

实现：每 CTA 处理 BLOCK_NPOL 个 pool 输出位置（即 BLOCK_NPOL × 4 个 LIF 神经元）。
- v: [BLOCK_NPOL × 4] register
- spike: [BLOCK_NPOL × 4]
- pool: [BLOCK_NPOL] = sum(spike_2x2) / 4
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NPOL": 64}, num_warps=4),
        triton.Config({"BLOCK_NPOL": 128}, num_warps=4),
        triton.Config({"BLOCK_NPOL": 256}, num_warps=8),
    ],
    key=["T", "N_pool", "C", "HW_in", "HAS_BIAS"],
)
@triton.jit
def _bias_lif_pool2_kernel(
    y_ptr, bias_ptr, pool_ptr,
    T: tl.constexpr,
    N_pool: tl.constexpr,    # B * C * Hp * Wp
    NCL_in: tl.constexpr,    # B * C * H * W (= 4 * N_pool when H,W even)
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,         # input spatial (assume H == W or pass W separately)
    HW_in: tl.constexpr,     # H * W
    Hp: tl.constexpr,        # H // 2
    Wp: tl.constexpr,        # W // 2
    HWp: tl.constexpr,       # Hp * Wp
    BLOCK_NPOL: tl.constexpr,
    decay_factor: tl.constexpr, input_scale: tl.constexpr,
    v_threshold: tl.constexpr, v_reset_val: tl.constexpr,
    HARD_RESET: tl.constexpr, HAS_BIAS: tl.constexpr,
):
    """LIF + AvgPool2x2 fused kernel (NCHW)。每 CTA 处理 BLOCK_NPOL 个 pool 输出位置。

    Pool 输出位置 p ∈ [0, N_pool)。对应输入 (B, C, h_in, w_in)：
       b   = p // (C * Hp * Wp)
       c   = (p // (Hp * Wp)) % C
       hp  = (p // Wp) % Hp
       wp  = p % Wp
       h_in = 2 * hp,  w_in = 2 * wp
       4 个 spike 位置：(h_in, w_in), (h_in, w_in+1), (h_in+1, w_in), (h_in+1, w_in+1)
       flat index = b * (C*H*W) + c * (H*W) + h_in * W + w_in (and the 3 neighbors)
    """
    pid = tl.program_id(0)
    p = pid * BLOCK_NPOL + tl.arange(0, BLOCK_NPOL)
    pmask = p < N_pool

    b = p // (C * HWp)
    c = (p // HWp) % C
    hp = (p // Wp) % Hp
    wp = p % Wp
    h_in = 2 * hp
    w_in = 2 * wp

    # base flat index for the (0,0) of each 2x2 input block
    base = (b * (C * HW_in) + c * HW_in + h_in * W + w_in).to(tl.int64)
    NCL_i64 = tl.full([], NCL_in, dtype=tl.int64)
    # 4 neighbors offsets
    off00 = base
    off01 = base + 1
    off10 = base + W
    off11 = base + W + 1

    if HAS_BIAS:
        bias = tl.load(bias_ptr + c, mask=pmask, other=0.0).to(tl.float32)
    else:
        bias = tl.zeros([BLOCK_NPOL], dtype=tl.float32)

    v00 = tl.zeros([BLOCK_NPOL], dtype=tl.float32)
    v01 = tl.zeros([BLOCK_NPOL], dtype=tl.float32)
    v10 = tl.zeros([BLOCK_NPOL], dtype=tl.float32)
    v11 = tl.zeros([BLOCK_NPOL], dtype=tl.float32)

    for t in tl.static_range(0, T, 1):
        t_off = tl.full([], t, dtype=tl.int64) * NCL_i64
        # 加载 4 个相邻输入位置
        y00 = tl.load(y_ptr + t_off + off00, mask=pmask, other=0.0).to(tl.float32)
        y01 = tl.load(y_ptr + t_off + off01, mask=pmask, other=0.0).to(tl.float32)
        y10 = tl.load(y_ptr + t_off + off10, mask=pmask, other=0.0).to(tl.float32)
        y11 = tl.load(y_ptr + t_off + off11, mask=pmask, other=0.0).to(tl.float32)

        v00 = decay_factor * v00 + input_scale * (y00 + bias)
        v01 = decay_factor * v01 + input_scale * (y01 + bias)
        v10 = decay_factor * v10 + input_scale * (y10 + bias)
        v11 = decay_factor * v11 + input_scale * (y11 + bias)

        s00 = (v00 >= v_threshold).to(tl.float32)
        s01 = (v01 >= v_threshold).to(tl.float32)
        s10 = (v10 >= v_threshold).to(tl.float32)
        s11 = (v11 >= v_threshold).to(tl.float32)

        if HARD_RESET:
            v00 = v00 * (1.0 - s00) + s00 * v_reset_val
            v01 = v01 * (1.0 - s01) + s01 * v_reset_val
            v10 = v10 * (1.0 - s10) + s10 * v_reset_val
            v11 = v11 * (1.0 - s11) + s11 * v_reset_val
        else:
            v00 = v00 - s00 * v_threshold
            v01 = v01 - s01 * v_threshold
            v10 = v10 - s10 * v_threshold
            v11 = v11 - s11 * v_threshold

        pool_t = (s00 + s01 + s10 + s11) * 0.25
        # pool output at (t, b, c, hp, wp)
        pool_off_t = tl.full([], t, dtype=tl.int64) * N_pool + p.to(tl.int64)
        tl.store(pool_ptr + pool_off_t, pool_t, mask=pmask)


def fused_lif_pool2(y_seq, bias, *, tau=2.0, decay_input=True, soft_reset=False,
                     v_threshold=1.0, v_reset=0.0):
    """Conv-BN-LIF-AvgPool2x2 fused 入口。
    y_seq: [T, B, C, H, W] NCHW contiguous
    返回:   [T, B, C, H//2, W//2] (avg pool 输出)
    要求 H 和 W 都是偶数。
    """
    T, B, C, H, W = y_seq.shape
    assert H % 2 == 0 and W % 2 == 0
    Hp, Wp = H // 2, W // 2
    HW_in = H * W
    HWp = Hp * Wp
    N_pool = B * C * Hp * Wp
    NCL_in = B * C * HW_in
    decay = 1.0 - 1.0 / tau
    scale = (1.0 / tau) if decay_input else 1.0
    HAS_BIAS = bias is not None
    bias_arg = bias if HAS_BIAS else y_seq
    pool_out = torch.empty((T, B, C, Hp, Wp), device=y_seq.device, dtype=y_seq.dtype)
    grid = lambda meta: (triton.cdiv(N_pool, meta["BLOCK_NPOL"]),)
    _bias_lif_pool2_kernel[grid](
        y_seq, bias_arg, pool_out,
        T=T, N_pool=N_pool, NCL_in=NCL_in,
        C=C, H=H, W=W, HW_in=HW_in,
        Hp=Hp, Wp=Wp, HWp=HWp,
        decay_factor=decay, input_scale=scale,
        v_threshold=v_threshold, v_reset_val=v_reset,
        HARD_RESET=(not soft_reset), HAS_BIAS=HAS_BIAS,
    )
    return pool_out


# ============================================================
# verify + bench
# ============================================================
def naive_lif_pool2(y_seq, bias, *, tau, decay_input, soft_reset, v_threshold, v_reset):
    import torch.nn.functional as F
    T = y_seq.shape[0]
    decay = 1.0 - 1.0 / tau
    scale = (1.0 / tau) if decay_input else 1.0
    bb = bias.float().view(1, -1, 1, 1) if bias is not None else 0.0
    v = torch.zeros_like(y_seq[0], dtype=torch.float32)
    pools = []
    for t in range(T):
        v = decay * v + scale * (y_seq[t].float() + bb)
        spk = (v >= v_threshold).float()
        if soft_reset:
            v = v - spk * v_threshold
        else:
            v = torch.where(spk > 0, torch.full_like(v, v_reset), v)
        p = F.avg_pool2d(spk, 2, 2)
        pools.append(p)
    return torch.stack(pools, 0)


def main():
    print("=== verify: fused LIF+Pool vs naive LIF→Pool ===")
    torch.manual_seed(0)
    T, B, C, H, W = 16, 2, 16, 14, 14
    y = torch.randn(T, B, C, H, W, device='cuda').contiguous()
    bias = torch.randn(C, device='cuda')
    for soft in [True, False]:
        ref = naive_lif_pool2(y, bias, tau=2.0, decay_input=True, soft_reset=soft,
                                v_threshold=1.0, v_reset=0.0)
        out = fused_lif_pool2(y, bias, tau=2.0, decay_input=True, soft_reset=soft,
                                v_threshold=1.0, v_reset=0.0)
        eq = torch.equal(ref, out)
        max_diff = (ref - out).abs().max().item()
        print(f"  {'soft' if soft else 'hard'}: bit-equal={eq} max|diff|={max_diff:.3e}")

    print("\n=== timing: LIF only / LIF+pool (separated) / LIF+pool (fused) ===")
    import time, torch.nn.functional as F
    from snn_compiler.kernels.fused import fused_bias_if_lif

    def time_ms(fn, n=50):
        for _ in range(5): fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n): fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000 / n

    for T in [4, 16, 64, 128]:
        B, C, H, W = 16, 64, 56, 56
        y = torch.randn(T, B, C, H, W, device='cuda', dtype=torch.bfloat16).contiguous()
        bias = torch.randn(C, device='cuda')

        def lif_only():
            return fused_bias_if_lif(y, bias, neuron='lif', tau=2.0, decay_input=True,
                                        soft_reset=False, v_threshold=1.0, v_reset=0.0)
        t_lif = time_ms(lif_only)

        def lif_then_pool():
            spike = fused_bias_if_lif(y, bias, neuron='lif', tau=2.0, decay_input=True,
                                         soft_reset=False, v_threshold=1.0, v_reset=0.0)
            T_, B_, C_, H_, W_ = spike.shape
            return F.avg_pool2d(spike.reshape(T_ * B_, C_, H_, W_), 2, 2)
        t_lif_pool = time_ms(lif_then_pool)

        def fused():
            return fused_lif_pool2(y, bias, tau=2.0, decay_input=True,
                                       soft_reset=False, v_threshold=1.0, v_reset=0.0)
        t_fused = time_ms(fused)

        print(f"T={T:>3d}  LIF only={t_lif:.3f}ms  LIF+pool={t_lif_pool:.3f}ms  "
              f"fused={t_fused:.3f}ms  "
              f"saved {(t_lif_pool - t_fused):.3f}ms ({t_lif_pool/t_fused:.2f}×)")


if __name__ == "__main__":
    main()
