"""候选 kernel 变体，与 baseline `_bias_if_lif_kernel` 对照。

V1: runtime-loop (`tl.range`) 替代 `tl.static_range`
    - 优点：T 大时 IR 不再爆炸；PTX 小、icache 友好
    - 风险：T 小时少了 unroll 的 ILP，可能更慢
    - 进一步：`tl.range(..., loop_unroll_factor=k)` 让编译器按 k 部分展开

V2: chunked rate-output（"poisson 累计"模式）
    - 不存储 [T, NCL] spike，只存储 [NCL] 累积 spike-count
    - 写带宽 ÷ T，T=128 时省 99%
    - 限制：下游需要 rate code（最后一层分类器适用；中间 conv 层不适用）

V3: persistent CTA tile (CHUNK_NCL = BLOCK_NCL × K)
    - 一个 CTA 处理 K 个 NCL tile，bias / threshold 加载共享
    - T 大时算术强度提升，bandwidth 内 v_th broadcast 不再每 CTA 重做
    - 代价：v 寄存器 × K 倍

每个 variant 都做与 baseline 的 bit-equal 验证 + 同形 shape 计时。
"""
import sys, pathlib, time, statistics

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch
import triton
import triton.language as tl

from snn_compiler.kernels.fused import fused_bias_if_lif


# ============================================================
#   V1: runtime-loop kernel
# ============================================================
@triton.autotune(
    configs=[
        # 简化：固定 UNROLL=1, num_stages=1，先排除 pipelining bug
        triton.Config({"BLOCK_NCL": 128, "UNROLL": 1},  num_warps=4, num_stages=1),
        triton.Config({"BLOCK_NCL": 256, "UNROLL": 1},  num_warps=4, num_stages=1),
        triton.Config({"BLOCK_NCL": 512, "UNROLL": 1},  num_warps=8, num_stages=1),
        triton.Config({"BLOCK_NCL": 1024, "UNROLL": 1}, num_warps=8, num_stages=1),
    ],
    key=["T", "NCL", "HAS_BIAS"],
)
@triton.jit
def _bias_lif_runtime_loop(
    y_ptr, bias_ptr, spike_ptr,
    T: tl.constexpr, NCL: tl.constexpr, C: tl.constexpr, HW: tl.constexpr,
    BLOCK_NCL: tl.constexpr, UNROLL: tl.constexpr,
    decay_factor: tl.constexpr, input_scale: tl.constexpr,
    v_threshold: tl.constexpr, v_reset_val: tl.constexpr,
    HARD_RESET: tl.constexpr, HAS_BIAS: tl.constexpr,
    CHANNEL_LAST: tl.constexpr,
):
    pid = tl.program_id(0)
    ncl_idx = pid.to(tl.int64) * BLOCK_NCL + tl.arange(0, BLOCK_NCL).to(tl.int64)
    mask = ncl_idx < NCL
    NCL_i64 = tl.full([], NCL, dtype=tl.int64)

    if HAS_BIAS:
        if CHANNEL_LAST:
            c_idx = ncl_idx % C
        else:
            c_idx = (ncl_idx // HW) % C
        bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    else:
        bias = tl.zeros([BLOCK_NCL], dtype=tl.float32)

    v = tl.zeros([BLOCK_NCL], dtype=tl.float32)

    # ★ runtime loop with loop_unroll_factor
    for t in tl.range(0, T, 1, loop_unroll_factor=UNROLL):
        t_off = t.to(tl.int64) * NCL_i64
        y_t = tl.load(y_ptr + t_off + ncl_idx, mask=mask, other=0.0).to(tl.float32)
        v = decay_factor * v + input_scale * (y_t + bias)
        spike = (v >= v_threshold).to(tl.float32)
        if HARD_RESET:
            v = v * (1.0 - spike) + spike * v_reset_val
        else:
            v = v - spike * v_threshold
        tl.store(spike_ptr + t_off + ncl_idx, spike, mask=mask)


def fused_lif_runtime(y_seq, bias, *, tau=2.0, decay_input=True, soft_reset=False,
                       v_threshold=1.0, v_reset=0.0, layout="NCHW"):
    """V1 entry point: 同 fused_bias_if_lif 接口子集（仅 LIF + scalar threshold）"""
    assert y_seq.is_cuda
    T = y_seq.shape[0]
    NCL = y_seq[0].numel()
    if y_seq.ndim == 5:
        C, HW = y_seq.shape[2], y_seq.shape[3] * y_seq.shape[4]
    elif y_seq.ndim == 4:
        C, HW = y_seq.shape[2], y_seq.shape[3]
    elif y_seq.ndim == 3:
        C, HW = y_seq.shape[-1], 1
    else:
        C, HW = y_seq[0].numel(), 1
    decay = 1.0 - 1.0 / tau
    scale = (1.0 / tau) if decay_input else 1.0
    HAS_BIAS = bias is not None
    bias_arg = bias if HAS_BIAS else y_seq
    spike = torch.empty_like(y_seq)
    grid = lambda meta: (triton.cdiv(NCL, meta["BLOCK_NCL"]),)
    _bias_lif_runtime_loop[grid](
        y_seq, bias_arg, spike,
        T=T, NCL=NCL, C=C, HW=HW,
        decay_factor=decay, input_scale=scale,
        v_threshold=v_threshold, v_reset_val=v_reset,
        HARD_RESET=(not soft_reset), HAS_BIAS=HAS_BIAS,
        CHANNEL_LAST=(layout == "NHWC"),
    )
    return spike


# ============================================================
#   V2: rate-output kernel (sum-over-T spike count)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": 256}, num_warps=4),
        triton.Config({"BLOCK_NCL": 512}, num_warps=8),
        triton.Config({"BLOCK_NCL": 1024}, num_warps=8),
    ],
    key=["T", "NCL", "HAS_BIAS"],
)
@triton.jit
def _bias_lif_rate_count(
    y_ptr, bias_ptr, count_ptr,
    T: tl.constexpr, NCL: tl.constexpr, C: tl.constexpr, HW: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    decay_factor: tl.constexpr, input_scale: tl.constexpr,
    v_threshold: tl.constexpr, v_reset_val: tl.constexpr,
    HARD_RESET: tl.constexpr, HAS_BIAS: tl.constexpr,
    CHANNEL_LAST: tl.constexpr,
):
    """只写 [NCL] 的 spike-count；T 全过流，不存 per-t spike。

    带宽：原版每 (T,NCL) 写 1 byte (bf16 spike) = T*NCL 字节
          rate 版只写 NCL 字节 (fp32 count) = NCL*4 字节
          ratio: T/4，T=128 时省 31×；T=4 时反而多 1×（注意 use case）
    """
    pid = tl.program_id(0)
    ncl_idx = pid.to(tl.int64) * BLOCK_NCL + tl.arange(0, BLOCK_NCL).to(tl.int64)
    mask = ncl_idx < NCL
    NCL_i64 = tl.full([], NCL, dtype=tl.int64)

    if HAS_BIAS:
        if CHANNEL_LAST:
            c_idx = ncl_idx % C
        else:
            c_idx = (ncl_idx // HW) % C
        bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    else:
        bias = tl.zeros([BLOCK_NCL], dtype=tl.float32)

    v = tl.zeros([BLOCK_NCL], dtype=tl.float32)
    count = tl.zeros([BLOCK_NCL], dtype=tl.float32)

    for t in tl.static_range(0, T, 1):
        t_off = tl.full([], t, dtype=tl.int64) * NCL_i64
        y_t = tl.load(y_ptr + t_off + ncl_idx, mask=mask, other=0.0).to(tl.float32)
        v = decay_factor * v + input_scale * (y_t + bias)
        spike = (v >= v_threshold).to(tl.float32)
        if HARD_RESET:
            v = v * (1.0 - spike) + spike * v_reset_val
        else:
            v = v - spike * v_threshold
        count = count + spike
    tl.store(count_ptr + ncl_idx, count, mask=mask)


def fused_lif_rate(y_seq, bias, *, tau=2.0, decay_input=True, soft_reset=False,
                    v_threshold=1.0, v_reset=0.0, layout="NCHW"):
    """V2 entry point: 输出 [B, C, H, W] spike-count（fp32）。"""
    assert y_seq.is_cuda
    T = y_seq.shape[0]
    NCL = y_seq[0].numel()
    if y_seq.ndim == 5:
        C, HW = y_seq.shape[2], y_seq.shape[3] * y_seq.shape[4]
    elif y_seq.ndim == 4:
        C, HW = y_seq.shape[2], y_seq.shape[3]
    elif y_seq.ndim == 3:
        C, HW = y_seq.shape[-1], 1
    else:
        C, HW = y_seq[0].numel(), 1
    decay = 1.0 - 1.0 / tau
    scale = (1.0 / tau) if decay_input else 1.0
    HAS_BIAS = bias is not None
    bias_arg = bias if HAS_BIAS else y_seq
    count = torch.empty(y_seq.shape[1:], device=y_seq.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(NCL, meta["BLOCK_NCL"]),)
    _bias_lif_rate_count[grid](
        y_seq, bias_arg, count,
        T=T, NCL=NCL, C=C, HW=HW,
        decay_factor=decay, input_scale=scale,
        v_threshold=v_threshold, v_reset_val=v_reset,
        HARD_RESET=(not soft_reset), HAS_BIAS=HAS_BIAS,
        CHANNEL_LAST=(layout == "NHWC"),
    )
    return count


# ============================================================
#   Validation
# ============================================================
def naive_lif(y_seq, bias, *, tau, decay_input, soft_reset, v_threshold, v_reset):
    T = y_seq.shape[0]
    decay = 1.0 - 1.0 / tau
    scale = (1.0 / tau) if decay_input else 1.0
    # broadcast 必须与 y_seq[t] 同 rank（4D），否则 PyTorch 会把结果升到 5D
    if bias is not None:
        broadcast = bias.float().view(1, -1, 1, 1)
    else:
        broadcast = 0.0
    v = torch.zeros_like(y_seq[0], dtype=torch.float32)
    spikes = []
    for t in range(T):
        v = decay * v + scale * (y_seq[t].float() + broadcast)
        spike = (v >= v_threshold).float()
        spikes.append(spike)
        if soft_reset:
            v = v - spike * v_threshold
        else:
            v = torch.where(spike > 0, torch.full_like(v, v_reset), v)
    return torch.stack(spikes, dim=0)


def verify():
    print("\n=== verify V1 (runtime-loop) bit-equal vs naive ===")
    torch.manual_seed(0)
    T, B, C, H, W = 16, 2, 32, 14, 14
    y = torch.randn(T, B, C, H, W, device='cuda').contiguous()
    bias = torch.randn(C, device='cuda')
    for soft in [True, False]:
        ref = naive_lif(y, bias, tau=2.0, decay_input=True,
                        soft_reset=soft, v_threshold=1.0, v_reset=0.0)
        out = fused_lif_runtime(y, bias, tau=2.0, decay_input=True,
                                  soft_reset=soft, v_threshold=1.0, v_reset=0.0)
        eq = torch.equal(ref, out)
        max_diff = (ref - out).abs().max().item()
        n_diff = (ref != out).sum().item()
        same_spikes = ((ref > 0) == (out > 0)).all().item()
        print(f"  {'soft' if soft else 'hard'}: bit-equal={eq}  same-spikes={same_spikes}  "
              f"max|diff|={max_diff:.3e}  numel-diff={n_diff}/{ref.numel()}")

    print("\n=== verify V2 (rate-count) sum-over-T vs naive ===")
    for soft in [True, False]:
        ref = naive_lif(y, bias, tau=2.0, decay_input=True,
                        soft_reset=soft, v_threshold=1.0, v_reset=0.0)
        ref_count = ref.sum(dim=0)        # [B, C, H, W]
        out = fused_lif_rate(y, bias, tau=2.0, decay_input=True,
                              soft_reset=soft, v_threshold=1.0, v_reset=0.0)
        eq = torch.equal(ref_count, out)
        print(f"  {'soft' if soft else 'hard'}: bit-equal={eq}  max|diff|="
              f"{(ref_count - out).abs().max().item():.3e}")


def time_ms(fn, n_warm=5, n_iter=50):
    for _ in range(n_warm): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / n_iter


def bench():
    print("\n=== A/B benchmark: baseline vs V1 vs V2 ===")
    print(f"{'T':>4s} {'B':>3s} {'NCL/k':>7s} {'baseline ms':>12s} "
          f"{'V1 runtime ms':>14s} {'V2 rate ms':>11s} "
          f"{'V1 spd':>7s} {'V2 spd':>7s}")
    print("-" * 75)
    for T in [4, 16, 64, 128]:
        for B, C, H, W in [(16, 64, 56, 56), (16, 256, 14, 14)]:
            NCL = B * C * H * W
            try:
                y = torch.randn(T, B, C, H, W, device='cuda', dtype=torch.bfloat16).contiguous()
            except torch.AcceleratorError:
                print(f"{T:>4d} {B:>3d} {NCL//1000:>7d} OOM")
                continue
            bias = torch.randn(C, device='cuda')
            # baseline
            def base(): return fused_bias_if_lif(y, bias, neuron='lif', tau=2.0,
                                                    decay_input=True, soft_reset=False,
                                                    v_threshold=1.0, v_reset=0.0, layout='NCHW')
            t_base = time_ms(base)
            def v1(): return fused_lif_runtime(y, bias, tau=2.0, decay_input=True,
                                                  soft_reset=False, v_threshold=1.0, v_reset=0.0)
            t_v1 = time_ms(v1)
            def v2(): return fused_lif_rate(y, bias, tau=2.0, decay_input=True,
                                               soft_reset=False, v_threshold=1.0, v_reset=0.0)
            t_v2 = time_ms(v2)
            print(f"{T:>4d} {B:>3d} {NCL//1000:>7d} {t_base:>12.4f} "
                  f"{t_v1:>14.4f} {t_v2:>11.4f} "
                  f"{t_base/t_v1:>6.2f}× {t_base/t_v2:>6.2f}×")
            del y, bias
            torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    verify()
    bench()
