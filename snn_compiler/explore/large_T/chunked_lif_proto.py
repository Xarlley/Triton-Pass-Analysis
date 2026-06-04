"""prototype: LIF kernel 接受 v_init 与可选返回 v_final，启用 T-chunked 执行。

数学不变：
    v_t = decay * v_{t-1} + scale * (x_t + bias)
    spike_t = (v_t >= v_th)
    v_t = (hard: where(spike, v_reset, v_t); soft: v_t - spike * v_th)

修改：
- Python 入口接受 v_init: torch.Tensor 与 v_init 同 shape 的 [B, C, H, W]
  作为 chunk 起点；不传则用 0
- Kernel 在 t=0 之前先 tl.load v_init（HAS_VINIT 时）
- Kernel 在 t=T-1 处把 v_final tl.store 到 v_out（HAS_VOUT 时）

整体 chunked 驱动：把网络的 [T, B, C, H, W] 拆成多个 [chunk, B, C, H, W]，
每个 LIF 模块保存自己的 v 状态，多 chunk 顺序前向。这样：
- 每层 per-chunk 显存 = T/chunks 倍峰值
- 跨 chunk 数据流为：spike_chunk → next layer → final → discard
- 各 LIF 持 v 状态（每层一个 [B, C, H, W] fp32 张量），总额外开销 ~ 1× 单步激活

T=128, chunk=16 时，单层瞬时显存 ≈ T=16 时的水平，让 VGG-16 在 16GiB 卡上跑得动。
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": 256}, num_warps=4),
        triton.Config({"BLOCK_NCL": 512}, num_warps=8),
        triton.Config({"BLOCK_NCL": 1024}, num_warps=8),
    ],
    key=["T", "NCL", "HAS_BIAS", "HAS_VINIT", "HAS_VOUT"],
)
@triton.jit
def _bias_lif_with_state(
    y_ptr, bias_ptr, spike_ptr, v_init_ptr, v_out_ptr,
    T: tl.constexpr, NCL: tl.constexpr, C: tl.constexpr, HW: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    decay_factor: tl.constexpr, input_scale: tl.constexpr,
    v_threshold: tl.constexpr, v_reset_val: tl.constexpr,
    HARD_RESET: tl.constexpr,
    HAS_BIAS: tl.constexpr, HAS_VINIT: tl.constexpr, HAS_VOUT: tl.constexpr,
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

    if HAS_VINIT:
        v = tl.load(v_init_ptr + ncl_idx, mask=mask, other=0.0).to(tl.float32)
    else:
        v = tl.zeros([BLOCK_NCL], dtype=tl.float32)

    for t in tl.static_range(0, T, 1):
        t_off = tl.full([], t, dtype=tl.int64) * NCL_i64
        y_t = tl.load(y_ptr + t_off + ncl_idx, mask=mask, other=0.0).to(tl.float32)
        v = decay_factor * v + input_scale * (y_t + bias)
        spike = (v >= v_threshold).to(tl.float32)
        if HARD_RESET:
            v = v * (1.0 - spike) + spike * v_reset_val
        else:
            v = v - spike * v_threshold
        tl.store(spike_ptr + t_off + ncl_idx, spike, mask=mask)

    if HAS_VOUT:
        tl.store(v_out_ptr + ncl_idx, v, mask=mask)


def fused_lif_with_state(y_seq, bias, *, v_init=None, return_v=True,
                          tau=2.0, decay_input=True, soft_reset=False,
                          v_threshold=1.0, v_reset=0.0, layout="NCHW"):
    """带 v_init/v_final 的 LIF 入口；返回 (spike_seq, v_final) 或 spike_seq。"""
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
    HAS_VINIT = v_init is not None
    v_init_arg = v_init if HAS_VINIT else y_seq
    if HAS_VINIT:
        assert v_init.numel() == NCL and v_init.dtype == torch.float32
        v_init_arg = v_init.reshape(-1)
    HAS_VOUT = return_v
    v_out = torch.empty(y_seq.shape[1:], device=y_seq.device, dtype=torch.float32) \
            if HAS_VOUT else y_seq
    spike = torch.empty_like(y_seq)
    grid = lambda meta: (triton.cdiv(NCL, meta["BLOCK_NCL"]),)
    _bias_lif_with_state[grid](
        y_seq, bias_arg, spike, v_init_arg, v_out if HAS_VOUT else y_seq,
        T=T, NCL=NCL, C=C, HW=HW,
        decay_factor=decay, input_scale=scale,
        v_threshold=v_threshold, v_reset_val=v_reset,
        HARD_RESET=(not soft_reset), HAS_BIAS=HAS_BIAS,
        HAS_VINIT=HAS_VINIT, HAS_VOUT=HAS_VOUT,
        CHANNEL_LAST=(layout == "NHWC"),
    )
    if HAS_VOUT:
        return spike, v_out
    return spike


# ============================================================
#   chunked-driver microbench: T-split 单层等价性 + 性能
# ============================================================
def naive_lif(y_seq, bias, *, tau, decay_input, soft_reset, v_threshold, v_reset):
    T = y_seq.shape[0]
    decay = 1.0 - 1.0 / tau
    scale = (1.0 / tau) if decay_input else 1.0
    broadcast = bias.float().view(1, -1, 1, 1) if bias is not None else 0.0
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


def chunked_lif(y_seq, bias, chunk, *, tau, decay_input, soft_reset, v_threshold, v_reset):
    """T 切成多 chunk 顺序前向，使用 fused_lif_with_state；保证与 naive 等价。"""
    T = y_seq.shape[0]
    chunks = []
    v = None
    for i in range(0, T, chunk):
        c = min(chunk, T - i)
        y_c = y_seq[i:i + c].contiguous()
        spike_c, v = fused_lif_with_state(
            y_c, bias, v_init=v, return_v=True,
            tau=tau, decay_input=decay_input, soft_reset=soft_reset,
            v_threshold=v_threshold, v_reset=v_reset,
        )
        chunks.append(spike_c)
    return torch.cat(chunks, dim=0)


def main():
    print("=== verify: chunked vs naive bit-equal ===")
    torch.manual_seed(0)
    T, B, C, H, W = 32, 2, 16, 14, 14
    y = torch.randn(T, B, C, H, W, device='cuda').contiguous()
    bias = torch.randn(C, device='cuda')
    for chunk in [4, 8, 16, 32]:
        for soft in [True, False]:
            ref = naive_lif(y, bias, tau=2.0, decay_input=True, soft_reset=soft,
                            v_threshold=1.0, v_reset=0.0)
            out = chunked_lif(y, bias, chunk, tau=2.0, decay_input=True,
                                soft_reset=soft, v_threshold=1.0, v_reset=0.0)
            eq = torch.equal(ref, out)
            max_diff = (ref - out).abs().max().item()
            print(f"  chunk={chunk} {'soft' if soft else 'hard'}: bit-equal={eq} "
                  f"max|diff|={max_diff:.3e}")

    # 一次 chunked-LIF 启动比一次满 T 慢吗？
    print("\n=== timing: full-T vs chunked ===")
    import time
    def time_ms(fn, n=50):
        for _ in range(5): fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n): fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000 / n

    T, B, C, H, W = 128, 16, 64, 56, 56
    y = torch.randn(T, B, C, H, W, device='cuda', dtype=torch.bfloat16).contiguous()
    bias = torch.randn(C, device='cuda')
    from snn_compiler.kernels.fused import fused_bias_if_lif

    def full_T():
        return fused_bias_if_lif(y, bias, neuron='lif', tau=2.0, decay_input=True,
                                    soft_reset=False, v_threshold=1.0, v_reset=0.0)
    t_full = time_ms(full_T)
    print(f"  full T={T}:        {t_full:.3f} ms")
    for chunk in [16, 32, 64]:
        def ck():
            return chunked_lif(y, bias, chunk, tau=2.0, decay_input=True,
                                 soft_reset=False, v_threshold=1.0, v_reset=0.0)
        t_ck = time_ms(ck)
        print(f"  chunk={chunk}:           {t_ck:.3f} ms   ({t_full/t_ck:.2f}× vs full)")


if __name__ == "__main__":
    main()
