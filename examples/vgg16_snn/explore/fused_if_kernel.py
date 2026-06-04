"""手工融合 IF Triton kernel：消除 Inductor autogen 在 T 维上的冗余 load/compute。

Inductor 把 PrefixSumIF 的多步 IF 逻辑融成一个 output-flat-parallel kernel，
导致每个时间步的 thread 独立从 t=0 重算前置状态（详见 mlir-perf-exploration-journal.md Step 1）。

本 kernel 改成 SJ multistep_lif 风格：1D grid 沿 NCL 并行，每 thread 在 register 里
做 T=4 顺序累加，无跨步冗余。提供 soft 与 hard 两种 reset 语义。

API:
    fused_if_kernel(x_seq, soft_reset=False, v_threshold=1.0) -> spike_seq

    输入 x_seq: [T, B, ...]  (任意形状的尾部，会 flatten 成 NCL)
    输出 spike_seq: 同形状

测试：与朴素 Python 实现 bit-equal（fp32 范围内）。
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": 64}, num_warps=2),
        triton.Config({"BLOCK_NCL": 128}, num_warps=4),
        triton.Config({"BLOCK_NCL": 256}, num_warps=4),
        triton.Config({"BLOCK_NCL": 256}, num_warps=8),
        triton.Config({"BLOCK_NCL": 512}, num_warps=8),
    ],
    key=["T", "NCL", "SOFT_RESET"],
)
@triton.jit
def _fused_if_forward_kernel(
    x_ptr,                  # [T, NCL]
    spike_ptr,              # [T, NCL]
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    v_threshold: tl.constexpr,
    SOFT_RESET: tl.constexpr,
):
    """Soft-reset: spike[t] 触发后 v -= threshold（excess 累加）
       Hard-reset: spike[t] 触发后 v = 0（excess 丢弃）

    1D grid 沿 NCL 并行；T 在编译期完全展开（tl.static_range）。
    """
    pid = tl.program_id(0)
    ncl_offset = pid * BLOCK_NCL
    ncl_idx = ncl_offset + tl.arange(0, BLOCK_NCL)
    mask = ncl_idx < NCL

    # 状态变量：v 全程驻留 register
    v = tl.zeros([BLOCK_NCL], dtype=tl.float32)

    for t in tl.static_range(0, T, 1):
        # 加载 x[t] 并累加到 v
        x_t = tl.load(x_ptr + t * NCL + ncl_idx, mask=mask, other=0.0)
        v = v + x_t
        spike = (v >= v_threshold).to(tl.float32)
        # Reset
        if SOFT_RESET:
            v = v - spike * v_threshold     # soft: 减阈值，excess 留下
        else:
            v = v * (1.0 - spike)           # hard: 清零
        tl.store(spike_ptr + t * NCL + ncl_idx, spike, mask=mask)


def fused_if(x_seq: torch.Tensor, soft_reset: bool = False, v_threshold: float = 1.0) -> torch.Tensor:
    """x_seq: [T, B, ...], returns same shape spike_seq.

    Args:
        soft_reset: True 用 soft-reset（v -= threshold）；False 用 hard-reset (v = 0)
        v_threshold: 发放阈值
    """
    assert x_seq.is_cuda and x_seq.dtype == torch.float32
    assert x_seq.is_contiguous()
    T = x_seq.shape[0]
    NCL = x_seq[0].numel()
    spike_seq = torch.empty_like(x_seq)
    grid = lambda meta: (triton.cdiv(NCL, meta["BLOCK_NCL"]),)
    _fused_if_forward_kernel[grid](
        x_seq, spike_seq, T=T, NCL=NCL,
        v_threshold=v_threshold,
        SOFT_RESET=soft_reset,
    )
    return spike_seq


# -------------- 朴素 PyTorch 参考实现，用于正确性比对 --------------
def naive_if(x_seq, soft_reset=False, v_threshold=1.0):
    """与 fused_if 等价的 Python 顺序实现，用于 bit-equal 比对。"""
    T = x_seq.shape[0]
    v = torch.zeros_like(x_seq[0])
    spikes = []
    for t in range(T):
        v = v + x_seq[t]
        spike = (v >= v_threshold).to(x_seq.dtype)
        spikes.append(spike)
        if soft_reset:
            v = v - spike * v_threshold
        else:
            v = v * (1 - spike)
    return torch.stack(spikes, dim=0)


# -------------- 正确性 sanity check --------------
def _selftest():
    torch.manual_seed(0)
    for shape in [(4, 16), (4, 1024), (4, 32, 64, 224, 224)]:
        for soft in [True, False]:
            x = torch.randn(*shape, device="cuda")
            x = x.contiguous()
            r_naive = naive_if(x, soft_reset=soft, v_threshold=1.0)
            r_fused = fused_if(x, soft_reset=soft, v_threshold=1.0)
            diff = (r_naive != r_fused).sum().item()
            total = r_naive.numel()
            tag = "soft" if soft else "hard"
            print(f"  shape={shape}  reset={tag}  diff={diff}/{total}  bit-equal={diff == 0}")


if __name__ == "__main__":
    _selftest()
