"""NHWC 版本 FusedBiasIF kernel：input 是 channels_last 内存布局。

针对 channels_last 内存布局 [T, B, H, W, C]（c 维在最内层）：
- bias index = ncl_idx % C  (大大简化于 NCHW 版的 (ncl_idx // HW) % C)
- 其余逻辑与 fused_bias_if_kernel.py 等价
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": 128}, num_warps=4),
        triton.Config({"BLOCK_NCL": 256}, num_warps=4),
        triton.Config({"BLOCK_NCL": 256}, num_warps=8),
        triton.Config({"BLOCK_NCL": 512}, num_warps=8),
    ],
    key=["T", "NCL", "C", "SOFT_RESET"],
)
@triton.jit
def _fused_bias_if_nhwc_kernel(
    x_ptr,                       # [T, B*H*W*C] flat (内存中按 channels_last 排列)
    bias_ptr,                    # [C]
    spike_ptr,                   # [T, B*H*W*C] flat (同上排列)
    T: tl.constexpr,
    NCL: tl.constexpr,           # = B*C*H*W (numel per timestep)
    C: tl.constexpr,             # 通道数
    BLOCK_NCL: tl.constexpr,
    v_threshold: tl.constexpr,
    SOFT_RESET: tl.constexpr,
):
    pid = tl.program_id(0)
    ncl_idx = pid * BLOCK_NCL + tl.arange(0, BLOCK_NCL)
    mask = ncl_idx < NCL

    # NHWC 布局下 c 是最内层，bias index = ncl_idx % C
    c_idx = ncl_idx % C
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    v = tl.zeros([BLOCK_NCL], dtype=tl.float32)

    for t in tl.static_range(0, T, 1):
        x_t = tl.load(x_ptr + t * NCL + ncl_idx, mask=mask, other=0.0)
        v = v + x_t + bias
        spike = (v >= v_threshold).to(tl.float32)
        if SOFT_RESET:
            v = v - spike * v_threshold
        else:
            v = v * (1.0 - spike)
        tl.store(spike_ptr + t * NCL + ncl_idx, spike, mask=mask)


def fused_bias_if_nhwc(x_seq_nhwc: torch.Tensor, bias: torch.Tensor,
                       soft_reset: bool = False, v_threshold: float = 1.0) -> torch.Tensor:
    """x_seq_nhwc: [T, B, C, H, W] but memory is channels_last (T·B·H·W·C order).
    bias: [C]
    返回 spike: 同 x_seq_nhwc 的 shape + 同 channels_last memory layout。
    """
    assert x_seq_nhwc.is_cuda and x_seq_nhwc.dtype == torch.float32
    T, B, C, H, W = x_seq_nhwc.shape
    NCL = B * C * H * W
    # 用 channels_last 4D 然后 unsqueeze 不直接行；这里我们假设 caller 保证内存按 [T,B,H,W,C] 排列
    # （把 4D channels_last tensor view 到 5D 时 stride 由 PyTorch 维护，flat index 顺序与内存一致）
    spike_seq = torch.empty_like(x_seq_nhwc)
    grid = lambda meta: (triton.cdiv(NCL, meta["BLOCK_NCL"]),)
    _fused_bias_if_nhwc_kernel[grid](
        x_seq_nhwc, bias, spike_seq,
        T=T, NCL=NCL, C=C,
        v_threshold=v_threshold,
        SOFT_RESET=soft_reset,
    )
    return spike_seq


def _selftest():
    """正确性验证：NHWC 实现应与 (NCHW + permute) 朴素参考逐位一致。"""
    torch.manual_seed(0)
    T, B, C, H, W = 4, 4, 32, 28, 28
    # 标准 NCHW 输入 + bias
    x_nchw = torch.randn(T, B, C, H, W, device="cuda")
    bias = torch.randn(C, device="cuda") * 0.3

    # 参考：在 NCHW 下做朴素 IF
    x_with_bias = x_nchw + bias.view(1, 1, C, 1, 1)
    v = torch.zeros_like(x_nchw[0])
    ref_nchw = []
    for t in range(T):
        v = v + x_with_bias[t]
        sp = (v >= 1.0).float()
        ref_nchw.append(sp)
        v = v * (1.0 - sp)
    ref_nchw = torch.stack(ref_nchw, dim=0)

    # 测试：把 x 转 channels_last，调 NHWC kernel
    # 4D channels_last view 是 PyTorch 支持的；对于 5D 我们用 reshape 配合 stride 处理
    # 简单方案：把 T 维 flatten 到 batch，做 4D channels_last
    x_4d_nchw = x_nchw.flatten(0, 1)                                     # [T*B, C, H, W]
    x_4d_cl   = x_4d_nchw.to(memory_format=torch.channels_last)          # 内存改成 [T*B, H, W, C]
    # view 回 5D 时 stride 由 PyTorch 自动推
    # 但 _kernel 读 flat 内存，按 [T*B*H*W*C] 顺序处理 —— 等价于按 [T, B*H*W*C] 切（C 最内层）
    spike_4d_cl = torch.empty_like(x_4d_cl)
    # 直接调底层 kernel 不经过 view，避免 stride 问题
    NCL = B * C * H * W
    grid = lambda meta: (triton.cdiv(NCL, meta["BLOCK_NCL"]),)
    _fused_bias_if_nhwc_kernel[grid](
        x_4d_cl, bias, spike_4d_cl,
        T=T, NCL=NCL, C=C,
        v_threshold=1.0, SOFT_RESET=False,
    )
    # spike_4d_cl 仍是 channels_last 内存
    # 转回 NCHW 后与 ref_nchw 比对
    spike_4d_nchw = spike_4d_cl.contiguous()
    spike_nchw = spike_4d_nchw.view(T, B, C, H, W)
    diff = (ref_nchw != spike_nchw).sum().item()
    total = ref_nchw.numel()
    print(f"  shape=(T={T},B={B},C={C},H={H},W={W})  diff={diff}/{total}  bit-equal={diff == 0}")


if __name__ == "__main__":
    _selftest()
