"""融合 conv 输出的 bias add + IF 神经元于一个 Triton kernel。

输入:
  conv_out: [T, B, C, H, W]   no-bias conv 的输出
  bias:     [C]                conv layer 的 bias 向量
输出:
  spike:    [T, B, C, H, W]

替代原来的「conv+bias (cuDNN/aten) → IF」两 kernel 链路。
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
    key=["T", "NCL", "HW", "SOFT_RESET"],
)
@triton.jit
def _fused_bias_if_kernel(
    x_ptr,                       # [T, NCL] = [T, B*C*H*W]，但要 broadcast bias[c]
    bias_ptr,                    # [C]
    spike_ptr,                   # [T, NCL]
    T: tl.constexpr,
    NCL: tl.constexpr,
    HW: tl.constexpr,            # H*W，用于从 NCL 索引推回 c
    BCHW_OVER_C: tl.constexpr,   # = C，用于 bias broadcast (modulo C)
    BLOCK_NCL: tl.constexpr,
    v_threshold: tl.constexpr,
    SOFT_RESET: tl.constexpr,
):
    """每个 program 处理 NCL 维上 BLOCK_NCL 个元素，T 步全展开。

    bias broadcast: bias 形状是 [C]，但 x[t] 形状是 [B, C, H, W]，c 维在中间。
    我们用 flatten(B, C, H, W) 后的索引推 c：(ncl // HW) % C。
    """
    pid = tl.program_id(0)
    ncl_offset = pid * BLOCK_NCL
    ncl_idx = ncl_offset + tl.arange(0, BLOCK_NCL)
    mask = ncl_idx < NCL

    # 推 c：ncl 索引内 c 的偏移 = (ncl // HW) % C
    # NCL = B * C * H * W; 每 B*H*W 个连续元素是同一个 c
    # 改用 (ncl // HW) % C
    c_idx = (ncl_idx // HW) % BCHW_OVER_C   # 注意 BCHW_OVER_C 实际是 C（命名歧义，下面修正）
    # 上面 modulo 错了。正确：x[B, C, H, W] flatten 后第 i 个元素的 (b, c, h, w)：
    #   b = i // (C * H * W)
    #   c = (i // (H * W)) % C
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    v = tl.zeros([BLOCK_NCL], dtype=tl.float32)

    for t in tl.static_range(0, T, 1):
        x_t = tl.load(x_ptr + t * NCL + ncl_idx, mask=mask, other=0.0)
        v = v + x_t + bias                        # ★ 融合 bias add
        spike = (v >= v_threshold).to(tl.float32)
        if SOFT_RESET:
            v = v - spike * v_threshold
        else:
            v = v * (1.0 - spike)
        tl.store(spike_ptr + t * NCL + ncl_idx, spike, mask=mask)


def fused_bias_if(x_seq: torch.Tensor, bias: torch.Tensor,
                  soft_reset: bool = False, v_threshold: float = 1.0) -> torch.Tensor:
    """x_seq: [T, B, C, H, W]，bias: [C]"""
    assert x_seq.is_cuda and x_seq.dtype == torch.float32 and x_seq.is_contiguous()
    assert bias.numel() == x_seq.shape[2]   # C
    T, B, C, H, W = x_seq.shape
    NCL = B * C * H * W
    HW = H * W
    spike_seq = torch.empty_like(x_seq)
    grid = lambda meta: (triton.cdiv(NCL, meta["BLOCK_NCL"]),)
    _fused_bias_if_kernel[grid](
        x_seq, bias, spike_seq,
        T=T, NCL=NCL, HW=HW, BCHW_OVER_C=C,
        v_threshold=v_threshold,
        SOFT_RESET=soft_reset,
    )
    return spike_seq


def _selftest():
    torch.manual_seed(0)
    for shape in [(4, 4, 32, 56, 56), (4, 2, 128, 28, 28)]:
        T, B, C, H, W = shape
        x = torch.randn(*shape, device="cuda").contiguous()
        bias = torch.randn(C, device="cuda") * 0.3
        # 参考：把 bias 加到 x 后用朴素 IF
        x_with_bias = x + bias.view(1, 1, C, 1, 1)
        v = torch.zeros_like(x[0])
        ref = []
        for t in range(T):
            v = v + x_with_bias[t]
            sp = (v >= 1.0).float()
            ref.append(sp)
            v = v * (1.0 - sp)
        ref = torch.stack(ref, dim=0)
        # 测试
        got = fused_bias_if(x, bias, soft_reset=False, v_threshold=1.0)
        diff = (ref != got).sum().item()
        print(f"  shape={shape}  diff={diff}/{ref.numel()}  bit-equal={diff == 0}")


if __name__ == "__main__":
    _selftest()
