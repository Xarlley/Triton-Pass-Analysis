"""Conv/Linear + bias + neuron 融合 kernel 库。

设计动机
========
SNN 推理时，conv/linear 的 bias 累加和 neuron 的膜电位更新在 elementwise 上是
相邻的两次访问 conv 输出 tensor 的操作。把 bias 加进 neuron kernel：
  - 省一次 launch（13 个 conv 层 → 省 13 次 elementwise bias-add）
  - 省一次 GMEM 读写（bias 通过 bias[c_idx] 直接进寄存器）

布局兼容
========
- NCHW: bias index = (ncl_idx // HW) % C
- NHWC: bias index = ncl_idx % C

支持的 neuron 类型
==================
- IF（含 leaky IF 通用 decay）
- LIF（含 decay_input True/False）
后续按需扩展 CubaLIF。

reset / threshold
=================
完全沿用 neurons.py 的语义。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .neurons import (
    RESET_SOFT, RESET_HARD,
    THR_SCALAR, THR_PER_CHANNEL, THR_PER_NEURON,
)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": 128}, num_warps=4),
        triton.Config({"BLOCK_NCL": 256}, num_warps=4),
        triton.Config({"BLOCK_NCL": 256}, num_warps=8),
        triton.Config({"BLOCK_NCL": 512}, num_warps=8),
        triton.Config({"BLOCK_NCL": 1024}, num_warps=8),
    ],
    key=["T", "NCL", "C", "THR_MODE", "RESET_MODE", "CHANNEL_LAST",
         "HAS_BIAS", "HAS_RESIDUAL"],
)
@triton.jit
def _bias_if_lif_kernel(
    y_ptr,                              # [T, NCL]，conv 输出（无 bias）
    bias_ptr,                           # [C]
    residual_ptr,                       # [T, NCL]，HAS_RESIDUAL 时有效
    spike_ptr,                          # 同 y 形状
    v_th_ptr,                           # [] / [C] / [NCL]
    T: tl.constexpr, NCL: tl.constexpr, C: tl.constexpr, HW: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    decay_factor: tl.constexpr,
    input_scale: tl.constexpr,
    v_threshold_const: tl.constexpr,
    v_reset_val: tl.constexpr,
    RESET_MODE: tl.constexpr,
    THR_MODE: tl.constexpr,
    CHANNEL_LAST: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    """conv-out + bias + (optional residual) + IF/LIF 融合：

       y 是 conv/conv-bn 输出（无 bias），形状 reshape 成 [T, NCL]。
       residual 是另一路同形 tensor（ResNet 的 identity / skip）。

       v_t = decay*v + scale*(y_t + bias + (residual_t if HAS_RESIDUAL else 0))
    """
    pid = tl.program_id(0)
    # 用 i64 算偏移：大 T × 大 NCL 时 (T-1) * NCL * sizeof(elem) 可能 > 2^31
    # 否则 Triton 默认 i32 字节偏移会 wrap-around → cudaErrorIllegalAddress。
    ncl_idx = pid.to(tl.int64) * BLOCK_NCL + tl.arange(0, BLOCK_NCL).to(tl.int64)
    mask = ncl_idx < NCL
    NCL_i64 = tl.full([], NCL, dtype=tl.int64)

    if CHANNEL_LAST:
        c_idx = ncl_idx % C
    else:
        c_idx = (ncl_idx // HW) % C

    if HAS_BIAS:
        bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    else:
        bias = tl.zeros([BLOCK_NCL], dtype=tl.float32)

    if THR_MODE == 0:
        v_th = v_threshold_const
    elif THR_MODE == 1:
        v_th = tl.load(v_th_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    else:
        v_th = tl.load(v_th_ptr + ncl_idx, mask=mask, other=0.0).to(tl.float32)

    v = tl.zeros([BLOCK_NCL], dtype=tl.float32)

    for t in tl.static_range(0, T, 1):
        t_off = tl.full([], t, dtype=tl.int64) * NCL_i64
        y_t = tl.load(y_ptr + t_off + ncl_idx, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            r_t = tl.load(residual_ptr + t_off + ncl_idx, mask=mask, other=0.0).to(tl.float32)
            v = decay_factor * v + input_scale * (y_t + bias + r_t)
        else:
            v = decay_factor * v + input_scale * (y_t + bias)
        spike = (v >= v_th).to(tl.float32)
        if RESET_MODE == 0:
            v = v - spike * v_th
        else:
            v = v * (1.0 - spike) + spike * v_reset_val
        tl.store(spike_ptr + t_off + ncl_idx, spike, mask=mask)


def fused_bias_if_lif(
    y_seq: torch.Tensor,            # [T, B, C, H, W]
    bias: torch.Tensor | None,      # [C] 或 None
    *,
    residual: torch.Tensor | None = None,    # 同形 spike/feature，None 时无残差
    neuron: str = "if",
    tau: float = 2.0,
    decay: float | None = None,
    decay_input: bool = True,
    soft_reset: bool = False,
    v_threshold=1.0,
    v_reset: float = 0.0,
    layout: str = "NCHW",
) -> torch.Tensor:
    """conv 输出 + bias + (optional residual) + IF/LIF 融合调用。

    - y_seq: F.conv2d 输出按 [T, B, ...] reshape；conv 不能自加 bias。
    - bias:  conv bias / BN-folded bias；None 表示无 bias。
    - residual: 残差路径输出（同形）；用于 ResNet 的 `conv(x) + identity → neuron`
      一步融合。None 时退化为普通 ConvBN-Neuron。
    """
    # 允许标准 contiguous 或 NHWC（channels_last）存储 — 二者在 [T, NCL] flatten 后
    # 沿 linear memory 的访存都是连续的；kernel 仅按 t*NCL + idx 索引。
    assert y_seq.is_cuda
    if not y_seq.is_contiguous():
        assert layout == "NHWC" and y_seq.ndim >= 4, \
            f"non-contiguous y_seq requires NHWC layout, got layout={layout} shape={tuple(y_seq.shape)}"
    T = y_seq.shape[0]
    NCL = y_seq[0].numel()

    # 注意：PyTorch 习惯总是 [T, B, C, H, W] 标注，channels_last 只是内存格式。
    # 所以 NCHW / NHWC 在 shape index 上一致，只是 kernel 内访存模式不同。
    if y_seq.ndim == 5:
        C = y_seq.shape[2]
        HW = y_seq.shape[3] * y_seq.shape[4]
    elif y_seq.ndim == 4:
        C = y_seq.shape[2]
        HW = y_seq.shape[3]
    elif y_seq.ndim == 3:
        C = y_seq.shape[-1]
        HW = 1
    else:
        C = y_seq[0].numel()
        HW = 1

    if neuron == "if":
        decay_factor = 1.0 if decay is None else float(decay)
        input_scale = 1.0
    elif neuron == "lif":
        decay_factor = (1.0 - 1.0 / tau) if decay is None else float(decay)
        input_scale = (1.0 / tau) if decay_input else 1.0
    else:
        raise ValueError(f"unknown neuron: {neuron!r}")

    # threshold
    if isinstance(v_threshold, torch.Tensor):
        if v_threshold.numel() == C:
            THR_MODE = THR_PER_CHANNEL
            thr_ptr = v_threshold
            thr_const = 0.0
        elif v_threshold.numel() == NCL:
            THR_MODE = THR_PER_NEURON
            thr_ptr = v_threshold
            thr_const = 0.0
        else:
            raise ValueError(f"v_threshold shape {tuple(v_threshold.shape)} unsupported")
    else:
        THR_MODE = THR_SCALAR
        thr_ptr = y_seq  # 占位
        thr_const = float(v_threshold)

    HAS_BIAS = bias is not None
    if not HAS_BIAS:
        bias_arg = y_seq  # 占位
    else:
        assert bias.is_cuda and bias.is_contiguous() and bias.dtype == torch.float32 \
               and bias.numel() == C, f"bias shape mismatch: {tuple(bias.shape)} vs C={C}"
        bias_arg = bias

    HAS_RESIDUAL = residual is not None
    if not HAS_RESIDUAL:
        residual_arg = y_seq  # 占位
    else:
        assert residual.is_cuda and residual.dtype == y_seq.dtype \
               and residual.shape == y_seq.shape, \
               f"residual shape {tuple(residual.shape)} must match y_seq {tuple(y_seq.shape)}"
        if not residual.is_contiguous():
            assert layout == "NHWC", \
                f"non-contiguous residual requires NHWC layout"
        residual_arg = residual

    RESET_MODE = RESET_SOFT if soft_reset else RESET_HARD
    channel_last = (layout == "NHWC")

    spike_seq = torch.empty_like(y_seq)
    grid = lambda meta: (triton.cdiv(NCL, meta["BLOCK_NCL"]),)
    _bias_if_lif_kernel[grid](
        y_seq, bias_arg, residual_arg, spike_seq, thr_ptr,
        T=T, NCL=NCL, C=C, HW=HW,
        decay_factor=decay_factor,
        input_scale=input_scale,
        v_threshold_const=thr_const,
        v_reset_val=v_reset,
        RESET_MODE=RESET_MODE,
        THR_MODE=THR_MODE,
        CHANNEL_LAST=channel_last,
        HAS_BIAS=HAS_BIAS,
        HAS_RESIDUAL=HAS_RESIDUAL,
    )
    return spike_seq


# ============================================================
#   Rate-coded output kernel：只写 sum-over-T spike-count
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": 256}, num_warps=4),
        triton.Config({"BLOCK_NCL": 256}, num_warps=8),
        triton.Config({"BLOCK_NCL": 512}, num_warps=8),
        triton.Config({"BLOCK_NCL": 1024}, num_warps=8),
    ],
    key=["T", "NCL", "C", "THR_MODE", "RESET_MODE", "CHANNEL_LAST", "HAS_BIAS"],
)
@triton.jit
def _bias_if_lif_rate_kernel(
    y_ptr, bias_ptr, count_ptr, v_th_ptr,
    T: tl.constexpr, NCL: tl.constexpr, C: tl.constexpr, HW: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    decay_factor: tl.constexpr,
    input_scale: tl.constexpr,
    v_threshold_const: tl.constexpr,
    v_reset_val: tl.constexpr,
    RESET_MODE: tl.constexpr,
    THR_MODE: tl.constexpr,
    CHANNEL_LAST: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Rate-coded：T 步内累加 spike-count，仅写一次 [NCL] 输出。

    带宽节省：
       原版 spike 写出 = T × NCL × sizeof(spike_dtype) (bf16=2B)
       rate 版         = NCL × 4B
       ratio           = T × 2 / 4 = T / 2  →  T=128 时 64×；T=4 时 2×

    用途：网络最后一层 LIF/IF（全连接分类器前/后），需要 rate code 投票。
          中间 LIF 不可替换（下游 conv 要 per-t spike）。
    """
    pid = tl.program_id(0)
    ncl_idx = pid.to(tl.int64) * BLOCK_NCL + tl.arange(0, BLOCK_NCL).to(tl.int64)
    mask = ncl_idx < NCL
    NCL_i64 = tl.full([], NCL, dtype=tl.int64)

    if CHANNEL_LAST:
        c_idx = ncl_idx % C
    else:
        c_idx = (ncl_idx // HW) % C

    if HAS_BIAS:
        bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    else:
        bias = tl.zeros([BLOCK_NCL], dtype=tl.float32)

    if THR_MODE == 0:
        v_th = v_threshold_const
    elif THR_MODE == 1:
        v_th = tl.load(v_th_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    else:
        v_th = tl.load(v_th_ptr + ncl_idx, mask=mask, other=0.0).to(tl.float32)

    v = tl.zeros([BLOCK_NCL], dtype=tl.float32)
    count = tl.zeros([BLOCK_NCL], dtype=tl.float32)

    for t in tl.static_range(0, T, 1):
        t_off = tl.full([], t, dtype=tl.int64) * NCL_i64
        y_t = tl.load(y_ptr + t_off + ncl_idx, mask=mask, other=0.0).to(tl.float32)
        v = decay_factor * v + input_scale * (y_t + bias)
        spike = (v >= v_th).to(tl.float32)
        if RESET_MODE == 0:
            v = v - spike * v_th
        else:
            v = v * (1.0 - spike) + spike * v_reset_val
        count = count + spike
    tl.store(count_ptr + ncl_idx, count, mask=mask)


def fused_bias_if_lif_rate(
    y_seq: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    neuron: str = "if",
    tau: float = 2.0,
    decay: float | None = None,
    decay_input: bool = True,
    soft_reset: bool = False,
    v_threshold=1.0,
    v_reset: float = 0.0,
    layout: str = "NCHW",
) -> torch.Tensor:
    """Rate-coded 入口：返回 ``[B, C, H, W]`` 的 fp32 spike-count（求和 over T）。

    输入参数与 :func:`fused_bias_if_lif` 完全一致；输出维度少一个 T 轴。
    用法：最后一个 LIF/IF（分类层之前/之后）的高效替换。
    """
    assert y_seq.is_cuda
    if not y_seq.is_contiguous():
        assert layout == "NHWC" and y_seq.ndim >= 4
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

    if neuron == "if":
        decay_factor = 1.0 if decay is None else float(decay)
        input_scale = 1.0
    elif neuron == "lif":
        decay_factor = (1.0 - 1.0 / tau) if decay is None else float(decay)
        input_scale = (1.0 / tau) if decay_input else 1.0
    else:
        raise ValueError(f"unknown neuron: {neuron!r}")

    if isinstance(v_threshold, torch.Tensor):
        if v_threshold.numel() == C:
            THR_MODE = THR_PER_CHANNEL
            thr_ptr = v_threshold
            thr_const = 0.0
        elif v_threshold.numel() == NCL:
            THR_MODE = THR_PER_NEURON
            thr_ptr = v_threshold
            thr_const = 0.0
        else:
            raise ValueError(f"v_threshold shape {tuple(v_threshold.shape)} unsupported")
    else:
        THR_MODE = THR_SCALAR
        thr_ptr = y_seq
        thr_const = float(v_threshold)

    HAS_BIAS = bias is not None
    bias_arg = bias if HAS_BIAS else y_seq
    if HAS_BIAS:
        assert bias.is_cuda and bias.dtype == torch.float32 and bias.numel() == C

    RESET_MODE = RESET_SOFT if soft_reset else RESET_HARD
    channel_last = (layout == "NHWC")

    count = torch.empty(y_seq.shape[1:], device=y_seq.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(NCL, meta["BLOCK_NCL"]),)
    _bias_if_lif_rate_kernel[grid](
        y_seq, bias_arg, count, thr_ptr,
        T=T, NCL=NCL, C=C, HW=HW,
        decay_factor=decay_factor,
        input_scale=input_scale,
        v_threshold_const=thr_const,
        v_reset_val=v_reset,
        RESET_MODE=RESET_MODE,
        THR_MODE=THR_MODE,
        CHANNEL_LAST=channel_last,
        HAS_BIAS=HAS_BIAS,
    )
    return count


# ============================================================
#   Stateful LIF (v_init / v_final I/O)：T-chunked execution 用
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": 256}, num_warps=4),
        triton.Config({"BLOCK_NCL": 512}, num_warps=8),
        triton.Config({"BLOCK_NCL": 1024}, num_warps=8),
    ],
    key=["T", "NCL", "C", "RESET_MODE", "CHANNEL_LAST",
          "HAS_BIAS", "HAS_VINIT", "HAS_VOUT"],
)
@triton.jit
def _bias_if_lif_stateful_kernel(
    y_ptr, bias_ptr, spike_ptr, v_init_ptr, v_out_ptr,
    T: tl.constexpr, NCL: tl.constexpr, C: tl.constexpr, HW: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    decay_factor: tl.constexpr, input_scale: tl.constexpr,
    v_threshold: tl.constexpr, v_reset_val: tl.constexpr,
    RESET_MODE: tl.constexpr,
    CHANNEL_LAST: tl.constexpr,
    HAS_BIAS: tl.constexpr, HAS_VINIT: tl.constexpr, HAS_VOUT: tl.constexpr,
):
    """带 v 初末值 I/O 的 LIF kernel；用于 chunked driver 串接 v 状态。"""
    pid = tl.program_id(0)
    ncl_idx = pid.to(tl.int64) * BLOCK_NCL + tl.arange(0, BLOCK_NCL).to(tl.int64)
    mask = ncl_idx < NCL
    NCL_i64 = tl.full([], NCL, dtype=tl.int64)

    if CHANNEL_LAST:
        c_idx = ncl_idx % C
    else:
        c_idx = (ncl_idx // HW) % C

    if HAS_BIAS:
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
        if RESET_MODE == 0:
            v = v - spike * v_threshold
        else:
            v = v * (1.0 - spike) + spike * v_reset_val
        tl.store(spike_ptr + t_off + ncl_idx, spike, mask=mask)

    if HAS_VOUT:
        tl.store(v_out_ptr + ncl_idx, v, mask=mask)


def fused_bias_if_lif_stateful(
    y_seq: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    v_init: torch.Tensor | None = None,
    return_v: bool = False,
    neuron: str = "if",
    tau: float = 2.0,
    decay: float | None = None,
    decay_input: bool = True,
    soft_reset: bool = False,
    v_threshold: float = 1.0,
    v_reset: float = 0.0,
    layout: str = "NCHW",
):
    """T-chunked driver 专用：可选 v_init / return_v。

    仅支持 scalar threshold（chunked 场景下 per-channel/per-neuron 阈值少见，简化签名）。
    """
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

    if neuron == "if":
        decay_factor = 1.0 if decay is None else float(decay)
        input_scale = 1.0
    elif neuron == "lif":
        decay_factor = (1.0 - 1.0 / tau) if decay is None else float(decay)
        input_scale = (1.0 / tau) if decay_input else 1.0
    else:
        raise ValueError(f"unknown neuron: {neuron!r}")

    HAS_BIAS = bias is not None
    bias_arg = bias if HAS_BIAS else y_seq
    HAS_VINIT = v_init is not None
    v_init_arg = v_init.reshape(-1) if HAS_VINIT else y_seq
    if HAS_VINIT:
        assert v_init.numel() == NCL and v_init.dtype == torch.float32
    HAS_VOUT = return_v
    v_out = torch.empty(y_seq.shape[1:], device=y_seq.device, dtype=torch.float32) \
            if HAS_VOUT else y_seq

    spike = torch.empty_like(y_seq)
    grid = lambda meta: (triton.cdiv(NCL, meta["BLOCK_NCL"]),)
    _bias_if_lif_stateful_kernel[grid](
        y_seq, bias_arg, spike, v_init_arg, v_out,
        T=T, NCL=NCL, C=C, HW=HW,
        decay_factor=decay_factor, input_scale=input_scale,
        v_threshold=float(v_threshold), v_reset_val=v_reset,
        RESET_MODE=(0 if soft_reset else 1),
        CHANNEL_LAST=(layout == "NHWC"),
        HAS_BIAS=HAS_BIAS, HAS_VINIT=HAS_VINIT, HAS_VOUT=HAS_VOUT,
    )
    if HAS_VOUT:
        return spike, v_out
    return spike


# ============================================================
#   高层封装：conv2d + bias + neuron 端到端
# ============================================================
def conv_neuron(
    x_seq: torch.Tensor,            # [T, B, in_C, H, W] (NCHW) or [T, B, H, W, in_C] (NHWC)
    weight: torch.Tensor,           # conv2d weight
    bias: torch.Tensor | None,      # [out_C] or None
    *,
    stride=1, padding=0, dilation=1, groups=1,
    neuron: str = "if",
    tau: float = 2.0,
    decay: float | None = None,
    decay_input: bool = True,
    soft_reset: bool = False,
    v_threshold=1.0,
    v_reset: float = 0.0,
    layout: str = "NCHW",
) -> torch.Tensor:
    """conv + bias + neuron 端到端入口（bias 不传给 conv，融进 neuron kernel）。"""
    T = x_seq.shape[0]
    if layout == "NCHW":
        B = x_seq.shape[1]
        x_4d = x_seq.reshape(T * B, *x_seq.shape[2:])
    else:
        B = x_seq.shape[1]
        x_4d = x_seq.reshape(T * B, *x_seq.shape[2:])
        x_4d = x_4d.to(memory_format=torch.channels_last)

    y = F.conv2d(x_4d, weight, bias=None,
                 stride=stride, padding=padding, dilation=dilation, groups=groups)
    # y: [T*B, out_C, H', W'] (NCHW) 或 [T*B, H', W', out_C] (NHWC if stored cl)
    # reshape 回 [T, B, ...]
    if layout == "NCHW":
        y_seq = y.view(T, B, *y.shape[1:])
    else:
        y_seq = y.contiguous(memory_format=torch.channels_last).view(T, B, *y.shape[1:])
    return fused_bias_if_lif(
        y_seq, bias,
        neuron=neuron, tau=tau, decay=decay, decay_input=decay_input,
        soft_reset=soft_reset, v_threshold=v_threshold, v_reset=v_reset,
        layout=layout,
    )


def linear_neuron(
    x_seq: torch.Tensor,            # [T, B, in_features]
    weight: torch.Tensor,           # [out_features, in_features]
    bias: torch.Tensor | None,      # [out_features] or None
    *,
    neuron: str = "if",
    tau: float = 2.0,
    decay: float | None = None,
    decay_input: bool = True,
    soft_reset: bool = False,
    v_threshold=1.0,
    v_reset: float = 0.0,
) -> torch.Tensor:
    """linear + bias + neuron 端到端。对 1D 张量场景做类比融合。"""
    T = x_seq.shape[0]
    B = x_seq.shape[1]
    x_2d = x_seq.reshape(T * B, -1)
    y = F.linear(x_2d, weight, bias=None)
    out_features = y.shape[-1]
    y_seq = y.view(T, B, out_features)
    return fused_bias_if_lif(
        y_seq, bias,
        neuron=neuron, tau=tau, decay=decay, decay_input=decay_input,
        soft_reset=soft_reset, v_threshold=v_threshold, v_reset=v_reset,
        layout="NCHW",   # 1D 不区分 layout，C 维就是最后一维；走 NCHW 分支：C=out_features, HW=1
    )


# ============================================================
#   conv + BN + neuron：BN folding 在 Python 侧完成
# ============================================================
def fold_conv_bn(conv_w, conv_b, bn_w, bn_b, bn_mean, bn_var, bn_eps):
    """把 BN 折叠进 conv weight/bias（数学等价）。

       y_bn = γ * (conv(x, W) + b - μ) / sqrt(σ² + ε) + β
            = γ/sqrt(σ²+ε) * conv(x, W) + (γ * (b - μ) / sqrt(σ²+ε) + β)

       折叠后 W' = γ/sqrt(σ²+ε) · W (沿 out_C scale)
              b' = γ/sqrt(σ²+ε) · (b - μ) + β
    """
    inv_std = bn_w / torch.sqrt(bn_var + bn_eps)        # [C]
    new_w = conv_w * inv_std.view(-1, 1, 1, 1).to(conv_w.dtype)
    if conv_b is None:
        conv_b = torch.zeros_like(bn_w)
    new_b = (conv_b - bn_mean) * inv_std + bn_b
    return new_w, new_b


def conv_bn_neuron(x_seq, conv_w, conv_b, bn_w, bn_b, bn_mean, bn_var, bn_eps,
                    *, stride=1, padding=0, dilation=1, groups=1, **neuron_kwargs):
    """conv + BN + neuron 端到端：先用 Python 折叠 BN，再走 conv_neuron。

       折叠后的 W'/b' 通常在模型加载时一次完成；这里仅给出推理调用接口。
    """
    new_w, new_b = fold_conv_bn(conv_w, conv_b, bn_w, bn_b, bn_mean, bn_var, bn_eps)
    return conv_neuron(x_seq, new_w, new_b,
                        stride=stride, padding=padding, dilation=dilation, groups=groups,
                        **neuron_kwargs)
