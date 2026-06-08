"""SNN Compiler 的 nn.Module 包装层。

提供与 SpikingJelly 类似的 IFNode / LIFNode / CubaLIFNode API，
但 forward 直接调用本框架的 Triton kernel。

每个 module 接收输入 x [T, B, ...] 并返回 spike [T, B, ...]，dtype 与 x 一致。
state（v）在模块内不持久化（仅推理）。如需训练或时序持久化，扩展时把
v 暴露为 module attribute 并允许外部 reset。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels.neurons import if_lif, cuba_lif, eif
from ..kernels.fused import (
    fused_bias_if_lif, conv_neuron, linear_neuron, fold_conv_bn,
    fused_bias_if_lif_rate, fused_bias_if_lif_stateful,
)


# ============================================================
#   纯 neuron 模块
# ============================================================
class IFNode(nn.Module):
    """Integrate-and-Fire 神经元。

    forward(x_seq) -> spike_seq， x_seq.shape = [T, B, ...]
    """
    def __init__(self, *, decay: float = 1.0, soft_reset: bool = False,
                 v_threshold=1.0, v_reset: float = 0.0, layout: str = "NCHW"):
        super().__init__()
        self.decay = decay
        self.soft_reset = soft_reset
        self.v_reset = v_reset
        self.layout = layout
        if isinstance(v_threshold, torch.Tensor):
            self.register_buffer("v_threshold", v_threshold.float().contiguous())
        else:
            self.v_threshold = float(v_threshold)

    def forward(self, x_seq):
        return if_lif(
            x_seq.contiguous(),
            neuron="if", decay=self.decay,
            soft_reset=self.soft_reset, v_threshold=self.v_threshold,
            v_reset=self.v_reset, layout=self.layout,
        )


class LIFNode(nn.Module):
    """Leaky IF 神经元。

    参数语义与 SpikingJelly LIFNode 对齐：
      v_t = decay_factor · v_{t-1} + input_scale · x_t

    其中 decay_factor 默认 = 1 - 1/τ，input_scale 由 decay_input 决定。
    用户也可通过显式 ``decay`` 参数（None=按 τ 推导，非 None=直接覆盖）传入任意
    衰减系数 —— 这允许在不改变 τ 语义的同时用例如 ``decay=0.5`` 做训练实验。
    """
    def __init__(self, *, tau: float = 2.0, decay: float | None = None,
                 decay_input: bool = True,
                 soft_reset: bool = False, v_threshold=1.0, v_reset: float = 0.0,
                 layout: str = "NCHW"):
        super().__init__()
        self.tau = tau
        self.decay = decay
        self.decay_input = decay_input
        self.soft_reset = soft_reset
        self.v_reset = v_reset
        self.layout = layout
        if isinstance(v_threshold, torch.Tensor):
            self.register_buffer("v_threshold", v_threshold.float().contiguous())
        else:
            self.v_threshold = float(v_threshold)

    def forward(self, x_seq):
        return if_lif(
            x_seq.contiguous(),
            neuron="lif", tau=self.tau, decay=self.decay,
            decay_input=self.decay_input,
            soft_reset=self.soft_reset, v_threshold=self.v_threshold,
            v_reset=self.v_reset, layout=self.layout,
        )


class CubaLIFNode(nn.Module):
    """CubaLIF (二阶 LIF): synaptic current i + membrane v 双状态。

    默认 α = exp(-dt/τ_syn), β = exp(-dt/τ_mem)。用户可通过 ``decay_syn`` /
    ``decay_mem`` 直接覆盖 α / β —— 例如 decay_syn=0 退化为单状态 LIF，
    decay_mem=1 关掉膜泄漏。
    """
    def __init__(self, *, tau_syn: float = 2.0, tau_mem: float = 4.0, dt: float = 1.0,
                 decay_syn: float | None = None, decay_mem: float | None = None,
                 input_scale: float = 1.0, soft_reset: bool = False,
                 v_threshold=1.0, v_reset: float = 0.0, layout: str = "NCHW"):
        super().__init__()
        self.tau_syn = tau_syn
        self.tau_mem = tau_mem
        self.dt = dt
        self.decay_syn = decay_syn
        self.decay_mem = decay_mem
        self.input_scale = input_scale
        self.soft_reset = soft_reset
        self.v_reset = v_reset
        self.layout = layout
        if isinstance(v_threshold, torch.Tensor):
            self.register_buffer("v_threshold", v_threshold.float().contiguous())
        else:
            self.v_threshold = float(v_threshold)

    def forward(self, x_seq):
        return cuba_lif(
            x_seq.contiguous(),
            tau_syn=self.tau_syn, tau_mem=self.tau_mem, dt=self.dt,
            decay_syn=self.decay_syn, decay_mem=self.decay_mem,
            input_scale=self.input_scale, soft_reset=self.soft_reset,
            v_threshold=self.v_threshold, v_reset=self.v_reset, layout=self.layout,
        )


class EIFNode(nn.Module):
    """Exponential IF。

    默认 decay = 1 - 1/τ。可通过 ``decay`` 显式覆盖（None=按 τ 推导）。
    """
    def __init__(self, *, tau: float = 2.0, decay: float | None = None,
                 delta_T: float = 1.0, v_rh: float = 0.5,
                 input_scale: float = 1.0, soft_reset: bool = False,
                 v_threshold=1.0, v_reset: float = 0.0, layout: str = "NCHW"):
        super().__init__()
        self.tau = tau
        self.decay = decay
        self.delta_T = delta_T
        self.v_rh = v_rh
        self.input_scale = input_scale
        self.soft_reset = soft_reset
        self.v_reset = v_reset
        self.layout = layout
        if isinstance(v_threshold, torch.Tensor):
            self.register_buffer("v_threshold", v_threshold.float().contiguous())
        else:
            self.v_threshold = float(v_threshold)

    def forward(self, x_seq):
        return eif(
            x_seq.contiguous(),
            tau=self.tau, decay=self.decay,
            delta_T=self.delta_T, v_rh=self.v_rh,
            input_scale=self.input_scale, soft_reset=self.soft_reset,
            v_threshold=self.v_threshold, v_reset=self.v_reset, layout=self.layout,
        )


# ============================================================
#   conv + neuron 融合模块
# ============================================================
class FusedConvNeuron(nn.Module):
    """Conv2d + (optional bias) + IF/LIF 融合（单一 Triton kernel 计算 neuron 部分）。

    输入 x_seq: [T, B, in_C, H, W] (NCHW) 或 [T, B, H, W, in_C] (NHWC)
    输出 spike_seq: 同布局
    """
    def __init__(self, in_ch, out_ch, kernel_size, *, stride=1, padding=0,
                 dilation=1, groups=1, bias: bool = True,
                 neuron: str = "if", tau: float = 2.0, decay: float | None = None,
                 decay_input: bool = True, soft_reset: bool = False,
                 v_threshold=1.0, v_reset: float = 0.0, layout: str = "NCHW"):
        super().__init__()
        conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding,
                          dilation=dilation, groups=groups, bias=bias)
        # 保留原 weight；bias 不传给 F.conv2d，融进 neuron kernel
        self.weight = nn.Parameter(conv.weight.detach().clone())
        self.bias = nn.Parameter(conv.bias.detach().clone()) if bias else None
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        self.neuron = neuron
        self.tau = tau
        self.decay = decay
        self.decay_input = decay_input
        self.soft_reset = soft_reset
        self.v_reset = v_reset
        self.layout = layout
        if isinstance(v_threshold, torch.Tensor):
            self.register_buffer("v_threshold", v_threshold.float().contiguous())
        else:
            self.v_threshold = float(v_threshold)

    def forward(self, x_seq):
        T, B = x_seq.shape[0], x_seq.shape[1]
        x_4d = x_seq.reshape(T * B, *x_seq.shape[2:])
        if self.layout == "NHWC":
            x_4d = x_4d.contiguous(memory_format=torch.channels_last)
            w = self.weight.to(memory_format=torch.channels_last)
        else:
            x_4d = x_4d.contiguous()
            w = self.weight
        y = F.conv2d(x_4d, w, bias=None, stride=self.stride, padding=self.padding,
                      dilation=self.dilation, groups=self.groups)
        if self.layout == "NHWC":
            y = y.contiguous(memory_format=torch.channels_last)
        y_seq = y.view(T, B, *y.shape[1:])

        bias = self.bias
        if bias is not None and bias.dtype != torch.float32:
            bias = bias.float()
        return fused_bias_if_lif(
            y_seq, bias.contiguous() if bias is not None else None,
            neuron=self.neuron, tau=self.tau, decay=self.decay,
            decay_input=self.decay_input, soft_reset=self.soft_reset,
            v_threshold=self.v_threshold, v_reset=self.v_reset, layout=self.layout,
        )


class FusedLinearNeuron(nn.Module):
    """Linear + (optional bias) + IF/LIF 融合。"""
    def __init__(self, in_features, out_features, *, bias: bool = True,
                 neuron: str = "if", tau: float = 2.0, decay: float | None = None,
                 decay_input: bool = True, soft_reset: bool = False,
                 v_threshold=1.0, v_reset: float = 0.0):
        super().__init__()
        lin = nn.Linear(in_features, out_features, bias=bias)
        self.weight = nn.Parameter(lin.weight.detach().clone())
        self.bias = nn.Parameter(lin.bias.detach().clone()) if bias else None
        self.neuron = neuron
        self.tau = tau
        self.decay = decay
        self.decay_input = decay_input
        self.soft_reset = soft_reset
        self.v_reset = v_reset
        if isinstance(v_threshold, torch.Tensor):
            self.register_buffer("v_threshold", v_threshold.float().contiguous())
        else:
            self.v_threshold = float(v_threshold)

    def forward(self, x_seq):
        T, B = x_seq.shape[0], x_seq.shape[1]
        x_2d = x_seq.reshape(T * B, -1)
        y = F.linear(x_2d, self.weight, bias=None)
        out_features = y.shape[-1]
        y_seq = y.view(T, B, out_features)
        bias = self.bias
        if bias is not None and bias.dtype != torch.float32:
            bias = bias.float()
        return fused_bias_if_lif(
            y_seq, bias.contiguous() if bias is not None else None,
            neuron=self.neuron, tau=self.tau, decay=self.decay,
            decay_input=self.decay_input, soft_reset=self.soft_reset,
            v_threshold=self.v_threshold, v_reset=self.v_reset, layout="NCHW",
        )


class FusedConvBNNeuron(nn.Module):
    """Conv2d + BN + IF/LIF 融合。

    构造时接受一个已存在的 nn.Conv2d 与 nn.BatchNorm2d（eval 状态）。

    两种数值模式（``fold_bn``）：

    - ``fold_bn=True``（默认，最快）：__init__ 内一次性把 BN 折叠进 conv
      weight/bias，运行时只剩 conv + neuron 两步。数学等价，但折叠会让卷积
      pre-activation 相对"conv 后再单独 BN"产生 ~1e-3 量级扰动；在脉冲网络的
      硬阈值下这会翻转个别处于阈值边界的脉冲并逐层级联，因此 **fold_bn=True 不
      保证与原网络逐位一致**（对率/精度通常无影响，但逐样本预测可能改变）。
    - ``fold_bn=False``（逐位精确）：conv 与 BN 仍作为两个独立的 eager 算子按
      原顺序计算，只把 neuron 融进 Triton kernel。pre-activation 与原网络逐位
      相同（同 dtype 下），**与原网络逐位一致**。代价是多一次 BN kernel 启动。

    需要"加速但绝不改变推理结果"时用 ``fold_bn=False``；想要最高吞吐且已用
    ``snn_compiler.verify`` 确认精度可接受时用 ``fold_bn=True``。
    """
    def __init__(self, conv: nn.Conv2d, bn: nn.BatchNorm2d, *,
                 fold_bn: bool = True,
                 neuron: str = "if", tau: float = 2.0, decay: float | None = None,
                 decay_input: bool = True, soft_reset: bool = False,
                 v_threshold=1.0, v_reset: float = 0.0, layout: str = "NCHW"):
        super().__init__()
        assert bn.running_mean is not None
        self.fold_bn = fold_bn
        if fold_bn:
            new_w, new_b = fold_conv_bn(
                conv.weight.detach(),
                conv.bias.detach() if conv.bias is not None else None,
                bn.weight.detach(), bn.bias.detach(),
                bn.running_mean.detach(), bn.running_var.detach(),
                bn.eps,
            )
            self.weight = nn.Parameter(new_w)
            self.bias = nn.Parameter(new_b)
        else:
            # bit-exact: 保留原 conv weight/bias，BN 作为独立 eager 算子运行时计算
            self.weight = nn.Parameter(conv.weight.detach().clone())
            self.register_parameter(
                "conv_bias",
                nn.Parameter(conv.bias.detach().clone()) if conv.bias is not None else None,
            )
            self.register_buffer("bn_weight", bn.weight.detach().clone())
            self.register_buffer("bn_bias", bn.bias.detach().clone())
            self.register_buffer("bn_mean", bn.running_mean.detach().clone())
            self.register_buffer("bn_var", bn.running_var.detach().clone())
            self.bn_eps = bn.eps
            self.bias = None                     # neuron 不再单独加 bias（BN 已含 affine）
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.neuron = neuron
        self.tau = tau
        self.decay = decay
        self.decay_input = decay_input
        self.soft_reset = soft_reset
        self.v_reset = v_reset
        self.layout = layout
        if isinstance(v_threshold, torch.Tensor):
            self.register_buffer("v_threshold", v_threshold.float().contiguous())
        else:
            self.v_threshold = float(v_threshold)

    def _conv_bn(self, x_4d):
        """返回 (pre_activation_4d, neuron_bias)。fold_bn 分支在此统一。"""
        if self.layout == "NHWC":
            x_4d = x_4d.contiguous(memory_format=torch.channels_last)
            w = self.weight.to(memory_format=torch.channels_last)
        else:
            x_4d = x_4d.contiguous()
            w = self.weight
        if self.fold_bn:
            y = F.conv2d(x_4d, w, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
            bias = self.bias.float() if self.bias.dtype != torch.float32 else self.bias
            bias = bias.contiguous()
        else:
            y = F.conv2d(x_4d, w, bias=self.conv_bias, stride=self.stride,
                          padding=self.padding, dilation=self.dilation, groups=self.groups)
            y = F.batch_norm(y, self.bn_mean, self.bn_var, self.bn_weight, self.bn_bias,
                              training=False, eps=self.bn_eps)
            bias = None
        if self.layout == "NHWC":
            y = y.contiguous(memory_format=torch.channels_last)
        return y, bias

    def forward(self, x_seq):
        T, B = x_seq.shape[0], x_seq.shape[1]
        x_4d = x_seq.reshape(T * B, *x_seq.shape[2:])
        y, bias = self._conv_bn(x_4d)
        y_seq = y.view(T, B, *y.shape[1:])
        return fused_bias_if_lif(
            y_seq, bias,
            neuron=self.neuron, tau=self.tau, decay=self.decay,
            decay_input=self.decay_input, soft_reset=self.soft_reset,
            v_threshold=self.v_threshold, v_reset=self.v_reset, layout=self.layout,
        )


class FusedConvBNAddNeuron(nn.Module):
    """Conv2d + BN + Add(residual) + IF/LIF 融合（ResNet 第二 conv 后的标准模式）。

    forward(x_seq, residual_seq) -> spike_seq
       y = conv-bn-fold(x);    spike = neuron(y + bias + residual)

    与 FusedConvBNNeuron 的差别在 forward 多接一个 residual_seq 输入，由
    Triton kernel 在 t 循环里一并消耗。
    """
    def __init__(self, conv: nn.Conv2d, bn: nn.BatchNorm2d, *,
                 fold_bn: bool = True,
                 neuron: str = "if", tau: float = 2.0, decay: float | None = None,
                 decay_input: bool = True, soft_reset: bool = False,
                 v_threshold=1.0, v_reset: float = 0.0, layout: str = "NCHW"):
        super().__init__()
        assert bn.running_mean is not None
        self.fold_bn = fold_bn
        if fold_bn:
            new_w, new_b = fold_conv_bn(
                conv.weight.detach(),
                conv.bias.detach() if conv.bias is not None else None,
                bn.weight.detach(), bn.bias.detach(),
                bn.running_mean.detach(), bn.running_var.detach(),
                bn.eps,
            )
            self.weight = nn.Parameter(new_w)
            self.bias = nn.Parameter(new_b)
        else:
            self.weight = nn.Parameter(conv.weight.detach().clone())
            self.register_parameter(
                "conv_bias",
                nn.Parameter(conv.bias.detach().clone()) if conv.bias is not None else None,
            )
            self.register_buffer("bn_weight", bn.weight.detach().clone())
            self.register_buffer("bn_bias", bn.bias.detach().clone())
            self.register_buffer("bn_mean", bn.running_mean.detach().clone())
            self.register_buffer("bn_var", bn.running_var.detach().clone())
            self.bn_eps = bn.eps
            self.bias = None
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.neuron = neuron
        self.tau = tau
        self.decay = decay
        self.decay_input = decay_input
        self.soft_reset = soft_reset
        self.v_reset = v_reset
        self.layout = layout
        if isinstance(v_threshold, torch.Tensor):
            self.register_buffer("v_threshold", v_threshold.float().contiguous())
        else:
            self.v_threshold = float(v_threshold)

    def forward(self, x_seq, residual_seq):
        T, B = x_seq.shape[0], x_seq.shape[1]
        x_4d = x_seq.reshape(T * B, *x_seq.shape[2:])
        if self.layout == "NHWC":
            x_4d = x_4d.contiguous(memory_format=torch.channels_last)
            w = self.weight.to(memory_format=torch.channels_last)
        else:
            x_4d = x_4d.contiguous()
            w = self.weight
        if self.fold_bn:
            y = F.conv2d(x_4d, w, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
            bias = self.bias.float() if self.bias.dtype != torch.float32 else self.bias
            bias = bias.contiguous()
        else:
            y = F.conv2d(x_4d, w, bias=self.conv_bias, stride=self.stride,
                          padding=self.padding, dilation=self.dilation, groups=self.groups)
            y = F.batch_norm(y, self.bn_mean, self.bn_var, self.bn_weight, self.bn_bias,
                              training=False, eps=self.bn_eps)
            bias = None
        if self.layout == "NHWC":
            y = y.contiguous(memory_format=torch.channels_last)
        y_seq = y.view(T, B, *y.shape[1:])
        return fused_bias_if_lif(
            y_seq, bias, residual=residual_seq,
            neuron=self.neuron, tau=self.tau, decay=self.decay,
            decay_input=self.decay_input, soft_reset=self.soft_reset,
            v_threshold=self.v_threshold, v_reset=self.v_reset, layout=self.layout,
        )


class FusedAddNeuron(nn.Module):
    """两条 spike/feature 路径相加，然后过 IF/LIF。

    forward(a_seq, b_seq) -> spike_seq
       v_t = decay * v + scale * (a_t + b_t)
       spike, reset

    用途：纯残差合流（无 conv 紧邻 neuron 时）；多分支 SNN；门控 SNN 求和。
    """
    def __init__(self, *, neuron: str = "if", tau: float = 2.0,
                 decay: float | None = None, decay_input: bool = True,
                 soft_reset: bool = False, v_threshold=1.0, v_reset: float = 0.0,
                 layout: str = "NCHW"):
        super().__init__()
        self.neuron = neuron
        self.tau = tau
        self.decay = decay
        self.decay_input = decay_input
        self.soft_reset = soft_reset
        self.v_reset = v_reset
        self.layout = layout
        if isinstance(v_threshold, torch.Tensor):
            self.register_buffer("v_threshold", v_threshold.float().contiguous())
        else:
            self.v_threshold = float(v_threshold)

    def forward(self, a_seq, b_seq):
        # 借 fused_bias_if_lif：把 b 当 residual，bias=None
        return fused_bias_if_lif(
            a_seq.contiguous() if a_seq.is_contiguous() or self.layout != "NHWC"
            else a_seq,
            None, residual=b_seq,
            neuron=self.neuron, tau=self.tau, decay=self.decay,
            decay_input=self.decay_input, soft_reset=self.soft_reset,
            v_threshold=self.v_threshold, v_reset=self.v_reset, layout=self.layout,
        )


# ============================================================
#   Rate-coded 神经元（spike-count 输出，T-axis collapsed）
# ============================================================
class RateCodedLIFNode(nn.Module):
    """LIF 但只输出 spike-count 总和（输出比 LIFNode 少一个 T 维）。

    forward(x_seq) -> spike_count
       x_seq:       [T, B, ...]
       spike_count: [B, ...]   fp32

    用途：网络最后一层 LIF（紧邻分类器或本身就是分类层）。下游不再需要
    per-t spike train，直接消费 spike-count 投票即可。
    带宽收益：spike 写出 NCL*4 vs T*NCL*2（bf16）≈ T/2 倍写入节省，T=128
    时 64× 写入节省，整体 ~2.2× 加速（B-2 phase 已验证 bit-equal）。
    """
    def __init__(self, *, tau: float = 2.0, decay: float | None = None,
                 decay_input: bool = True, soft_reset: bool = False,
                 v_threshold=1.0, v_reset: float = 0.0, layout: str = "NCHW"):
        super().__init__()
        self.tau = tau
        self.decay = decay
        self.decay_input = decay_input
        self.soft_reset = soft_reset
        self.v_reset = v_reset
        self.layout = layout
        if isinstance(v_threshold, torch.Tensor):
            self.register_buffer("v_threshold", v_threshold.float().contiguous())
        else:
            self.v_threshold = float(v_threshold)

    def forward(self, x_seq):
        return fused_bias_if_lif_rate(
            x_seq.contiguous() if (x_seq.is_contiguous() or self.layout != "NHWC")
            else x_seq,
            None,
            neuron="lif", tau=self.tau, decay=self.decay,
            decay_input=self.decay_input, soft_reset=self.soft_reset,
            v_threshold=self.v_threshold, v_reset=self.v_reset, layout=self.layout,
        )


class RateCodedIFNode(nn.Module):
    """IF 版的 rate-coded（spike-count 输出）。同 RateCodedLIFNode 但 decay=1.0、scale=1.0。"""
    def __init__(self, *, decay: float = 1.0, soft_reset: bool = False,
                 v_threshold=1.0, v_reset: float = 0.0, layout: str = "NCHW"):
        super().__init__()
        self.decay = decay
        self.soft_reset = soft_reset
        self.v_reset = v_reset
        self.layout = layout
        if isinstance(v_threshold, torch.Tensor):
            self.register_buffer("v_threshold", v_threshold.float().contiguous())
        else:
            self.v_threshold = float(v_threshold)

    def forward(self, x_seq):
        return fused_bias_if_lif_rate(
            x_seq.contiguous() if (x_seq.is_contiguous() or self.layout != "NHWC")
            else x_seq,
            None,
            neuron="if", decay=self.decay,
            soft_reset=self.soft_reset, v_threshold=self.v_threshold,
            v_reset=self.v_reset, layout=self.layout,
        )


# ============================================================
#   Stateful LIF（可以串接 v 状态：T-chunked execution 用）
# ============================================================
class StatefulLIFNode(nn.Module):
    """LIF 但 forward 接受 ``v_init`` 与可选返回 ``v_final``，用于 T-chunked execution。

    forward(x_seq, v_init=None, return_v=False)
      - x_seq: [T_chunk, B, ...]
      - v_init: 上一个 chunk 末态 [B, ...] fp32，或 None
      - return_v=True 时返回 (spike_seq, v_final)；否则返回 spike_seq

    用法：T-chunked driver 把整个 [T, B, ...] 按 chunk_t 切片，逐 chunk 前向；
    每层 StatefulLIFNode 保留自己的 v 状态。如此可让 T=128 的 VGG-16 适配
    16GiB 显卡（峰值 ≈ T/chunk_t 倍降）。
    """
    def __init__(self, *, tau: float = 2.0, decay: float | None = None,
                 decay_input: bool = True, soft_reset: bool = False,
                 v_threshold: float = 1.0, v_reset: float = 0.0, layout: str = "NCHW"):
        super().__init__()
        self.tau = tau
        self.decay = decay
        self.decay_input = decay_input
        self.soft_reset = soft_reset
        self.v_threshold = float(v_threshold)
        self.v_reset = v_reset
        self.layout = layout

    def forward(self, x_seq, v_init=None, return_v=False):
        return fused_bias_if_lif_stateful(
            x_seq.contiguous() if (x_seq.is_contiguous() or self.layout != "NHWC")
            else x_seq,
            None,
            v_init=v_init, return_v=return_v,
            neuron="lif", tau=self.tau, decay=self.decay,
            decay_input=self.decay_input, soft_reset=self.soft_reset,
            v_threshold=self.v_threshold, v_reset=self.v_reset, layout=self.layout,
        )
