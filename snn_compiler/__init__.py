"""SNN Compiler — Triton 后端的通用 IF/LIF/CubaLIF/EIF SNN 推理优化框架。

模块组织
========
- snn_compiler.kernels.neurons : 纯 neuron Triton kernel + Python entrypoints
- snn_compiler.kernels.fused   : Conv-bias-neuron / Conv-BN-neuron 融合 kernel
- snn_compiler.nn.modules      : nn.Module 包装层（IFNode, LIFNode, FusedConvNeuron, ...）
- snn_compiler.passes.fuse     : torch.fx 图重写 pass（自动识别 Conv→BN→Neuron 并融合）

支持矩阵
========
| 神经元模型 | decay         | reset       | threshold              | 备注           |
| ---------- | ------------- | ----------- | ---------------------- | -------------- |
| IF         | 0 / 任意      | soft / hard | scalar / per-C / per-N | 含 leaky IF    |
| LIF        | (1-1/τ)       | soft / hard | scalar / per-C / per-N | decay_input ±  |
| CubaLIF    | 双 τ          | soft / hard | scalar / per-C / per-N | i_syn + v 双态 |
| EIF        | (1-1/τ) + exp | soft / hard | scalar / per-C / per-N | 非线性指数项   |

任意 v_reset 常数都支持（hard reset 时把 v 置为 v_reset_val）。
"""
from .kernels.neurons import if_lif, cuba_lif, eif, naive_if_lif, naive_cuba_lif, naive_eif
from .kernels.fused import (
    fused_bias_if_lif, conv_neuron, linear_neuron, conv_bn_neuron, fold_conv_bn,
)
from .nn.modules import (
    IFNode, LIFNode, CubaLIFNode, EIFNode,
    FusedConvNeuron, FusedLinearNeuron, FusedConvBNNeuron,
)

__all__ = [
    "if_lif", "cuba_lif", "eif",
    "naive_if_lif", "naive_cuba_lif", "naive_eif",
    "fused_bias_if_lif", "conv_neuron", "linear_neuron", "conv_bn_neuron",
    "fold_conv_bn",
    "IFNode", "LIFNode", "CubaLIFNode", "EIFNode",
    "FusedConvNeuron", "FusedLinearNeuron", "FusedConvBNNeuron",
]
