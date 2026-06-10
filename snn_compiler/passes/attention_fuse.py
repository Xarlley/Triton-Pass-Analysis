"""探测并融合脉冲注意力块 —— 卷积探测/融合之外，snn_compiler 对注意力的对应能力。

与 ``fuse_snn_model`` 同思路：遍历 named_modules，duck-type 识别脉冲注意力块（无 softmax 的
脉冲 Q/K/V + 线性序矩阵乘 + LIF），就地替换为 ``FusedSpikeAttention``。
"""
from __future__ import annotations

import torch.nn as nn

from ..nn.attention import FusedSpikeAttention, is_spiking_attention
from .fuse import _set_submodule


def fuse_spiking_attention(model: nn.Module, *, fold_bn: bool = False,
                           ktv_mode: str = "bmm") -> int:
    """就地把模型里所有脉冲注意力块替换为 FusedSpikeAttention。返回替换计数。

    Args:
        model: eval() 状态模型。
        fold_bn: False（默认）逐位一致；True 更快但 BN 折叠会翻转个别脉冲。
        ktv_mode: 'bmm'（默认）/ 'popcount'（bit-pack+popcount KᵀV，逐位一致且更快，见 README §6.5）。

    建议：替换后用 ``snn_compiler.verify.assert_equivalent`` 守门。
    """
    targets = [(n, m) for n, m in model.named_modules()
               if is_spiking_attention(m)]
    for name, m in targets:
        _set_submodule(model, name,
                       FusedSpikeAttention.from_reference(m, fold_bn=fold_bn, ktv_mode=ktv_mode))
    return len(targets)
