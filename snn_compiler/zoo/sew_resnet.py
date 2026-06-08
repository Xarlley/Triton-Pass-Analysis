"""SEW-ResNet (Spike-Element-Wise ResNet) 参考实现。

与标准 ResNet-SNN（见 ``zoo/resnet.py``）的**关键拓扑差异**——这正是直接套用
标准 ResNet 融合 module 会静默算错的原因：

| | 标准 ResNet-SNN | **SEW-ResNet** |
|---|---|---|
| 残差相加位置 | 在第二个 neuron **之前**：``neuron2(conv2_bn2(x) + identity)`` | 在第二个 neuron **之后**：``neuron2(conv2_bn2(x)) ⊕ identity`` |
| downsample 分支 | 仅 conv+bn，无 neuron | conv+bn **+ 自带一个 neuron**（输出也是脉冲） |
| ⊕ 连接函数 | 只有加法 | ADD(``+``) / AND(``*``) / IAND(``identity*(1-out)``) |

因此 SEW 块用 ``FusedConvBNNeuron``（×2 或 ×3）+ 一个**普通逐元素 ⊕**实现，
**不能**用 ``FusedConvBNAddNeuron``（那是"加在 neuron 之前"的标准 ResNet 语义）。

参考：Fang et al., "Deep Residual Learning in Spiking Neural Networks", NeurIPS 2021。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nn.modules import FusedConvBNNeuron
from .resnet import _neuron, _init_bn


def _sew(out, identity, connect_f):
    if connect_f == "ADD":
        return out + identity
    if connect_f == "AND":
        return out * identity
    if connect_f == "IAND":
        return identity * (1.0 - out)
    raise ValueError(f"unknown connect_f: {connect_f!r} (expected ADD/AND/IAND)")


class SEWBasicBlockSNN(nn.Module):
    """SEW BasicBlock-SNN。

    naive forward 等价于::

        out = neuron1(bn1(conv1(x)))
        out = neuron2(bn2(conv2(out)))            # 注意：neuron2 已发放
        identity = downsample_neuron(bn(conv(x))) if downsample else x
        out = sew(out, identity, connect_f)        # ⊕ 加在 neuron 之后

    fused=True 时三段 conv-bn-neuron 各换成一个 FusedConvBNNeuron，⊕ 仍是普通逐元素。
    """
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None,
                 *, connect_f="ADD", neuron="if", tau=2.0, decay_input=True,
                 soft_reset=False, v_threshold=1.0, v_reset=0.0, layout="NCHW"):
        super().__init__()
        self.connect_f = connect_f.upper()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.neuron1 = _neuron(neuron, tau=tau, decay_input=decay_input,
                               soft_reset=soft_reset, v_threshold=v_threshold,
                               v_reset=v_reset, layout=layout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.neuron2 = _neuron(neuron, tau=tau, decay_input=decay_input,
                               soft_reset=soft_reset, v_threshold=v_threshold,
                               v_reset=v_reset, layout=layout)
        self.downsample = downsample             # nn.Sequential(Conv2d, BN) 或 None
        self.downsample_neuron = (
            _neuron(neuron, tau=tau, decay_input=decay_input, soft_reset=soft_reset,
                    v_threshold=v_threshold, v_reset=v_reset, layout=layout)
            if downsample is not None else None)
        self.layout = layout
        self._fused = False

    def fuse(self, *, fold_bn: bool = True):
        from ..passes.fuse import _neuron_kwargs
        kw1 = _neuron_kwargs(self.neuron1); kw1["layout"] = self.layout
        kw2 = _neuron_kwargs(self.neuron2); kw2["layout"] = self.layout
        dev, dt = self.conv1.weight.device, self.conv1.weight.dtype
        self.block1 = FusedConvBNNeuron(self.conv1.eval(), self.bn1.eval(),
                                        fold_bn=fold_bn, **kw1).to(device=dev, dtype=dt)
        self.block2 = FusedConvBNNeuron(self.conv2.eval(), self.bn2.eval(),
                                        fold_bn=fold_bn, **kw2).to(device=dev, dtype=dt)
        if self.downsample is not None:
            ds_conv = next(m for m in self.downsample.modules() if isinstance(m, nn.Conv2d))
            ds_bn = next(m for m in self.downsample.modules() if isinstance(m, nn.BatchNorm2d))
            kwd = _neuron_kwargs(self.downsample_neuron); kwd["layout"] = self.layout
            self.ds_block = FusedConvBNNeuron(ds_conv.eval(), ds_bn.eval(),
                                              fold_bn=fold_bn, **kwd).to(device=dev, dtype=dt)
        else:
            self.ds_block = None
        self.conv1 = self.bn1 = self.neuron1 = nn.Identity()
        self.conv2 = self.bn2 = self.neuron2 = nn.Identity()
        self.downsample = self.downsample_neuron = None
        self._fused = True

    def forward(self, x_seq):
        T, B = x_seq.shape[0], x_seq.shape[1]
        if self._fused:
            out = self.block2(self.block1(x_seq))
            identity = x_seq if self.ds_block is None else self.ds_block(x_seq)
            return _sew(out, identity, self.connect_f)

        # naive
        if self.downsample is None:
            identity = x_seq
        else:
            x4 = x_seq.reshape(T * B, *x_seq.shape[2:])
            if self.layout == "NHWC":
                x4 = x4.contiguous(memory_format=torch.channels_last)
            id4 = self.downsample(x4)
            if self.layout == "NHWC":
                id4 = id4.contiguous(memory_format=torch.channels_last)
            identity = self.downsample_neuron(id4.view(T, B, *id4.shape[1:]))
        x4 = x_seq.reshape(T * B, *x_seq.shape[2:])
        out4 = self.bn1(self.conv1(x4))
        out = self.neuron1(out4.view(T, B, *out4.shape[1:]))
        out4 = self.bn2(self.conv2(out.reshape(T * B, *out.shape[2:])))
        out = self.neuron2(out4.view(T, B, *out4.shape[1:]))   # neuron2 已发放
        return _sew(out, identity, self.connect_f)


class SEWResNetSNN(nn.Module):
    """通用 SEW-ResNet-SNN 骨架（BasicBlock 系列）。输入 [T, B, 3, H, W]。"""

    def __init__(self, layers, num_classes=1000, *, in_ch=3, base_ch=64,
                 connect_f="ADD", neuron="if", tau=2.0, decay_input=True,
                 soft_reset=False, v_threshold=1.0, v_reset=0.0, layout="NCHW"):
        super().__init__()
        self.layout = layout
        self.connect_f = connect_f.upper()
        self.nk = dict(connect_f=self.connect_f, neuron=neuron, tau=tau,
                       decay_input=decay_input, soft_reset=soft_reset,
                       v_threshold=v_threshold, v_reset=v_reset, layout=layout)
        self.stem_conv = nn.Conv2d(in_ch, base_ch, 7, stride=2, padding=3, bias=False)
        self.stem_bn = nn.BatchNorm2d(base_ch)
        self.stem_neuron = _neuron(neuron, tau=tau, decay_input=decay_input,
                                   soft_reset=soft_reset, v_threshold=v_threshold,
                                   v_reset=v_reset, layout=layout)
        self.stem_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.in_ch = base_ch
        self.layer1 = self._make_layer(base_ch, layers[0], stride=1)
        self.layer2 = self._make_layer(base_ch * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_ch * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_ch * 8, layers[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_ch * 8, num_classes)
        self._fused = False

    def _make_layer(self, out_ch, n_blocks, stride):
        downsample = None
        if stride != 1 or self.in_ch != out_ch * SEWBasicBlockSNN.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_ch, out_ch * SEWBasicBlockSNN.expansion, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_ch * SEWBasicBlockSNN.expansion),
            )
        blocks = [SEWBasicBlockSNN(self.in_ch, out_ch, stride=stride,
                                   downsample=downsample, **self.nk)]
        self.in_ch = out_ch * SEWBasicBlockSNN.expansion
        for _ in range(1, n_blocks):
            blocks.append(SEWBasicBlockSNN(self.in_ch, out_ch, **self.nk))
        return nn.Sequential(*blocks)

    def fuse(self, *, fold_bn: bool = True):
        for m in self.modules():
            if isinstance(m, SEWBasicBlockSNN):
                m.fuse(fold_bn=fold_bn)
        dev, dt = self.stem_conv.weight.device, self.stem_conv.weight.dtype
        from ..passes.fuse import _neuron_kwargs
        kw = _neuron_kwargs(self.stem_neuron); kw["layout"] = self.layout
        self.stem_fused = FusedConvBNNeuron(self.stem_conv.eval(), self.stem_bn.eval(),
                                            fold_bn=fold_bn, **kw).to(device=dev, dtype=dt)
        self.stem_conv = self.stem_bn = self.stem_neuron = nn.Identity()
        self._fused = True
        return self

    def forward(self, x_seq):
        T, B = x_seq.shape[0], x_seq.shape[1]
        if self._fused:
            x = self.stem_fused(x_seq)
        else:
            x4 = x_seq.reshape(T * B, *x_seq.shape[2:])
            if self.layout == "NHWC":
                x4 = x4.contiguous(memory_format=torch.channels_last)
            x4 = self.stem_bn(self.stem_conv(x4))
            x = self.stem_neuron(x4.view(T, B, *x4.shape[1:]))
        x4 = x.reshape(T * B, *x.shape[2:])
        x4 = self.stem_pool(x4)
        x = x4.view(T, B, *x4.shape[1:])
        for stage in (self.layer1, self.layer2, self.layer3, self.layer4):
            for blk in stage:
                x = blk(x)
        x4 = x.reshape(T * B, *x.shape[2:])
        x4 = self.gap(x4).flatten(1)
        x = x4.view(T, B, -1)
        return self.fc(x.reshape(T * B, -1)).view(T, B, -1)


def sew_resnet18_snn(*, fused=False, fold_bn=True, init_bn=True, **kw):
    m = SEWResNetSNN([2, 2, 2, 2], **kw)
    if init_bn: _init_bn(m)
    if fused: m.eval().fuse(fold_bn=fold_bn)
    return m


def sew_resnet34_snn(*, fused=False, fold_bn=True, init_bn=True, **kw):
    m = SEWResNetSNN([3, 4, 6, 3], **kw)
    if init_bn: _init_bn(m)
    if fused: m.eval().fuse(fold_bn=fold_bn)
    return m
