"""ResNet-style SNN 参考实现。

ResNet 与 VGG 的本质差别：每个 BasicBlock 含一个残差求和与一个 post-add neuron。
本框架用 FusedConvBNAddNeuron 把 "conv2-bn2 + identity → neuron2" 三步并到一个
Triton kernel；FusedConvBNNeuron 处理"conv1-bn1 → neuron1"。

支持：ResNet-18 / ResNet-34（BasicBlock，每残差路径 2 conv），后续可扩展到
ResNet-50/101（Bottleneck，每路径 3 conv）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nn.modules import (
    IFNode, LIFNode,
    FusedConvBNNeuron, FusedConvBNAddNeuron,
)


def _neuron(neuron, *, tau, decay_input, soft_reset, v_threshold, v_reset, layout):
    if neuron == "if":
        return IFNode(soft_reset=soft_reset, v_threshold=v_threshold,
                      v_reset=v_reset, layout=layout)
    return LIFNode(tau=tau, decay_input=decay_input, soft_reset=soft_reset,
                    v_threshold=v_threshold, v_reset=v_reset, layout=layout)


class BasicBlockSNN(nn.Module):
    """ResNet BasicBlock-SNN，已就地为本框架的融合 module 形态。

    naive forward 等价于:
       identity = downsample(x) if downsample else x
       out = neuron1(bn1(conv1(x)))
       out = neuron2(bn2(conv2(out)) + identity)

    fused=True 时:
       block1 = FusedConvBNNeuron(conv1, bn1)
       block2 = FusedConvBNAddNeuron(conv2, bn2)
       forward:  identity = downsample(x); out = block1(x); out = block2(out, identity)
    """
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None,
                 *, neuron="lif", tau=2.0, decay_input=True, soft_reset=False,
                 v_threshold=1.0, v_reset=0.0, layout="NCHW"):
        super().__init__()
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
        self.downsample = downsample        # nn.Sequential(Conv2d, BN) 或 None
        self.layout = layout
        self._fused = False

    def fuse(self):
        """就地把本 block 替换为融合形式。要求处于 eval 状态。"""
        from ..passes.fuse import _neuron_kwargs
        kw1 = _neuron_kwargs(self.neuron1); kw1["layout"] = self.layout
        kw2 = _neuron_kwargs(self.neuron2); kw2["layout"] = self.layout
        # 用 conv weight 的 device/dtype 保证 to() 正确
        dev, dt = self.conv1.weight.device, self.conv1.weight.dtype
        self.block1 = FusedConvBNNeuron(self.conv1.eval(), self.bn1.eval(), **kw1).to(
            device=dev, dtype=dt
        )
        self.block2 = FusedConvBNAddNeuron(self.conv2.eval(), self.bn2.eval(), **kw2).to(
            device=dev, dtype=dt
        )
        # 让 downsample 仍然是普通 conv-bn（没融 neuron）；保留原样
        self.conv1 = nn.Identity()
        self.bn1 = nn.Identity()
        self.neuron1 = nn.Identity()
        self.conv2 = nn.Identity()
        self.bn2 = nn.Identity()
        self.neuron2 = nn.Identity()
        self._fused = True

    def forward(self, x_seq):
        # x_seq: [T, B, C, H, W]
        T, B = x_seq.shape[0], x_seq.shape[1]
        if self.downsample is None:
            identity = x_seq
        else:
            x_4d = x_seq.reshape(T * B, *x_seq.shape[2:])
            if self.layout == "NHWC":
                x_4d = x_4d.contiguous(memory_format=torch.channels_last)
            id_4d = self.downsample(x_4d)
            if self.layout == "NHWC":
                id_4d = id_4d.contiguous(memory_format=torch.channels_last)
            identity = id_4d.view(T, B, *id_4d.shape[1:])

        if self._fused:
            out = self.block1(x_seq)
            out = self.block2(out, identity)
            return out

        # 朴素路径
        x_4d = x_seq.reshape(T * B, *x_seq.shape[2:])
        out = self.bn1(self.conv1(x_4d))
        out = out.view(T, B, *out.shape[1:])
        out = self.neuron1(out)
        out_4d = out.reshape(T * B, *out.shape[2:])
        out_4d = self.bn2(self.conv2(out_4d))
        out = out_4d.view(T, B, *out_4d.shape[1:])
        out = self.neuron2(out + identity)
        return out


class ResNetSNN(nn.Module):
    """通用 ResNet-SNN 骨架（BasicBlock 系列）。"""
    def __init__(self, layers, num_classes=1000, *,
                 in_ch=3, base_ch=64,
                 neuron="lif", tau=2.0, decay_input=True, soft_reset=False,
                 v_threshold=1.0, v_reset=0.0, layout="NCHW"):
        super().__init__()
        self.layout = layout
        self.neuron_kwargs = dict(
            neuron=neuron, tau=tau, decay_input=decay_input,
            soft_reset=soft_reset, v_threshold=v_threshold, v_reset=v_reset,
            layout=layout,
        )
        # Stem
        self.stem_conv = nn.Conv2d(in_ch, base_ch, 7, stride=2, padding=3, bias=False)
        self.stem_bn = nn.BatchNorm2d(base_ch)
        self.stem_neuron = _neuron(**self.neuron_kwargs)
        self.stem_pool = nn.MaxPool2d(3, stride=2, padding=1)

        self.in_ch = base_ch
        self.layer1 = self._make_layer(base_ch, layers[0], stride=1)
        self.layer2 = self._make_layer(base_ch * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_ch * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_ch * 8, layers[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_ch * 8, num_classes)

    def _make_layer(self, out_ch, n_blocks, stride):
        downsample = None
        if stride != 1 or self.in_ch != out_ch * BasicBlockSNN.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_ch, out_ch * BasicBlockSNN.expansion, 1,
                           stride=stride, bias=False),
                nn.BatchNorm2d(out_ch * BasicBlockSNN.expansion),
            )
        blocks = [BasicBlockSNN(self.in_ch, out_ch, stride=stride,
                                  downsample=downsample, **self.neuron_kwargs)]
        self.in_ch = out_ch * BasicBlockSNN.expansion
        for _ in range(1, n_blocks):
            blocks.append(BasicBlockSNN(self.in_ch, out_ch, **self.neuron_kwargs))
        return nn.Sequential(*blocks)

    def fuse(self):
        """递归融合每个 BasicBlock。Stem 仍走 Sequential pass。"""
        for m in self.modules():
            if isinstance(m, BasicBlockSNN):
                m.fuse()
        # Stem 是相邻的 conv-bn-neuron，可由 fuse_snn_model 处理 — 但 stem 在
        # 顶层不是 nn.Sequential，所以这里直接构造融合 module。
        kw = dict(self.neuron_kwargs)
        dev, dt = self.stem_conv.weight.device, self.stem_conv.weight.dtype
        stem_fused = FusedConvBNNeuron(
            self.stem_conv.eval(), self.stem_bn.eval(),
            neuron=kw["neuron"], tau=kw["tau"], decay_input=kw["decay_input"],
            soft_reset=kw["soft_reset"], v_threshold=kw["v_threshold"],
            v_reset=kw["v_reset"], layout=kw["layout"],
        ).to(device=dev, dtype=dt)
        self.stem_fused = stem_fused
        self.stem_conv = nn.Identity()
        self.stem_bn = nn.Identity()
        self.stem_neuron = nn.Identity()
        self._fused = True
        return self

    def _stem_naive(self, x_seq):
        T, B = x_seq.shape[0], x_seq.shape[1]
        x_4d = x_seq.reshape(T * B, *x_seq.shape[2:])
        if self.layout == "NHWC":
            x_4d = x_4d.contiguous(memory_format=torch.channels_last)
        x_4d = self.stem_bn(self.stem_conv(x_4d))
        x = x_4d.view(T, B, *x_4d.shape[1:])
        x = self.stem_neuron(x)
        return x

    def forward(self, x_seq):
        T, B = x_seq.shape[0], x_seq.shape[1]
        if getattr(self, "_fused", False):
            x = self.stem_fused(x_seq)
        else:
            x = self._stem_naive(x_seq)
        x_4d = x.reshape(T * B, *x.shape[2:])
        x_4d = self.stem_pool(x_4d)
        x = x_4d.view(T, B, *x_4d.shape[1:])
        for stage in (self.layer1, self.layer2, self.layer3, self.layer4):
            for blk in stage:
                x = blk(x)
        x_4d = x.reshape(T * B, *x.shape[2:])
        x_4d = self.gap(x_4d).flatten(1)
        x = x_4d.view(T, B, -1)
        # fc 不带 neuron，最后一层走线性分类头
        T_, B_ = x.shape[0], x.shape[1]
        y = self.fc(x.reshape(T_ * B_, -1)).view(T_, B_, -1)
        return y


def _init_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.copy_(torch.randn_like(m.running_mean) * 0.1)
            m.running_var.copy_(torch.rand_like(m.running_var) + 0.5)
            m.weight.data.copy_(torch.rand_like(m.weight) + 0.5)
            m.bias.data.copy_(torch.randn_like(m.bias) * 0.1)


def resnet18_snn(*, fused=False, init_bn=True, **kw):
    m = ResNetSNN([2, 2, 2, 2], **kw)
    if init_bn: _init_bn(m)
    if fused: m.eval().fuse()
    return m


def resnet34_snn(*, fused=False, init_bn=True, **kw):
    m = ResNetSNN([3, 4, 6, 3], **kw)
    if init_bn: _init_bn(m)
    if fused: m.eval().fuse()
    return m
