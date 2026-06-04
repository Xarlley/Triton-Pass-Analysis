"""MobileNet-V2-style SNN：用 depthwise + pointwise conv + 倒残差展示框架对
分组卷积 + 残差的兼容性。

每个 InvertedResidual 块结构（expand_ratio>1）：
    1×1 expand-conv  → BN → Neuron
    3×3 dw-conv      → BN → Neuron
    1×1 project-conv → BN → (+ identity 当 stride=1 且 in_ch=out_ch) → Neuron

本框架直接支持：每个 (Conv→BN→Neuron) 都能融合，残差合流时用
FusedConvBNAddNeuron。当 stride > 1 / 通道不匹配（即没残差）时退化为普通
FusedConvBNNeuron。
"""
from __future__ import annotations

import torch
import torch.nn as nn

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


class InvertedResidualSNN(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio,
                 *, neuron="lif", tau=2.0, decay_input=True, soft_reset=False,
                 v_threshold=1.0, v_reset=0.0, layout="NCHW"):
        super().__init__()
        self.layout = layout
        self.stride = stride
        self.use_res_connect = stride == 1 and in_ch == out_ch
        hidden_dim = in_ch * expand_ratio
        nk = dict(neuron=neuron, tau=tau, decay_input=decay_input,
                  soft_reset=soft_reset, v_threshold=v_threshold,
                  v_reset=v_reset, layout=layout)

        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                _neuron(**nk),
            ]
        layers += [
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1,
                       groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            _neuron(**nk),
            nn.Conv2d(hidden_dim, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.body = nn.Sequential(*layers)
        # 最后一个 neuron：当有残差时由 FusedConvBNAddNeuron 接管，否则普通 neuron
        self.final_neuron = _neuron(**nk)
        self._fused = False
        self.neuron_kwargs = nk

    def fuse(self):
        from ..passes.fuse import fuse_snn_model, _neuron_kwargs
        # 把 body 里所有 (Conv→BN→Neuron) 用 Sequential pass 处理
        self.body, _ = fuse_snn_model(self.body, layout=self.layout)
        # 然后把最后的 (Conv, BN, final_neuron) → FusedConvBNAddNeuron / FusedConvBNNeuron
        # 当前 body 末尾是 Conv2d + BatchNorm2d；final_neuron 在 body 之外
        last_conv = None
        last_bn = None
        for m in list(self.body):
            if isinstance(m, nn.Conv2d):
                last_conv = m
                last_bn = None
            elif isinstance(m, nn.BatchNorm2d):
                last_bn = m
        # 重新组装 body：去掉末尾的 conv-bn
        kept = []
        for m in list(self.body):
            kept.append(m)
        # 倒数第二第三个是 Conv 与 BN（dwsep）
        # 这里改写策略：完全重建 body 列表，把末尾两层 + final_neuron 改成融合 module
        new_body = []
        items = list(self.body)
        if len(items) >= 2 and isinstance(items[-2], nn.Conv2d) \
           and isinstance(items[-1], nn.BatchNorm2d):
            new_body = items[:-2]
            conv = items[-2].eval(); bn = items[-1].eval()
            kw = _neuron_kwargs(self.final_neuron)
            kw["layout"] = self.layout
            dev, dt = conv.weight.device, conv.weight.dtype
            if self.use_res_connect:
                head = FusedConvBNAddNeuron(conv, bn, **kw).to(device=dev, dtype=dt)
            else:
                head = FusedConvBNNeuron(conv, bn, **kw).to(device=dev, dtype=dt)
            self.head = head
            self.body = nn.Sequential(*new_body)
            self.final_neuron = nn.Identity()
            self._fused = True
        return self

    def forward(self, x_seq):
        T, B = x_seq.shape[0], x_seq.shape[1]
        if self._fused:
            out = x_seq
            for layer in self.body:
                if isinstance(layer, (FusedConvBNNeuron,)):
                    out = layer(out)
                else:
                    # 朴素 dwsep/conv 等
                    out_4d = out.reshape(T * B, *out.shape[2:])
                    out_4d = layer(out_4d)
                    out = out_4d.view(T, B, *out_4d.shape[1:])
            if self.use_res_connect:
                return self.head(out, x_seq)
            return self.head(out)

        # 朴素路径
        out = x_seq
        for layer in self.body:
            if isinstance(layer, (IFNode, LIFNode)):
                out = layer(out)
            else:
                T_, B_ = out.shape[0], out.shape[1]
                out_4d = out.reshape(T_ * B_, *out.shape[2:])
                out_4d = layer(out_4d)
                out = out_4d.view(T_, B_, *out_4d.shape[1:])
        if self.use_res_connect:
            out = out + x_seq
        return self.final_neuron(out)


# MobileNet-V2 config: (expand_ratio, out_ch, n, stride)
MBV2_CFG = [
    (1,  16, 1, 1),
    (6,  24, 2, 2),
    (6,  32, 3, 2),
    (6,  64, 4, 2),
    (6,  96, 3, 1),
    (6, 160, 3, 2),
    (6, 320, 1, 1),
]


class MobileNetV2SNN(nn.Module):
    def __init__(self, num_classes=1000, *,
                 neuron="lif", tau=2.0, decay_input=True, soft_reset=False,
                 v_threshold=1.0, v_reset=0.0, layout="NCHW"):
        super().__init__()
        self.layout = layout
        nk = dict(neuron=neuron, tau=tau, decay_input=decay_input,
                  soft_reset=soft_reset, v_threshold=v_threshold,
                  v_reset=v_reset, layout=layout)

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            _neuron(**nk),
        )
        # InvertedResidual blocks
        blocks = []
        in_ch = 32
        for t, c, n, s in MBV2_CFG:
            for i in range(n):
                stride = s if i == 0 else 1
                blocks.append(InvertedResidualSNN(in_ch, c, stride, t, **nk))
                in_ch = c
        self.blocks = nn.Sequential(*blocks)
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            _neuron(**nk),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, num_classes)

    def fuse(self):
        from ..passes.fuse import fuse_snn_model
        # stem 与 head 都是 Sequential → 直接 fuse
        self.stem, _ = fuse_snn_model(self.stem, layout=self.layout)
        self.head, _ = fuse_snn_model(self.head, layout=self.layout)
        for blk in self.blocks:
            blk.fuse()
        return self

    def forward(self, x_seq):
        T, B = x_seq.shape[0], x_seq.shape[1]
        # stem: Sequential of Conv-BN-Neuron 或已融合
        x = x_seq
        for layer in self.stem:
            if isinstance(layer, (FusedConvBNNeuron, IFNode, LIFNode)):
                x = layer(x)
            else:
                T_, B_ = x.shape[0], x.shape[1]
                x_4d = x.reshape(T_ * B_, *x.shape[2:])
                x_4d = layer(x_4d)
                x = x_4d.view(T_, B_, *x_4d.shape[1:])
        for blk in self.blocks:
            x = blk(x)
        for layer in self.head:
            if isinstance(layer, (FusedConvBNNeuron, IFNode, LIFNode)):
                x = layer(x)
            else:
                T_, B_ = x.shape[0], x.shape[1]
                x_4d = x.reshape(T_ * B_, *x.shape[2:])
                x_4d = layer(x_4d)
                x = x_4d.view(T_, B_, *x_4d.shape[1:])
        T_, B_ = x.shape[0], x.shape[1]
        x_4d = x.reshape(T_ * B_, *x.shape[2:])
        x_4d = self.gap(x_4d).flatten(1)
        x = x_4d.view(T_, B_, -1)
        y = self.fc(x.reshape(T_ * B_, -1)).view(T_, B_, -1)
        return y


def _init_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.copy_(torch.randn_like(m.running_mean) * 0.1)
            m.running_var.copy_(torch.rand_like(m.running_var) + 0.5)
            m.weight.data.copy_(torch.rand_like(m.weight) + 0.5)
            m.bias.data.copy_(torch.randn_like(m.bias) * 0.1)


def mobilenet_v2_snn(*, fused=False, init_bn=True, **kw):
    m = MobileNetV2SNN(**kw)
    if init_bn: _init_bn(m)
    if fused: m.eval().fuse()
    return m
