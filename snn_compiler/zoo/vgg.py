"""VGG-style SNN 参考实现。

支持 VGG-D (16)、VGG-19、以及含/不含 BN 的两类变体。

构造方式：
    >>> from snn_compiler.zoo.vgg import vgg16_snn
    >>> m = vgg16_snn(num_classes=1000, neuron="lif", soft_reset=False,
    ...                with_bn=True, layout="NHWC", fused=True).cuda().eval()
    >>> y = m(x_seq)   # x_seq: [T, B, 3, 224, 224]

fused=True 时，所有可融合层（Conv→BN→Neuron / Linear→Neuron）已经在构造
阶段调用 fuse_snn_model 替换为本框架的 fused module。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nn.modules import IFNode, LIFNode, FusedConvBNNeuron, FusedConvNeuron, FusedLinearNeuron
from ..passes.fuse import fuse_snn_model


# 通用 VGG 配置：数字 = 输出通道；"P" = pooling
VGG16_CFG = [64, 64, "P", 128, 128, "P", 256, 256, 256, "P",
             512, 512, 512, "P", 512, 512, 512, "P"]
VGG19_CFG = [64, 64, "P", 128, 128, "P",
             256, 256, 256, 256, "P",
             512, 512, 512, 512, "P",
             512, 512, 512, 512, "P"]
VGG11_CFG = [64, "P", 128, "P", 256, 256, "P", 512, 512, "P", 512, 512, "P"]
VGG13_CFG = [64, 64, "P", 128, 128, "P", 256, 256, "P", 512, 512, "P", 512, 512, "P"]


def _neuron_factory(neuron, *, tau, decay_input, soft_reset, v_threshold, v_reset, layout):
    if neuron == "if":
        return IFNode(soft_reset=soft_reset, v_threshold=v_threshold,
                      v_reset=v_reset, layout=layout)
    return LIFNode(tau=tau, decay_input=decay_input, soft_reset=soft_reset,
                    v_threshold=v_threshold, v_reset=v_reset, layout=layout)


class VGGSNN(nn.Module):
    """通用 VGG-SNN 骨架。

    forward(x_seq) -> [T, B, num_classes]
       x_seq: [T, B, 3, H, W]
    """
    def __init__(self, cfg, num_classes=1000, *, in_ch=3, fc_hidden=4096,
                 with_bn=True, neuron="lif", tau=2.0, decay_input=True,
                 soft_reset=False, v_threshold=1.0, v_reset=0.0,
                 layout="NCHW", pool="avg"):
        super().__init__()
        feats = []
        ch = in_ch
        pool_layer = nn.AvgPool2d if pool == "avg" else nn.MaxPool2d
        for v in cfg:
            if v == "P":
                feats.append(pool_layer(2, 2))
            else:
                feats.append(nn.Conv2d(ch, v, 3, padding=1, bias=True))
                if with_bn:
                    feats.append(nn.BatchNorm2d(v))
                feats.append(_neuron_factory(
                    neuron, tau=tau, decay_input=decay_input,
                    soft_reset=soft_reset, v_threshold=v_threshold,
                    v_reset=v_reset, layout=layout,
                ))
                ch = v
        self.features = nn.Sequential(*feats)
        # 推断 flatten 后 feature dim：cfg 中 P 的个数决定空间分辨率折半次数
        n_pool = sum(1 for x in cfg if x == "P")
        # 默认输入 224×224 → 224 / 2**n_pool
        self.fc_in_dim = ch * (224 // 2**n_pool) ** 2
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_in_dim, fc_hidden),
            _neuron_factory(neuron, tau=tau, decay_input=decay_input,
                            soft_reset=soft_reset, v_threshold=v_threshold,
                            v_reset=v_reset, layout="NCHW"),
            nn.Linear(fc_hidden, fc_hidden),
            _neuron_factory(neuron, tau=tau, decay_input=decay_input,
                            soft_reset=soft_reset, v_threshold=v_threshold,
                            v_reset=v_reset, layout="NCHW"),
            nn.Linear(fc_hidden, num_classes),
        )
        self.layout = layout

    def forward(self, x_seq):
        T, B = x_seq.shape[0], x_seq.shape[1]
        x = x_seq
        for layer in self.features:
            if isinstance(layer, (FusedConvBNNeuron, FusedConvNeuron, IFNode, LIFNode)):
                # 接受 [T, B, ...] 输入
                x = layer(x)
            elif isinstance(layer, (nn.Conv2d, nn.BatchNorm2d)):
                # 朴素路径：[T*B, C, H, W]
                T_, B_ = x.shape[0], x.shape[1]
                x = x.reshape(T_ * B_, *x.shape[2:])
                x = layer(x)
                x = x.view(T_, B_, *x.shape[1:])
            else:
                # Pool 等：当作 [T*B, C, H, W] 即可
                T_, B_ = x.shape[0], x.shape[1]
                x = layer(x.reshape(T_ * B_, *x.shape[2:]))
                x = x.view(T_, B_, *x.shape[1:])
        # flatten 进 fc
        x = x.contiguous().reshape(T, B, -1)
        for layer in self.classifier:
            if isinstance(layer, (FusedLinearNeuron, IFNode, LIFNode)):
                x = layer(x)
            elif isinstance(layer, nn.Linear):
                T_, B_ = x.shape[0], x.shape[1]
                x = layer(x.reshape(T_ * B_, -1))
                x = x.view(T_, B_, -1)
        return x


def _maybe_init_bn(model):
    """给 BN 写入有意义的 running stats（用于在合成基准里替代真实训练数据）。"""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.copy_(torch.randn_like(m.running_mean) * 0.1)
            m.running_var.copy_(torch.rand_like(m.running_var) + 0.5)
            m.weight.data.copy_(torch.rand_like(m.weight) + 0.5)
            m.bias.data.copy_(torch.randn_like(m.bias) * 0.1)


def vgg11_snn(**kw): return _build(VGG11_CFG, **kw)
def vgg13_snn(**kw): return _build(VGG13_CFG, **kw)
def vgg16_snn(**kw): return _build(VGG16_CFG, **kw)
def vgg19_snn(**kw): return _build(VGG19_CFG, **kw)


def _build(cfg, *, fused=False, init_bn=True, **kwargs):
    m = VGGSNN(cfg, **kwargs)
    if init_bn:
        _maybe_init_bn(m)
    if fused:
        m.eval()
        layout = kwargs.get("layout", "NCHW")
        m, n = fuse_snn_model(m, layout=layout)
        m._n_fused = n
    return m
