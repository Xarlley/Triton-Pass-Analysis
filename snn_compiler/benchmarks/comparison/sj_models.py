"""SpikingJelly-equivalent reference SNN 网络。

每个模型与 snn_compiler.zoo 中的同名网络**逐层对应**：
- 通道数 / kernel / stride / padding 完全相同
- BN 接同位置
- LIF 参数（tau, v_threshold, v_reset, decay_input）相同

两个变体：
- `mode='eager'`：每个 LIFNode 用 step_mode='m', backend='triton'（SJ 多步快路径）。
  forward 接 5D [T, B, C, H, W] 直接送 SJ 多步。
- `mode='compile'`：每个 LIFNode 用 step_mode='s', backend='torch'。
  forward 自己写 for-t 循环（torch.compile 友好）。

注意：SJ 没有 conv-bn fold；BN 走 BatchNorm2d 单独 layer。这是 SJ 的"baseline"行为。
"""
from __future__ import annotations

import sys, pathlib
HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3] / "spikingjelly"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, layer, functional, surrogate


def _make_neuron(tau, v_threshold, v_reset, decay_input, *, mode):
    """根据 mode 构造合适的 LIFNode。"""
    if mode == "eager":
        n = neuron.LIFNode(tau=tau, decay_input=decay_input,
                            v_threshold=v_threshold, v_reset=v_reset,
                            step_mode='m', backend='triton',
                            surrogate_function=surrogate.ATan())
    else:
        # compile 模式：用 single-step + torch 后端，外层 for-t 循环
        n = neuron.LIFNode(tau=tau, decay_input=decay_input,
                            v_threshold=v_threshold, v_reset=v_reset,
                            step_mode='s', backend='torch',
                            surrogate_function=surrogate.ATan())
    return n


# ============================================================
# VGG-16 SNN
# ============================================================
SJ_VGG16_CFG = [64, 64, "P", 128, 128, "P", 256, 256, 256, "P",
                512, 512, 512, "P", 512, 512, 512, "P"]


class SJVGG16(nn.Module):
    def __init__(self, num_classes=1000, *, tau=2.0, v_threshold=1.0,
                 v_reset=0.0, decay_input=True, mode="eager"):
        super().__init__()
        self.mode = mode
        feats = []
        in_ch = 3
        for v in SJ_VGG16_CFG:
            if v == "P":
                feats.append(nn.AvgPool2d(2, 2))
            else:
                feats.append(nn.Conv2d(in_ch, v, 3, padding=1, bias=True))
                feats.append(nn.BatchNorm2d(v))
                feats.append(_make_neuron(tau, v_threshold, v_reset,
                                            decay_input, mode=mode))
                in_ch = v
        self.features = nn.Sequential(*feats)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            _make_neuron(tau, v_threshold, v_reset, decay_input, mode=mode),
            nn.Linear(4096, 4096),
            _make_neuron(tau, v_threshold, v_reset, decay_input, mode=mode),
            nn.Linear(4096, num_classes),
        )

    def _forward_step(self, x):
        """[B, C, H, W] -> [B, num_classes]，用于 compile 模式的 for-t 循环里。"""
        h = self.features(x)
        h = h.flatten(1)
        h = self.classifier(h)
        return h

    def forward(self, x_seq):
        # x_seq: [T, B, 3, H, W]
        if self.mode == "eager":
            # 多步：features (Sequential) 含 LIFNode 接 5D [T,B,...] (step_mode='m' 自动)
            # 但 nn.Conv2d 不接 5D；需要 reshape T*B
            T, B = x_seq.shape[0], x_seq.shape[1]
            # 让 features 走 multi-step layer：用 SJ 的 functional.multi_step_forward 包装
            return self._forward_multistep(x_seq)
        else:
            return self._forward_loop_t(x_seq)

    def _forward_multistep(self, x_seq):
        """SJ multistep：逐层判断；conv/bn 走 nn.Conv2d 单步，LIF 走 step_mode='m'。
        统一策略：把 x 一直保持 5D [T,B,...]；遇 nn.Conv2d/BN/Linear/AvgPool/Flatten 时
        reshape T*B → 计算 → reshape 回；遇 LIFNode 直接送 [T,B,...]。
        """
        x = x_seq
        for m in self.features:
            if isinstance(m, neuron.BaseNode):
                x = m(x)              # step_mode='m' 接 [T,B,C,H,W]
            else:
                T, B = x.shape[0], x.shape[1]
                x = m(x.reshape(T * B, *x.shape[2:]))
                x = x.view(T, B, *x.shape[1:])
        # flatten
        T, B = x.shape[0], x.shape[1]
        x = x.contiguous().view(T, B, -1)
        for m in self.classifier:
            if isinstance(m, neuron.BaseNode):
                x = m(x)
            else:
                T, B = x.shape[0], x.shape[1]
                x = m(x.reshape(T * B, -1))
                x = x.view(T, B, -1)
        return x

    def _forward_loop_t(self, x_seq):
        """compile 模式：外层 for t in range(T)，每步过 conv-bn-... -fc，
        LIFNode 是 step_mode='s'，内部维护 v 状态（functional.reset_net 调用方负责）。
        """
        T = x_seq.shape[0]
        outs = []
        for t in range(T):
            outs.append(self._forward_step(x_seq[t]))
        return torch.stack(outs, dim=0)


# ============================================================
# ResNet-{18,34} SNN
# ============================================================
class SJBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1, downsample=None,
                  *, tau, v_threshold, v_reset, decay_input, mode):
        super().__init__()
        self.mode = mode
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.neuron1 = _make_neuron(tau, v_threshold, v_reset, decay_input, mode=mode)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.neuron2 = _make_neuron(tau, v_threshold, v_reset, decay_input, mode=mode)
        self.downsample = downsample

    def forward_step(self, x):
        """[B, C, H, W] -> [B, C', H', W']  (compile 模式调用)"""
        identity = x if self.downsample is None else self.downsample(x)
        out = self.neuron1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.neuron2(out + identity)
        return out

    def forward_multistep(self, x_seq):
        """[T, B, C, H, W] -> [T, B, C', H', W']  (eager 模式调用)"""
        T, B = x_seq.shape[0], x_seq.shape[1]
        if self.downsample is None:
            identity = x_seq
        else:
            id_4d = self.downsample(x_seq.reshape(T * B, *x_seq.shape[2:]))
            identity = id_4d.view(T, B, *id_4d.shape[1:])
        out_4d = self.bn1(self.conv1(x_seq.reshape(T * B, *x_seq.shape[2:])))
        out = out_4d.view(T, B, *out_4d.shape[1:])
        out = self.neuron1(out)                            # step_mode='m'
        out_4d = self.bn2(self.conv2(out.reshape(T * B, *out.shape[2:])))
        out = out_4d.view(T, B, *out_4d.shape[1:])
        out = self.neuron2(out + identity)
        return out

    def forward(self, x):
        if self.mode == "eager":
            return self.forward_multistep(x)
        return self.forward_step(x)


class SJResNet(nn.Module):
    def __init__(self, layers, num_classes=1000, *,
                 tau=2.0, v_threshold=1.0, v_reset=0.0, decay_input=True,
                 mode="eager"):
        super().__init__()
        self.mode = mode
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.decay_input = decay_input
        # Stem
        self.stem_conv = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.stem_bn = nn.BatchNorm2d(64)
        self.stem_neuron = _make_neuron(tau, v_threshold, v_reset, decay_input,
                                         mode=mode)
        self.stem_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.in_ch = 64
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_ch, n, stride):
        downsample = None
        if stride != 1 or self.in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        blocks = [SJBasicBlock(self.in_ch, out_ch, stride=stride,
                                downsample=downsample,
                                tau=self.tau, v_threshold=self.v_threshold,
                                v_reset=self.v_reset, decay_input=self.decay_input,
                                mode=self.mode)]
        self.in_ch = out_ch
        for _ in range(1, n):
            blocks.append(SJBasicBlock(self.in_ch, out_ch,
                                        tau=self.tau, v_threshold=self.v_threshold,
                                        v_reset=self.v_reset, decay_input=self.decay_input,
                                        mode=self.mode))
        return nn.Sequential(*blocks)

    def _forward_step(self, x):
        h = self.stem_bn(self.stem_conv(x))
        h = self.stem_neuron(h)
        h = self.stem_pool(h)
        for stage in (self.layer1, self.layer2, self.layer3, self.layer4):
            for blk in stage:
                h = blk(h)
        h = self.gap(h).flatten(1)
        return self.fc(h)

    def _forward_multistep(self, x_seq):
        T, B = x_seq.shape[0], x_seq.shape[1]
        x4 = x_seq.reshape(T * B, *x_seq.shape[2:])
        x4 = self.stem_bn(self.stem_conv(x4))
        x = x4.view(T, B, *x4.shape[1:])
        x = self.stem_neuron(x)
        x4 = x.reshape(T * B, *x.shape[2:])
        x4 = self.stem_pool(x4)
        x = x4.view(T, B, *x4.shape[1:])
        for stage in (self.layer1, self.layer2, self.layer3, self.layer4):
            for blk in stage:
                x = blk(x)
        x4 = x.reshape(T * B, *x.shape[2:])
        x4 = self.gap(x4).flatten(1)
        x = x4.view(T, B, -1)
        T_, B_ = x.shape[0], x.shape[1]
        y = self.fc(x.reshape(T_ * B_, -1)).view(T_, B_, -1)
        return y

    def forward(self, x_seq):
        if self.mode == "eager":
            return self._forward_multistep(x_seq)
        T = x_seq.shape[0]
        outs = [self._forward_step(x_seq[t]) for t in range(T)]
        return torch.stack(outs, dim=0)


def sj_vgg16(**kw): return SJVGG16(**kw)
def sj_resnet18(**kw): return SJResNet([2, 2, 2, 2], **kw)
def sj_resnet34(**kw): return SJResNet([3, 4, 6, 3], **kw)


def init_bn_running_stats(model):
    """与 snn_compiler.zoo 用同种方式初始化 running stats，便于公平比较。"""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.copy_(torch.randn_like(m.running_mean) * 0.1)
            m.running_var.copy_(torch.rand_like(m.running_var) + 0.5)
            m.weight.data.copy_(torch.rand_like(m.weight) + 0.5)
            m.bias.data.copy_(torch.randn_like(m.bias) * 0.1)


def reset_sj_state(model):
    """SJ 在每次新 forward batch 前需 reset 神经元 v 状态。"""
    functional.reset_net(model)
