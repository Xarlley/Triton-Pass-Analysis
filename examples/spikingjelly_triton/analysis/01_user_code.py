import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, surrogate

# ============================================================
# Level 1: User Python Code (High-Level Framework Layer)
# ============================================================
# 这是从用户视角能看到的唯一代码层。
# SpikingJelly 封装了所有的生物神经模型细节，用户只需像搭积木一样组合模块。
#
# 关键组件解析：
#   - layer.Conv2d / BatchNorm2d / MaxPool2d / Flatten / Linear
#       这些是对 PyTorch 原生层的包装 (SpikingJelly 的 activation_based.layer 模块)，
#       为多时间步 (multi-step) 推理模式提供了自动管理 T 维度的能力。
#
#   - neuron.LIFNode(surrogate_function=surrogate.ATan())
#       Leaky Integrate-and-Fire (LIF) 神经元。
#       在前向传播中，它实现 Heaviside 阈值函数 (脉冲发放)：
#           spike = (v >= threshold).float()
#       但 Heaviside 不可导！因此在反向传播时，它使用 ATan 作为替代梯度 (surrogate gradient)：
#           g'(x) = alpha / (2 * (1 + (pi/2 * alpha * x)^2))   (alpha=2.0)
#       这是 SNN 可训练性的核心：用一个光滑的函数近似不可导的 Heaviside 函数，
#       使得梯度可以正常反向传播。
#
# 对应 SpikingJelly 源代码路径:
#   - neuron/lif.py : class LIFNode (第96行)
#       LIFNode.neuronal_charge()    -> 实现 H[t] = V[t-1] + (X[t] - (V[t-1] - V_reset)) / tau
#       LIFNode.single_step_forward() -> 调用 super().single_step_forward(x) (backend="torch" 时)
#
#   - activation_based/surrogate.py : class ATan (第792行)
#       atan.forward()  : return heaviside(x)   <- 前向: 硬阈值
#       atan.backward() : return atan_backward() <- 反向: ATan 替代梯度
#       atan_backward() (第770行): a / (1 + ax * ax) * grad_output  where ax = pi * a * x
# ============================================================

class SimpleSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = layer.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn = layer.BatchNorm2d(16)
        # 带有反向传播 surrogate gradient（此处使用 ATan 替代梯度, alpha=2.0）的 LIF 神经元
        self.lif = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.pool = layer.MaxPool2d(2, 2)
        self.flatten = layer.Flatten()
        self.fc = layer.Linear(16 * 14 * 14, 10)
        self.lif2 = neuron.LIFNode(surrogate_function=surrogate.ATan())

    def forward(self, x):
        # x.shape: [batch_size, 1, 28, 28]
        x = self.conv(x)    # -> [4, 16, 28, 28]  Conv2d(1, 16, 3, padding=1)
        x = self.bn(x)      # -> [4, 16, 28, 28]  BatchNorm2d(16)
        x = self.lif(x)     # -> [4, 16, 28, 28]  LIF 脉冲, Heaviside + ATan surrogate
        x = self.pool(x)    # -> [4, 16, 14, 14]  MaxPool2d(2, 2)
        x = self.flatten(x) # -> [4, 3136]         Flatten
        x = self.fc(x)      # -> [4, 10]           Linear(3136, 10)
        x = self.lif2(x)    # -> [4, 10]           LIF 脉冲
        return x
