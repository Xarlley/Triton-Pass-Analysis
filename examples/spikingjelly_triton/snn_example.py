import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, surrogate

# 定义一个简单的脉冲神经网络
class SimpleSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = layer.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn = layer.BatchNorm2d(16)
        # 使用基于 surrogate gradient 的 LIF 神经元
        self.lif = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.pool = layer.MaxPool2d(2, 2)
        self.flatten = layer.Flatten()
        self.fc = layer.Linear(16 * 14 * 14, 10)
        self.lif2 = neuron.LIFNode(surrogate_function=surrogate.ATan())

    def forward(self, x):
        # x.shape: [batch_size, 1, 28, 28]
        x = self.conv(x)
        x = self.bn(x)
        x = self.lif(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.lif2(x)
        return x

def main():
    print("Initializing SimpleSNN on CUDA...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleSNN().to(device)

    # 包装为编译模式，重点观察后端 Triton 生成
    print("Compiling model using torch.compile (Triton backend)...")
    compiled_model = torch.compile(model)

    # 构造模拟输入 (Batch=4, Channels=1, H=28, W=28)
    x = torch.randn(4, 1, 28, 28).to(device)

    print("Running forward pass...")
    # 第一次前向传播会触发 AOTAutograd 和 Triton Kernel 生成
    out = compiled_model(x)
    print("Forward pass completed. Output shape:", out.shape)

    print("Running backward pass...")
    # 使用 sum() 构造标量 Loss，然后反向传播
    loss = out.sum()
    loss.backward()
    print("Backward pass completed. SNN gradients calculated.")

if __name__ == '__main__':
    main()
