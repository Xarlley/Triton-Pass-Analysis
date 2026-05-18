import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, functional

class SimpleVGG16SNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 简化的 VGG16 结构，提取主要层用于测试 Triton Pass
        self.features = nn.Sequential(
            layer.Conv2d(3, 64, kernel_size=3, padding=1),
            layer.BatchNorm2d(64),
            neuron.LIFNode(),
            layer.MaxPool2d(2, 2),
            layer.Conv2d(64, 128, kernel_size=3, padding=1),
            layer.BatchNorm2d(128),
            neuron.LIFNode(),
            layer.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            layer.Flatten(),
            layer.Linear(128 * 56 * 56, 10),
            neuron.LIFNode()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def main():
    print("Setting up VGG16 SNN with T=4...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SimpleVGG16SNN().to(device)
    
    # 启用多步模式 T=4
    T = 4
    functional.set_step_mode(model, 'm')
    
    # 输入 shape: [T, N, C, H, W]
    x = torch.randn(T, 1, 3, 224, 224).to(device)
    
    print("Compiling model using torch.compile...")
    compiled_model = torch.compile(model)
    
    print("Running forward pass...")
    with torch.no_grad():
        out = compiled_model(x)
        
    print("Forward pass completed. Output shape:", out.shape)
    
if __name__ == '__main__':
    main()
