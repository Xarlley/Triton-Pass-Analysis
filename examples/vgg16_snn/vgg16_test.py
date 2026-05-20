"""
vgg16_test.py — 标准 VGG16 结构的脉冲神经网络（SNN）推理测试。

用途：为 dev-plan §2.1（时间/空间拆分 Pass）的开发提供一个**可复现的基准**。
本脚本使用「随机但固定保存」的权重与输入，保证每次推理输出完全一致，从而可在
Pass 改动前后比对结果，确认变化后的 IR 没有引入计算错误。

结构（标准 VGG16-D，共 16 个权重层）：
  - 特征提取：13 个 3x3 卷积，分 5 个 block，每个 block 末尾接 MaxPool；
              每个卷积后接 BatchNorm + LIF 脉冲神经元（替代 VGG 原本的 ReLU）。
  - 分类器：  3 个全连接层 4096-4096-1000，前两层后接 LIF，末层为线性读出
              （与 VGG16 末层 fc8 一致，输出 logits，便于数值级别的正确性比对）。

可复现机制：
  - 首次运行：用固定 seed 生成随机权重与输入，分别保存为 .pth 文件。
  - 之后运行：直接加载已保存的 .pth，保证权重/输入逐位一致。
  - 推理在 eval() 模式下进行（BN 使用 running stats），前向前调用 reset_net
    复位 LIF 膜电位状态。
  - 首次保存「黄金输出」，之后每次与之比对并报告是否一致。

注意：权重文件约 530MB，超过 GitHub 单文件上限，已通过 .gitignore 排除；
      由固定 seed 决定，删除后再次运行会确定性地重新生成。

运行：
    python vgg16_test.py
"""
import os
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, functional

# ----------------------------- 配置 -----------------------------
SEED = 42
T = 4                              # SNN 时间步
NUM_CLASSES = 1000                 # ImageNet 类别数
INPUT_SHAPE = (T, 1, 3, 224, 224)  # [T, N, C, H, W]

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(HERE, "vgg16_snn_weights.pth")
INPUT_PATH = os.path.join(HERE, "vgg16_snn_input.pth")
OUTPUT_PATH = os.path.join(HERE, "vgg16_snn_output.pth")

# VGG16-D 卷积配置：数字 = 输出通道数，'M' = MaxPool2d(2, 2)
VGG16_CFG = [64, 64, 'M',
             128, 128, 'M',
             256, 256, 256, 'M',
             512, 512, 512, 'M',
             512, 512, 512, 'M']


class VGG16SNN(nn.Module):
    """标准 VGG16 结构的脉冲神经网络（13 卷积 + 3 全连接）。"""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = self._make_features(VGG16_CFG)
        self.classifier = nn.Sequential(
            layer.Flatten(),
            layer.Linear(512 * 7 * 7, 4096),
            neuron.LIFNode(),
            layer.Linear(4096, 4096),
            neuron.LIFNode(),
            layer.Linear(4096, num_classes),   # 线性读出，与 VGG16 末层一致
        )

    @staticmethod
    def _make_features(cfg):
        layers, in_ch = [], 3
        for v in cfg:
            if v == 'M':
                layers.append(layer.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(layer.Conv2d(in_ch, v, kernel_size=3, padding=1))
                layers.append(layer.BatchNorm2d(v))
                layers.append(neuron.LIFNode())   # 替代 VGG 的 ReLU
                in_ch = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def describe(model):
    """打印并断言模型确为标准 VGG16 结构。"""
    n_conv = sum(1 for m in model.modules() if isinstance(m, layer.Conv2d))
    n_fc = sum(1 for m in model.modules() if isinstance(m, layer.Linear))
    n_lif = sum(1 for m in model.modules() if isinstance(m, neuron.LIFNode))
    n_param = sum(p.numel() for p in model.parameters())
    print(f"  结构: {n_conv} 卷积层 + {n_fc} 全连接层 "
          f"(= {n_conv + n_fc} 个权重层) | LIF 神经元层: {n_lif}")
    print(f"  参数量: {n_param:,} ({n_param / 1e6:.1f}M)")
    assert n_conv == 13 and n_fc == 3, \
        f"结构不符合标准 VGG16（应为 13 卷积 + 3 全连接，实为 {n_conv}+{n_fc}）"
    print("  ✅ 已确认为标准 VGG16 结构")


def build_model(device):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = VGG16SNN(NUM_CLASSES)
    functional.set_step_mode(model, 'm')   # 多步（multi-step）模式
    model.eval()                           # 推理模式：BN 使用 running stats
    return model.to(device)


def load_or_create_weights(model):
    if os.path.exists(WEIGHTS_PATH):
        print(f"  加载已保存权重: {os.path.basename(WEIGHTS_PATH)}")
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
    else:
        print(f"  首次运行：用 seed={SEED} 生成随机权重并保存 -> "
              f"{os.path.basename(WEIGHTS_PATH)}")
        torch.save(model.state_dict(), WEIGHTS_PATH)


def load_or_create_input(device):
    if os.path.exists(INPUT_PATH):
        print(f"  加载已保存输入: {os.path.basename(INPUT_PATH)}")
        x = torch.load(INPUT_PATH, map_location='cpu')
    else:
        print(f"  首次运行：用 seed={SEED} 生成随机输入并保存 -> "
              f"{os.path.basename(INPUT_PATH)}")
        g = torch.Generator().manual_seed(SEED)
        x = torch.randn(*INPUT_SHAPE, generator=g)
        torch.save(x, INPUT_PATH)
    return x.to(device)


def check_golden(out):
    """与黄金参考输出比对；首次运行则保存。"""
    out = out.detach().cpu()
    if os.path.exists(OUTPUT_PATH):
        golden = torch.load(OUTPUT_PATH, map_location='cpu')
        if torch.equal(out, golden):
            print("  ✅ 与黄金输出逐位一致 —— 推理结果可复现")
        else:
            diff = (out.float() - golden.float()).abs().max().item()
            ok = torch.allclose(out, golden, rtol=1e-4, atol=1e-5)
            tag = "在容差内一致" if ok else "❌ 超出容差，结果发生变化"
            print(f"  ⚠️ 与黄金输出不逐位相同，最大差异 {diff:.3e} —— {tag}")
    else:
        torch.save(out, OUTPUT_PATH)
        print(f"  首次运行：已保存黄金参考输出 -> {os.path.basename(OUTPUT_PATH)}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device} | 时间步 T={T}")

    print("[1/5] 构建 VGG16-SNN 模型...")
    model = build_model(device)
    describe(model)

    print("[2/5] 准备权重（随机但固定保存）...")
    load_or_create_weights(model)

    print("[3/5] 准备输入...")
    x = load_or_create_input(device)
    print(f"  输入形状: {tuple(x.shape)}")

    print("[4/5] torch.compile 编译并执行推理...")
    compiled = torch.compile(model)
    functional.reset_net(model)            # 复位 LIF 膜电位状态
    with torch.no_grad():
        out = compiled(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    print("[5/5] 校验结果...")
    print(f"  输出形状: {tuple(out.shape)}  (= [T, N, NUM_CLASSES])")
    out_cpu = out.detach().cpu().float()
    print(f"  输出统计: sum={out_cpu.sum().item():.6f}  "
          f"mean={out_cpu.mean().item():.6f}")
    pred = out_cpu.mean(dim=0).argmax(dim=-1)   # 对 T 求平均后取类别
    print(f"  预测类别 (对 T 平均后 argmax): {pred.tolist()}")
    check_golden(out)

    print("\n推理完成。")


if __name__ == '__main__':
    main()
