"""vgg16_via_nir.py — 用 NIR 搭一个与 vgg16_test.py 结构对齐的 VGG16-SNN，并测推理延迟。

与 vgg16_test.py 的差异 / NIR 协议带来的妥协：
  1. 池化 MaxPool → AvgPool：NIR 协议没有 MaxPool 原语，只有 AvgPool2d。
  2. BatchNorm：NIR 协议没有 BN 原语，因此在 eval 模式下用 SJ 自带的
     `fuse_conv_bn_eval_modules` 把 Conv→BN 数学等价地折成单个带 bias 的 Conv，
     从而消除 BN 节点再做 NIR 导出。
  3. 神经元：vgg16_test.py 的 LIFNode 默认 v_reset=0.0（硬复位），可直接映射到
     nir.LIF（NIR 不区分 soft/hard reset，本网络默认就是硬复位，无信息损失）。

流程：
  build SJ VGG16-SNN (BN + AvgPool, step_mode='s', eval) →
  fold-BN →
  export_to_nir →
  import_from_nir(device='cuda', step_mode='m') →
  在 [T, B, C, H, W] = (4, 1, 3, 224, 224) 上测推理延迟。

随机权重 + 随机输入，不关心数值正确性。
"""
import time

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, layer, neuron, nir_exchange
from spikingjelly.activation_based.functional.conv_bn_fusion import (
    fuse_conv_bn_eval_modules,
)


SEED = 42
T = 4
NUM_CLASSES = 1000
DEVICE = "cuda"
WARMUP = 3
ITERS = 10

# 与 vgg16_test.py 完全一致的 13-Conv 配置，'P' 代替 'M'：原本是 MaxPool(2,2)
# 这里替换为 AvgPool(2,2)，是 NIR 协议下唯一可表达的池化。
VGG16_CFG = [64, 64, "P",
             128, 128, "P",
             256, 256, 256, "P",
             512, 512, 512, "P",
             512, 512, 512, "P"]


def build_vgg16_snn(num_classes: int = NUM_CLASSES) -> nn.Module:
    """SJ 版 VGG16-SNN（BN + AvgPool）。单步模式便于 fold-BN 与 NIR 导出。

    结构与 vgg16_test.py 相同：13 Conv + 13 BN + 13 LIF + 5 AvgPool（特征提取）
    + Flatten + Linear-LIF-Linear-LIF-Linear（分类器，末层为线性读出）。

    说明：所有无状态层用原生 ``nn.*``（而非 ``spikingjelly.layer.*``）。
    原因：``fuse_conv_bn_eval_modules`` 内的 ``_EvalFusionTracer`` 会穿透
    SJ 的 layer wrapper、把 ``layer.AvgPool2d.forward`` 内联成
    ``torch._C._nn.avg_pool2d`` 这样的 ``call_function``；而 nirtorch 的
    tracer 只接受 ``call_module``（额外仅允许 ``operator.add``），遇到内联
    函数会抛 ``ValueError: The only supported function is addition``。
    原生 ``nn.*`` 在 fx 里默认是 leaf module，能避开这条规则。
    多步前向不靠 SJ 的 layer wrapper 而是靠 ``import_from_nir(step_mode='m')``
    返回的 fx.GraphModule，它内部用的就是 SJ ``layer.*``。
    """
    feats, in_ch = [], 3
    for v in VGG16_CFG:
        if v == "P":
            feats.append(nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            feats.append(nn.Conv2d(in_ch, v, kernel_size=3, padding=1))
            feats.append(nn.BatchNorm2d(v))
            feats.append(neuron.LIFNode(step_mode="s"))   # 默认 v_reset=0.0
            in_ch = v
    features = nn.Sequential(*feats)
    classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512 * 7 * 7, 4096),
        neuron.LIFNode(step_mode="s"),
        nn.Linear(4096, 4096),
        neuron.LIFNode(step_mode="s"),
        nn.Linear(4096, num_classes),
    )
    return nn.Sequential(features, classifier)


def count_modules(model: nn.Module) -> dict:
    return {
        "Conv2d": sum(1 for m in model.modules() if isinstance(m, (nn.Conv2d, layer.Conv2d))),
        "BatchNorm2d": sum(1 for m in model.modules() if isinstance(m, (nn.BatchNorm2d, layer.BatchNorm2d))),
        "AvgPool2d": sum(1 for m in model.modules() if isinstance(m, (nn.AvgPool2d, layer.AvgPool2d))),
        "Linear": sum(1 for m in model.modules() if isinstance(m, (nn.Linear, layer.Linear))),
        "LIFNode": sum(1 for m in model.modules() if isinstance(m, neuron.LIFNode)),
    }


def measure(model_callable, x, label: str, reset_fn=None):
    """WARMUP 次预热 + ITERS 次计时，打印平均/最小/最大延迟（ms）。"""
    for _ in range(WARMUP):
        if reset_fn is not None:
            reset_fn()
        out = model_callable(x)
        if isinstance(out, tuple):
            out = out[0]
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(ITERS):
        if reset_fn is not None:
            reset_fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model_callable(x)
        if isinstance(out, tuple):
            out = out[0]
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    avg = sum(times_ms) / len(times_ms)
    print(
        f"  {label}\n"
        f"    avg = {avg:7.2f} ms | min = {min(times_ms):7.2f} | max = {max(times_ms):7.2f}  "
        f"(over {ITERS} iters)"
    )
    return avg


def main():
    print(f"设备: {DEVICE} | T={T} | warmup={WARMUP} | iters={ITERS}")
    print(f"输入形状: [T, B, C, H, W] = ({T}, 1, 3, 224, 224)")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print("\n[1/4] 构建 SJ VGG16-SNN（BN + AvgPool, single-step, eval）...")
    model = build_vgg16_snn().eval()
    info = count_modules(model)
    print(
        f"  原始结构: Conv={info['Conv2d']} BN={info['BatchNorm2d']} "
        f"AvgPool={info['AvgPool2d']} Linear={info['Linear']} LIF={info['LIFNode']}"
    )
    n_param = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_param:,} ({n_param / 1e6:.1f}M)")

    print("\n[2/4] fold BN 进 Conv（NIR 协议无 BN 原语，必须先消除）...")
    folded = fuse_conv_bn_eval_modules(model)
    info_after = count_modules(folded)
    print(
        f"  fold 后:  Conv={info_after['Conv2d']} BN={info_after['BatchNorm2d']} "
        f"AvgPool={info_after['AvgPool2d']} Linear={info_after['Linear']} "
        f"LIF={info_after['LIFNode']}"
    )
    assert info_after["BatchNorm2d"] == 0, "fold 失败：仍有 BN 残留"

    print("\n[3/4] export_to_nir → import_from_nir(device=cuda, step_mode=m)...")
    example_input = torch.rand(1, 3, 224, 224)        # 单步、单 batch
    graph = nir_exchange.export_to_nir(folded, example_input=example_input, dt=1e-4)
    print(f"  NIR 图节点数: {len(graph.nodes)} | 边数: {len(graph.edges)}")
    gm = nir_exchange.import_from_nir(graph, dt=1e-4, device=DEVICE, step_mode="m")
    # 强制 LIF 用 torch 后端，避开 SJ 自带 multi-step Triton kernel 与本项目
    # triton fork 的不兼容（_multistep_lif_forward_kernel 在 convert_and_store
    # 处报 CompilationError）。
    functional.set_backend(gm, "torch", instance=neuron.LIFNode)
    nir_param = sum(p.numel() for p in gm.parameters())
    print(f"  导入回 SJ 后参数量: {nir_param:,} ({nir_param / 1e6:.1f}M)")

    print("\n[4/4] 推理延迟测量...")
    x = torch.randn(T, 1, 3, 224, 224, device=DEVICE)

    # NIR-imported model: forward 返回 (out, state)，取 [0]。
    def nir_forward(inp):
        return gm(inp)

    print("  ----------------- NIR 版（NIR 导出→导入后） -----------------")
    measure(
        nir_forward, x,
        label="NIR-imported VGG16-SNN (T=4, B=1, 224x224)",
        reset_fn=lambda: functional.reset_net(gm),
    )

    print("\n完成。")


if __name__ == "__main__":
    main()
