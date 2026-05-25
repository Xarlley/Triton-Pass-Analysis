"""benchmark_compare.py — 同条件下对比两种 VGG16-SNN 的推理延迟。

A) 原始模型：vgg16_test.py 的 VGG16SNN（BN + MaxPool + LIFNode），
   设置 `configure_full_triton_compilation` + `patch_spikingjelly_for_full_graph` 后
   走 `torch.compile`，整网经由用户本地 triton fork 编译（max_autotune，conv/gemm 后端
   限定 TRITON）。

B) NIR 版：在 SJ 中先构 BN+AvgPool 网络（NIR 协议没有 MaxPool 与 BN，BN 用
   `fuse_conv_bn_eval_modules` 折叠到 Conv），`export_to_nir` → `import_from_nir`
   返回 multi-step fx.GraphModule，eager 推理（不走 torch.compile）。

输入：(T=4, B=1, 3, 224, 224)，从 vgg16_test.py 保存的 `vgg16_snn_input.pth` 加载，
两侧共享同一份输入。各跑 WARMUP=5 次预热（A 的首次预热含编译+autotune，单独计时不算入）
+ MEASURE=100 次计时，报告平均延迟。

注意：本脚本只关心延迟，不关心 A、B 输出一致性 —— B 已经用 AvgPool 替换 MaxPool 并
fold-BN，数学语义已不同。

用法：
    python benchmark_compare.py
"""
import gc
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 从 vgg16_test.py 复用：模型类、Triton 编译配置、SJ patch、文件路径
from vgg16_test import (  # noqa: E402
    INPUT_PATH,
    NUM_CLASSES,
    T,
    VGG16SNN,
    WEIGHTS_PATH,
    configure_full_triton_compilation,
    patch_spikingjelly_for_full_graph,
)

import torch.nn as nn  # noqa: E402
from spikingjelly.activation_based import functional, neuron, nir_exchange  # noqa: E402
from spikingjelly.activation_based.functional.conv_bn_fusion import (  # noqa: E402
    fuse_conv_bn_eval_modules,
)

SEED = 42
DEVICE = torch.device("cuda")
WARMUP = 5
MEASURE = 100
BATCH = int(os.environ.get("BATCH", 1))   # 1 = 延迟模式（默认）；>1 = 吞吐对照
MODE = os.environ.get("MODE", "both").upper()   # "A" / "B" / "both"。BATCH 较大时
                                                 # 必须分两次进程跑，避免 A 的 138M
                                                 # 参数 + activations 与 B 的 LIF
                                                 # autotune clone 同时存活撑爆显存。


def measure(callable_fn, x, label: str, reset_fn=None) -> float:
    """跑 WARMUP 次预热 + MEASURE 次计时。返回每次 forward 调用的平均延迟（ms）。
    同时打印「单张折算延迟」= avg / BATCH，便于跨 BATCH 比对。
    """
    for _ in range(WARMUP):
        if reset_fn is not None:
            reset_fn()
        out = callable_fn(x)
        if isinstance(out, tuple):
            out = out[0]
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(MEASURE):
        if reset_fn is not None:
            reset_fn()
        out = callable_fn(x)
        if isinstance(out, tuple):
            out = out[0]
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000.0
    avg_call = total_ms / MEASURE
    per_image = avg_call / BATCH
    n_samples = MEASURE * BATCH
    print(
        f"  {label}\n"
        f"    每次 forward avg = {avg_call:8.3f} ms  | 单张折算 = {per_image:7.3f} ms  "
        f"| 共 {n_samples} 张样本 ({MEASURE} iters × BATCH={BATCH})"
    )
    return avg_call


# --------------------- A) 原始 VGG16-SNN + 全 Triton ---------------------
def bench_original(x: torch.Tensor) -> float:
    print("\n=== A) 原始 vgg16_test.py 模型 (BN + MaxPool, torch.compile + 全 Triton) ===")
    # 配置全 Triton 路径（max_autotune、conv/gemm 限 TRITON、关 Inductor cache）。
    configure_full_triton_compilation()
    # 替换 SJ.seq_to_ann_forward 为 dynamo 友好版（消除图中断）。
    patch_spikingjelly_for_full_graph()

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = VGG16SNN(NUM_CLASSES)
    if os.path.exists(WEIGHTS_PATH):
        print(f"  加载已保存权重: {os.path.basename(WEIGHTS_PATH)}")
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    else:
        print("  未发现已保存权重，使用 seed 生成的随机权重")
    functional.set_step_mode(model, "m")
    model.eval()
    model = model.to(DEVICE)
    compiled = torch.compile(model)

    @torch.no_grad()
    def fwd(inp):
        return compiled(inp)

    print("  首次调用：torch.compile 编译 + max_autotune（可能数分钟）...")
    tc0 = time.perf_counter()
    functional.reset_net(model)
    fwd(x)
    torch.cuda.synchronize()
    print(f"  编译 + autotune + 首次前向耗时: {time.perf_counter() - tc0:.1f} s")

    return measure(
        fwd, x,
        label="原始 VGG16-SNN (compiled, full Triton)",
        reset_fn=lambda: functional.reset_net(model),
    )


# --------------------- B) NIR 版 ---------------------
def build_nir_vgg16_snn(num_classes: int):
    """与 vgg16_via_nir.py 中相同：原生 nn.* + neuron.LIFNode，单步、eval。"""
    cfg = [64, 64, "P", 128, 128, "P", 256, 256, 256, "P",
           512, 512, 512, "P", 512, 512, 512, "P"]
    feats, in_ch = [], 3
    for v in cfg:
        if v == "P":
            feats.append(nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            feats.append(nn.Conv2d(in_ch, v, kernel_size=3, padding=1))
            feats.append(nn.BatchNorm2d(v))
            feats.append(neuron.LIFNode(step_mode="s"))
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


def bench_nir(x: torch.Tensor) -> float:
    print("\n=== B) NIR 版 (BN-folded + AvgPool, eager) ===")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = build_nir_vgg16_snn(NUM_CLASSES).eval()
    folded = fuse_conv_bn_eval_modules(model)
    example_input = torch.rand(1, 3, 224, 224)
    graph = nir_exchange.export_to_nir(folded, example_input=example_input, dt=1e-4)
    print(f"  NIR 图节点数: {len(graph.nodes)} | 边数: {len(graph.edges)}")
    gm = nir_exchange.import_from_nir(graph, dt=1e-4, device=DEVICE, step_mode="m")
    gm.eval()   # 导入回来的 fx.GraphModule 默认 train 模式

    @torch.no_grad()
    def fwd(inp):
        return gm(inp)

    return measure(
        fwd, x,
        label="NIR-imported VGG16-SNN (eager)",
        reset_fn=lambda: functional.reset_net(gm),
    )


def main():
    print(f"设备: {DEVICE} | T={T} | BATCH={BATCH} | warmup={WARMUP} | measure={MEASURE}")

    # 输入：BATCH=1 时复用 vgg16_test.py 的黄金输入以保留延迟模式可复现；
    # 其余 BATCH 走 seed 生成的随机张量。
    if BATCH == 1 and os.path.exists(INPUT_PATH):
        print(f"加载共享输入 (BATCH=1): {os.path.basename(INPUT_PATH)}")
        x_cpu = torch.load(INPUT_PATH, map_location="cpu")
    else:
        print(f"用 seed={SEED} 生成随机输入 (BATCH={BATCH})")
        g = torch.Generator().manual_seed(SEED)
        x_cpu = torch.randn(T, BATCH, 3, 224, 224, generator=g)
    x = x_cpu.to(DEVICE)
    print(f"输入形状: {tuple(x.shape)}")

    if MODE in ("A", "BOTH"):
        avg_a = bench_original(x)
    else:
        avg_a = None

    if MODE == "BOTH":
        # 同进程内继续跑 B：清 dynamo/inductor 缓存 + GC + empty_cache 把 A 占的显存
        # 尽量还回去。BATCH 较大（≳32）时仍可能 OOM —— 那时请改成 MODE=A / MODE=B 分别
        # 跑两次进程，详见 print 提示。
        import torch._dynamo as _dynamo
        _dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_b, total_b = torch.cuda.mem_get_info()
        print(f"\n  [显存] A 释放后: free={free_b / 2**30:.2f} GiB / total={total_b / 2**30:.2f} GiB")

    if MODE in ("B", "BOTH"):
        avg_b = bench_nir(x)
    else:
        avg_b = None

    print("\n" + "=" * 78)
    print(f"  BATCH = {BATCH}    measure = {MEASURE} iters")
    if avg_a is not None:
        print(f"  A) 原始 VGG16-SNN  (BN+MaxPool, torch.compile+Triton):"
              f"  {avg_a:8.3f} ms / forward  |  {avg_a / BATCH:7.3f} ms / 张")
    if avg_b is not None:
        print(f"  B) NIR 版          (BN-folded+AvgPool, eager)         :"
              f"  {avg_b:8.3f} ms / forward  |  {avg_b / BATCH:7.3f} ms / 张")
    if avg_a is not None and avg_b is not None:
        print(f"  B / A = {avg_b / avg_a:.2f}x  (单张同比)")
    print("=" * 78)


if __name__ == "__main__":
    main()
