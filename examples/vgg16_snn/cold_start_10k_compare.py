"""冷启动 10000 样本对照: path B (VGG16SNN BN+MaxPool) vs NIR-compile (fold-BN+AvgPool)。

两条路径走完全等价的：
- dynamo / inductor 全 Triton 配置
- BATCH=56, T=4, 输入 [T, B, 3, 224, 224]
- WARMUP 5 + MEASURE 179  (179×56=10024 ≥ 10000)
- 测量都在 first forward (含编译) 之后、5 次 warmup 之后

通过 MODE 环境变量选择路径:
    MODE=B    ... 调 VGG16SNN（vgg16_test.py 同款，含 BN + MaxPool），需 patch_spikingjelly_for_full_graph()
    MODE=NIR  ... 调 NIR roundtrip（fold-BN + AvgPool），不需要 patch（无 BN 子模块）
    MODE=SJ   ... 直接用 SJ layer.* 搭一个和 NIR-imported gm 算子集等价的网络
                  （layer.Conv2d + neuron.LIFNode + layer.AvgPool2d + layer.Flatten + layer.Linear，
                  无 BN，作为"非 NIR 但等价"对照）
"""
import os, sys, time, json, pathlib

sys.path.insert(0, "/home/charlley/Code/Triton-Pass-Analysis/examples/vgg16_snn")

import torch
import torch.nn as nn
import torch._dynamo
import torch._inductor.config as inductor_cfg

MODE = os.environ.get("MODE", "NIR").upper()
BATCH = int(os.environ.get("BATCH", 56))
TOTAL_SAMPLES = int(os.environ.get("TOTAL_SAMPLES", 10000))
WARMUP = int(os.environ.get("WARMUP", 5))
MEASURE_ITERS = (TOTAL_SAMPLES + BATCH - 1) // BATCH

# === 全 Triton 编译配置（两侧完全一致）===
torch._dynamo.config.recompile_limit = 256
torch._dynamo.config.cache_size_limit = 256
inductor_cfg.max_autotune = True
inductor_cfg.max_autotune_gemm_backends = "TRITON"
inductor_cfg.max_autotune_conv_backends = "TRITON"
inductor_cfg.force_disable_caches = True

print(f"[config] MODE={MODE}  BATCH={BATCH}  MEASURE_ITERS={MEASURE_ITERS}  "
      f"WARMUP={WARMUP}  目标样本 = {MEASURE_ITERS * BATCH} (≥{TOTAL_SAMPLES})")

from spikingjelly.activation_based import functional, neuron, layer, nir_exchange
from spikingjelly.activation_based.functional.conv_bn_fusion import (
    fuse_conv_bn_eval_modules,
)

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

if MODE == "B":
    # path B: 与 vgg16_test.py 完全一致的网络（layer.* 多步包装 + BN + MaxPool）
    from vgg16_test import (
        VGG16SNN, NUM_CLASSES, T as T_CONST,
        patch_spikingjelly_for_full_graph,
    )
    patch_spikingjelly_for_full_graph()
    model = VGG16SNN(NUM_CLASSES)
    functional.set_step_mode(model, "m")
    model.eval().cuda()
    gm = model
    print(f"[build] path B: VGG16SNN (13 Conv + 13 BN + 15 LIF + 5 MaxPool + 3 FC), step_mode=m")

elif MODE == "NIR":
    # NIR path: build → fold BN → export/import via NIR
    VGG16_CFG = [64, 64, "P", 128, 128, "P", 256, 256, 256, "P",
                 512, 512, 512, "P", 512, 512, 512, "P"]
    feats, in_ch = [], 3
    for v in VGG16_CFG:
        if v == "P":
            feats.append(nn.AvgPool2d(2, 2))
        else:
            feats.extend([nn.Conv2d(in_ch, v, 3, padding=1), nn.BatchNorm2d(v),
                          neuron.LIFNode(step_mode="s")])
            in_ch = v
    raw = nn.Sequential(
        nn.Sequential(*feats),
        nn.Sequential(nn.Flatten(),
                      nn.Linear(512 * 7 * 7, 4096), neuron.LIFNode(step_mode="s"),
                      nn.Linear(4096, 4096), neuron.LIFNode(step_mode="s"),
                      nn.Linear(4096, 1000))).eval()
    folded = fuse_conv_bn_eval_modules(raw)
    graph = nir_exchange.export_to_nir(folded, example_input=torch.rand(1, 3, 224, 224), dt=1e-4)
    gm = nir_exchange.import_from_nir(graph, dt=1e-4, device="cuda", step_mode="m")
    gm.eval()
    print(f"[build] NIR path: 13 Conv (fold-BN) + 15 LIF + 5 AvgPool + 3 Linear, step_mode=m")

elif MODE == "SJ":
    # SJ-direct: 与 NIR-imported gm 算子集等价（无 BN, AvgPool, 全部用 SJ layer.* + neuron.LIFNode）
    # 跟 path B 的差别: 没有 BN, MaxPool 换 AvgPool
    # 跟 NIR 的差别: 不经过 export/import 翻译，直接用 SJ layer 构造（也就没有 nirtorch 自动生成的
    #                forward 里那些 ones/_operator_is_/ternary_operator dead helper）
    VGG16_CFG = [64, 64, "P", 128, 128, "P", 256, 256, 256, "P",
                 512, 512, 512, "P", 512, 512, 512, "P"]
    feats, in_ch = [], 3
    for v in VGG16_CFG:
        if v == "P":
            feats.append(layer.AvgPool2d(kernel_size=2, stride=2, step_mode="s"))
        else:
            feats.append(layer.Conv2d(in_ch, v, kernel_size=3, padding=1, step_mode="s"))
            feats.append(neuron.LIFNode(step_mode="s"))
            in_ch = v
    classifier = nn.Sequential(
        layer.Flatten(step_mode="s"),
        layer.Linear(512 * 7 * 7, 4096), neuron.LIFNode(step_mode="s"),
        layer.Linear(4096, 4096), neuron.LIFNode(step_mode="s"),
        layer.Linear(4096, 1000),
    )
    gm = nn.Sequential(nn.Sequential(*feats), classifier)
    functional.set_step_mode(gm, "m")
    gm.eval().cuda()
    print(f"[build] SJ-direct: 13 layer.Conv2d + 15 LIFNode + 5 layer.AvgPool2d + "
          f"layer.Flatten + 3 layer.Linear, step_mode=m  (与 NIR 算子集等价)")

else:
    raise ValueError(f"unknown MODE: {MODE!r}")

compiled = torch.compile(gm)

x = torch.randn(4, BATCH, 3, 224, 224, device="cuda")
print(f"[run] input shape: {tuple(x.shape)}")

# === 首次 forward = 编译 + autotune（冷启动开销）===
print(f"[cold-start] compiling + autotune (~50-120s, Triton cache should be empty) ...")
torch.cuda.synchronize()
t0 = time.perf_counter()
functional.reset_net(gm)
with torch.no_grad():
    out = compiled(x)
    if isinstance(out, tuple): out = out[0]
torch.cuda.synchronize()
compile_s = time.perf_counter() - t0
print(f"[cold-start] compile + first forward: {compile_s:.2f}s   out shape={tuple(out.shape)}")

torch.cuda.reset_peak_memory_stats()

# === Warmup ===
print(f"[run] warmup {WARMUP} iters ...")
for _ in range(WARMUP):
    functional.reset_net(gm)
    with torch.no_grad():
        out = compiled(x)
        if isinstance(out, tuple): out = out[0]
torch.cuda.synchronize()

# === 测量 ===
print(f"[run] measure {MEASURE_ITERS} iters × BATCH={BATCH} = {MEASURE_ITERS * BATCH} samples ...")
per_iter_ms = []
torch.cuda.synchronize()
t0 = time.perf_counter()
for i in range(MEASURE_ITERS):
    functional.reset_net(gm)
    ti = time.perf_counter()
    with torch.no_grad():
        out = compiled(x)
        if isinstance(out, tuple): out = out[0]
    torch.cuda.synchronize()
    per_iter_ms.append((time.perf_counter() - ti) * 1000)
torch.cuda.synchronize()
total_s = time.perf_counter() - t0

peak_mem = torch.cuda.max_memory_allocated() / 2**30
n_samples = MEASURE_ITERS * BATCH

# === stats ===
import statistics
per_img = [x / BATCH for x in per_iter_ms]
mean_iter = statistics.mean(per_iter_ms)
std_iter  = statistics.stdev(per_iter_ms) if len(per_iter_ms) > 1 else 0.0
median_iter = statistics.median(per_iter_ms)
mn = min(per_iter_ms); mx = max(per_iter_ms)
print()
print("=" * 78)
print(f"  MODE = {MODE}    BATCH = {BATCH}    cold-start = YES")
print(f"  forward 调用次数 = {MEASURE_ITERS}     总样本 = {n_samples}")
print(f"  总耗时           = {total_s:.4f} s")
print(f"  每次 forward     = avg {mean_iter:.4f} ms  | median {median_iter:.4f}  "
      f"| std {std_iter:.4f}  | min {mn:.4f}  | max {mx:.4f}")
print(f"  单张折算         = avg {mean_iter / BATCH:.5f} ms / 张  "
      f"| median {median_iter / BATCH:.5f}")
print(f"  吞吐             = {n_samples / total_s:.2f} 张/秒")
print(f"  GPU peak memory  = {peak_mem:.2f} GiB")
print(f"  cold compile     = {compile_s:.1f} s")
print("=" * 78)

# === 结果落盘到 /tmp/cold_start_results.jsonl，便于多次跑后聚合 ===
result = {
    "mode": MODE, "batch": BATCH, "iters": MEASURE_ITERS, "n_samples": n_samples,
    "total_s": total_s, "mean_iter_ms": mean_iter, "median_iter_ms": median_iter,
    "std_iter_ms": std_iter, "min_iter_ms": mn, "max_iter_ms": mx,
    "mean_per_img_ms": mean_iter / BATCH, "throughput_imgs_per_s": n_samples / total_s,
    "peak_mem_gib": peak_mem, "compile_s": compile_s,
}
with open("/tmp/cold_start_results.jsonl", "a") as f:
    f.write(json.dumps(result) + "\n")
print(f"[stat] result appended to /tmp/cold_start_results.jsonl")
