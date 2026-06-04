"""跨架构 benchmark：VGG / ResNet / MobileNet-V2 SNN 在朴素 vs 融合两种模式下的延迟对比。

每个架构跑 (fused=False vs fused=True)，BATCH/T/DTYPE/LAYOUT 可由环境变量控制。
"""
import os, sys, time, statistics, json, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
import torch.nn as nn

from snn_compiler.zoo import (
    vgg11_snn, vgg16_snn, vgg19_snn,
    resnet18_snn, resnet34_snn,
    mobilenet_v2_snn,
)


BATCH    = int(os.environ.get("BATCH", 16))
T        = int(os.environ.get("T", 4))
TOTAL    = int(os.environ.get("TOTAL", 200))
WARMUP   = int(os.environ.get("WARMUP", 3))
DTYPE    = {"fp32": torch.float32, "bf16": torch.bfloat16}[os.environ.get("MODE", "fp32")]
LAYOUT   = os.environ.get("LAYOUT", "NCHW")
NEURON   = os.environ.get("NEURON", "lif")
RESET    = os.environ.get("RESET", "hard")
INPUT_H  = int(os.environ.get("INPUT_H", 224))
ITERS    = max(1, (TOTAL + BATCH - 1) // BATCH)


def measure(model, x, name):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(WARMUP):
        with torch.no_grad():
            model(x)
    torch.cuda.synchronize()
    per_iter = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x)
        torch.cuda.synchronize()
        per_iter.append((time.perf_counter() - t0) * 1000)
    mean = statistics.mean(per_iter)
    per_img = mean / BATCH
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"  {name:<32s}  iter {mean:8.3f} ms  per-img {per_img:7.4f} ms  "
          f"peak {peak:.2f} GiB")
    return dict(name=name, mean_ms=mean, per_img_ms=per_img, peak_gib=peak)


def run_one(factory, name, *, num_classes=1000):
    soft = (RESET == "soft")
    kw = dict(num_classes=num_classes, neuron=NEURON,
              soft_reset=soft, v_threshold=1.0, v_reset=0.0,
              layout=LAYOUT)
    torch.manual_seed(0)
    m_naive = factory(fused=False, init_bn=True, **kw).cuda().eval()
    torch.manual_seed(0)
    m_fused = factory(fused=True, init_bn=True, **kw).cuda().eval()
    if DTYPE == torch.bfloat16:
        m_naive = m_naive.to(DTYPE)
        m_fused = m_fused.to(DTYPE)
    if LAYOUT == "NHWC":
        for m in [m_naive, m_fused]:
            for mod in m.modules():
                if isinstance(mod, nn.Conv2d):
                    mod.weight.data = mod.weight.data.to(memory_format=torch.channels_last)
    x = torch.randn(T, BATCH, 3, INPUT_H, INPUT_H, device="cuda", dtype=DTYPE)
    print(f"\n== {name}  (BATCH={BATCH}, T={T}, H={INPUT_H}, dtype={DTYPE}, layout={LAYOUT}) ==")
    rn = measure(m_naive, x, f"{name} naive")
    rf = measure(m_fused, x, f"{name} fused")
    speedup = rn["mean_ms"] / rf["mean_ms"]
    print(f"  → speedup {speedup:.3f}×")
    return rn, rf, speedup


def main():
    results = []
    # VGG-11 用 224；ResNet/MobileNet 在 224 上比较有意义；这里默认 224
    cases = [
        (vgg11_snn,        "VGG-11 SNN"),
        (vgg16_snn,        "VGG-16 SNN"),
        (resnet18_snn,     "ResNet-18 SNN"),
        (resnet34_snn,     "ResNet-34 SNN"),
        (mobilenet_v2_snn, "MobileNet-V2 SNN"),
    ]
    for factory, name in cases:
        try:
            rn, rf, sp = run_one(factory, name)
            results.append((name, rn, rf, sp))
        except torch.AcceleratorError as e:
            print(f"  [SKIP] {name}: {e}")
        except RuntimeError as e:
            print(f"  [SKIP] {name}: {e}")

    print("\n" + "=" * 78)
    print(f"{'Architecture':<22s} {'Naive (ms/img)':<16s} {'Fused (ms/img)':<16s} {'Speedup':<10s}")
    print("=" * 78)
    for name, rn, rf, sp in results:
        print(f"{name:<22s} {rn['per_img_ms']:<16.4f} {rf['per_img_ms']:<16.4f} {sp:<10.3f}×")
    print("=" * 78)

    out = pathlib.Path(__file__).resolve().parent / "zoo_bench_results.jsonl"
    with open(out, "a") as f:
        for name, rn, rf, sp in results:
            f.write(json.dumps({
                "arch": name, "config": {
                    "batch": BATCH, "T": T, "dtype": str(DTYPE),
                    "layout": LAYOUT, "neuron": NEURON, "reset": RESET,
                    "input_h": INPUT_H,
                },
                "naive_per_img_ms": rn["per_img_ms"],
                "fused_per_img_ms": rf["per_img_ms"],
                "naive_peak_gib": rn["peak_gib"],
                "fused_peak_gib": rf["peak_gib"],
                "speedup": sp,
            }) + "\n")
    print(f"[results appended to {out}]")


if __name__ == "__main__":
    main()
