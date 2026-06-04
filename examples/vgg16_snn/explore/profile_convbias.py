"""Profile FusedIF-eager 看剩余瓶颈在哪。"""
import os, sys, pathlib, time, gc
HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "explore"))

import torch
from torch.profiler import profile, ProfilerActivity, record_function
from fused_bias_if_snn import build

BATCH = 32
T = 4

OUT = pathlib.Path("/home/charlley/Code/Triton-Pass-Analysis/Document/IR-Trace/exploration")
OUT.mkdir(parents=True, exist_ok=True)


def is_gpu_kernel_name(name):
    return (name.startswith("void ") or name.startswith("triton_") or
            name.startswith("_multistep_") or name.startswith("_fused_if") or
            name.startswith("Memcpy") or name.startswith("Memset"))


def categorize(k):
    if "_fused_if" in k:
        return "ConvBiasIF / FusedIF (hand)"
    if "cutlass_5x_cudnn" in k or "cutlass_tensorop" in k:
        return "conv (cuDNN cutlass)"
    if "sm80_xmma_fprop" in k or "implicit_convolve_sgemm" in k:
        return "conv (cuDNN xmma/sgemm)"
    if "winograd" in k:
        return "conv (cuDNN winograd)"
    if "nchwToNhwc" in k or "nhwcToNchw" in k:
        return "conv (cuDNN layout xform)"
    if "cutlass_80_simt_sgemm" in k or "gemmSN_" in k:
        return "gemm (cuBLAS)"
    if "avg_pool" in k and "at::native" in k:
        return "AvgPool (ATen native)"
    if "elementwise_kernel" in k or "vectorized_elementwise" in k:
        return "elementwise (ATen)"
    if "Memcpy" in k or "Memset" in k:
        return "memcpy/memset"
    return "other"


model = build()
x = torch.randn(T, BATCH, 3, 224, 224, device="cuda")

# warmup
for _ in range(3):
    with torch.no_grad():
        model(x)
torch.cuda.synchronize()

# profile
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function("forward_fused_eager"), torch.no_grad():
        for _ in range(3):
            model(x)
    torch.cuda.synchronize()


by_cat = {}
rows = []
for ev in prof.key_averages():
    name = ev.key
    if not is_gpu_kernel_name(name):
        continue
    self_us = getattr(ev, "self_device_time_total", 0)
    if self_us <= 0:
        continue
    self_us_per_iter = self_us / 3
    rows.append((name, self_us_per_iter, max(1, ev.count // 3)))
    cat = categorize(name)
    by_cat.setdefault(cat, {"us": 0.0, "calls": 0})
    by_cat[cat]["us"] += self_us_per_iter
    by_cat[cat]["calls"] += max(1, ev.count // 3)

rows.sort(key=lambda r: -r[1])

total = sum(v["us"] for v in by_cat.values())
print(f"\nFusedIF-eager kernel categories (BATCH={BATCH}, T={T}):")
print(f"{'category':<40s} {'us/iter':>12s} {'% of total':>12s} {'calls':>8s}")
print("-" * 76)
for cat in sorted(by_cat, key=lambda k: -by_cat[k]["us"]):
    v = by_cat[cat]
    print(f"{cat:<40s} {v['us']:>12.1f} {v['us']/total*100:>11.2f}% {v['calls']:>8d}")
print("-" * 76)
print(f"{'TOTAL':<40s} {total:>12.1f}")

print(f"\nTop 15 kernels by self time:")
print(f"{'rank':>4s}  {'self_us':>10s}  {'calls':>5s}  name")
for i, (n, su, c) in enumerate(rows[:15]):
    print(f"{i+1:>4d}  {su:>10.1f}  {c:>5d}  {n[:80]}")

# 同时落盘
with open(OUT / "convbias_eager_kernels.txt", "w") as f:
    f.write(f"# FusedIF-eager (BATCH={BATCH}, T={T}) kernel breakdown\n\n")
    f.write(f"{'category':<40s} {'us/iter':>12s} {'% of total':>12s} {'calls':>8s}\n")
    f.write("-" * 76 + "\n")
    for cat in sorted(by_cat, key=lambda k: -by_cat[k]["us"]):
        v = by_cat[cat]
        f.write(f"{cat:<40s} {v['us']:>12.1f} {v['us']/total*100:>11.2f}% {v['calls']:>8d}\n")
    f.write("-" * 76 + "\n")
    f.write(f"{'TOTAL':<40s} {total:>12.1f}\n\n")
    f.write(f"Top 25 kernels:\n{'rank':>4s}  {'self_us':>10s}  {'calls':>5s}  name\n")
    for i, (n, su, c) in enumerate(rows[:25]):
        f.write(f"{i+1:>4d}  {su:>10.1f}  {c:>5d}  {n}\n")
