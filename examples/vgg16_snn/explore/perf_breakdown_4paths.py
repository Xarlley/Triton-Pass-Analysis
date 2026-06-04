"""Step 0: 给 4 个 baseline 都抓 per-kernel self_cuda_time，并按算子类别聚合。

输出到 Document/IR-Trace/exploration/breakdown_4paths.txt
"""
import os
import sys
import time
import pathlib
import gc

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import torch
import torch.nn as nn
import torch._dynamo
import torch._inductor.config as inductor_cfg
from torch.profiler import profile, ProfilerActivity, record_function

from spikingjelly.activation_based import functional, neuron, layer

BATCH = int(os.environ.get("BATCH", 32))
T = 4
SEED = 42

OUT = pathlib.Path("/home/charlley/Code/Triton-Pass-Analysis/Document/IR-Trace/exploration")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda")


# ---------------- kernel 分类（覆盖 4 个 path 涉及的所有 kernel 名） ----------------
def categorize(name: str) -> str:
    k = name
    # SJ 手写 LIF kernel
    if "_multistep_lif" in k or "multistep_lif_forward" in k:
        return "LIF/IF kernel (SJ Triton)"
    # PrefixSumIF 路径下 Inductor 生成的 IF kernel
    if "triton_" in k and ("cumsum" in k.lower() or "scan" in k.lower()):
        return "cumsum (Inductor Triton scan)"
    if "tl_scan" in k or "scan_kernel" in k.lower():
        return "cumsum (ATen native scan)"
    # ATen native cumsum
    if "cumsum" in k.lower():
        return "cumsum (ATen native)"
    # 卷积（Inductor Triton 模板 vs cuDNN）
    if "triton_tem_fused_convolution" in k:
        return "conv (Inductor Triton tem)"
    if "triton_poi_fused__native_batch_norm" in k:
        return "BN+conv-epilogue (Inductor Triton poi)"
    if "triton_poi_fused_convolution_max_pool" in k:
        return "MaxPool+conv-epilogue (Inductor Triton poi)"
    if "triton_poi_fused_convolution_avg_pool" in k or "triton_poi_fused_avg_pool2d_convolution" in k:
        return "AvgPool+conv-epilogue (Inductor Triton poi)"
    if "triton_poi_fused_convolution" in k:
        return "conv-epilogue (Inductor Triton poi)"
    if "triton_poi_fused_avg_pool" in k:
        return "AvgPool (Inductor Triton poi)"
    if "triton_poi_fused_max_pool" in k:
        return "MaxPool (Inductor Triton poi)"
    if "triton_tem_fused_addmm" in k:
        return "gemm (Inductor Triton tem)"
    if "triton_poi_fused" in k or "triton_red_fused" in k or "triton_per_fused" in k:
        return "elementwise (Inductor Triton poi/red/per)"
    # cuDNN / cuBLAS / ATen native CUDA
    if "cutlass__5x_cudnn" in k or "cutlass_tensorop" in k:
        return "conv (cuDNN cutlass)"
    if "sm80_xmma_fprop" in k or "implicit_convolve_sgemm" in k:
        return "conv (cuDNN xmma/sgemm)"
    if "winograd" in k:
        return "conv (cuDNN winograd)"
    if "nchwToNhwc" in k or "nhwcToNchw" in k:
        return "conv (cuDNN layout xform)"
    if "cutlass_80_simt_sgemm" in k or "ampere_sgemm" in k or "cublas" in k.lower() or "gemmSN_" in k:
        return "gemm (cuBLAS)"
    if "bn_fw_inf" in k or "bn_bw_" in k or "batch_norm" in k.lower():
        return "BN (cuDNN)"
    if "max_pool" in k and "at::native" in k:
        return "MaxPool (ATen native CUDA)"
    if "avg_pool" in k and "at::native" in k:
        return "AvgPool (ATen native CUDA)"
    if "elementwise_kernel" in k or "vectorized_elementwise" in k or "FillFunctor" in k:
        return "elementwise (ATen native CUDA)"
    if "Memcpy" in k or "Memset" in k:
        return "memcpy/memset"
    return "other"


def is_gpu_kernel_name(name: str) -> bool:
    return (
        name.startswith("void ")
        or name.startswith("triton_")
        or name.startswith("_multistep_")
        or name.startswith("Memcpy")
        or name.startswith("Memset")
    )


# ---------------- 4 个 path 的模型构造 ----------------
def build_path_B():
    """SJ LIFNode (VGG16SNN, BN+MaxPool)"""
    from vgg16_test import VGG16SNN, NUM_CLASSES
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    m = VGG16SNN(NUM_CLASSES)
    functional.set_step_mode(m, "m")
    m.eval().cuda()
    return m


VGG16_CFG = [64, 64, "P", 128, 128, "P", 256, 256, 256, "P",
             512, 512, 512, "P", 512, 512, 512, "P"]


class TimeBatchWrapper(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    def forward(self, x):
        T, B = x.shape[0], x.shape[1]
        y = self.layer(x.flatten(0, 1))
        return y.view(T, B, *y.shape[1:])


class PrefixSumHardResetIFNode(nn.Module):
    def __init__(self, v_threshold=1.0):
        super().__init__()
        self.v_threshold = v_threshold
    def forward(self, x):
        cum = torch.cumsum(x, dim=0)
        last_cum_at_spike = torch.zeros_like(x[0])
        spikes = []
        for t in range(x.shape[0]):
            v_t = cum[t] - last_cum_at_spike
            spike_t = (v_t >= self.v_threshold).to(x.dtype)
            spikes.append(spike_t)
            last_cum_at_spike = torch.where(spike_t > 0, cum[t], last_cum_at_spike)
        return torch.stack(spikes, dim=0)


def build_prefix_sum():
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    feats, in_ch = [], 3
    for v in VGG16_CFG:
        if v == "P":
            feats.append(TimeBatchWrapper(nn.AvgPool2d(2, 2)))
        else:
            feats.append(TimeBatchWrapper(nn.Conv2d(in_ch, v, 3, padding=1)))
            feats.append(PrefixSumHardResetIFNode())
            in_ch = v
    cls = nn.Sequential(
        TimeBatchWrapper(nn.Flatten()),
        TimeBatchWrapper(nn.Linear(512*7*7, 4096)), PrefixSumHardResetIFNode(),
        TimeBatchWrapper(nn.Linear(4096, 4096)),    PrefixSumHardResetIFNode(),
        TimeBatchWrapper(nn.Linear(4096, 1000)),
    )
    m = nn.Sequential(nn.Sequential(*feats), cls)
    return m.eval().cuda()


def configure_full_triton():
    from vgg16_test import patch_spikingjelly_for_full_graph
    torch._dynamo.config.recompile_limit = 256
    torch._dynamo.config.cache_size_limit = 256
    inductor_cfg.max_autotune = True
    inductor_cfg.max_autotune_gemm_backends = "TRITON"
    inductor_cfg.max_autotune_conv_backends = "TRITON"
    inductor_cfg.force_disable_caches = True
    patch_spikingjelly_for_full_graph()


# ---------------- 抓 kernel 表 ----------------
def to_kernel_rows(prof, iters):
    seen = {}
    for ev in prof.key_averages():
        name = ev.key
        if not is_gpu_kernel_name(name):
            continue
        self_us = getattr(ev, "self_device_time_total", 0)
        total_us = getattr(ev, "device_time_total", 0)
        if self_us <= 0 and total_us <= 0:
            continue
        if name in seen:
            continue
        # 平摊到单 iter
        seen[name] = (name, self_us / iters, max(1, ev.count // iters), total_us / iters)
    return sorted(seen.values(), key=lambda r: -r[1])


def profile_path(label, callable_fn, x, model_for_reset, warmup=3, iters=3):
    # reset_net only applies for SJ-LIFNode path; PrefixSum is stateless
    has_reset = isinstance(model_for_reset, nn.Module) and any(
        isinstance(m, neuron.LIFNode) for m in model_for_reset.modules()
    )
    def reset():
        if has_reset:
            functional.reset_net(model_for_reset)

    for _ in range(warmup):
        reset()
        with torch.no_grad():
            out = callable_fn(x)
            if isinstance(out, tuple): out = out[0]
        torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function(f"forward_{label}"), torch.no_grad():
            for _ in range(iters):
                reset()
                out = callable_fn(x)
                if isinstance(out, tuple): out = out[0]
        torch.cuda.synchronize()
    return to_kernel_rows(prof, iters)


def write_kernel_table(rows, path, title):
    with open(path, "w") as f:
        f.write(f"# {title}\n# rank | self_us | calls | total_us | name\n\n")
        f.write(f"{'rank':>4}  {'self_us':>10}  {'calls':>5}  {'total_us':>10}  name\n")
        f.write("-" * 100 + "\n")
        for i, (name, su, n, tu) in enumerate(rows):
            f.write(f"{i+1:>4}  {su:>10.1f}  {n:>5}  {tu:>10.1f}  {name}\n")


def aggregate(rows):
    by = {}
    for name, su, n, tu in rows:
        cat = categorize(name)
        if cat not in by:
            by[cat] = {"us": 0.0, "calls": 0, "n_kernels": 0}
        by[cat]["us"] += su
        by[cat]["calls"] += n
        by[cat]["n_kernels"] += 1
    return by


def main():
    x = torch.randn(T, BATCH, 3, 224, 224, device=DEVICE)
    print(f"BATCH={BATCH}  T={T}\n")

    results = {}

    # --- Path 1: SJ-eager ---
    print("=" * 70); print("Path 1: SJ-LIFNode + eager")
    m = build_path_B()
    rows = profile_path("sj_eager", m, x, m)
    write_kernel_table(rows, OUT / "kernels_sj_eager.txt", "SJ-LIFNode + eager")
    results["1. SJ-eager"] = aggregate(rows)
    del m; gc.collect(); torch.cuda.empty_cache()

    # --- Path 2: SJ-compile ---
    print("=" * 70); print("Path 2: SJ-LIFNode + torch.compile (full Triton)")
    configure_full_triton()
    m = build_path_B()
    compiled = torch.compile(m)
    print("  cold compile...")
    t0 = time.perf_counter()
    functional.reset_net(m)
    with torch.no_grad(): compiled(x)
    torch.cuda.synchronize()
    print(f"  cold {time.perf_counter()-t0:.1f}s")
    rows = profile_path("sj_compile", compiled, x, m)
    write_kernel_table(rows, OUT / "kernels_sj_compile.txt", "SJ-LIFNode + compile")
    results["2. SJ-compile"] = aggregate(rows)
    del m, compiled
    import torch._dynamo as _dynamo; _dynamo.reset()
    gc.collect(); torch.cuda.empty_cache()

    # --- Path 3: PrefixSum-eager ---
    print("=" * 70); print("Path 3: PrefixSumIF + eager")
    m = build_prefix_sum()
    rows = profile_path("ps_eager", m, x, m)
    write_kernel_table(rows, OUT / "kernels_ps_eager.txt", "PrefixSumIF + eager")
    results["3. PrefixSum-eager"] = aggregate(rows)
    del m; gc.collect(); torch.cuda.empty_cache()

    # --- Path 4: PrefixSum-compile ---
    print("=" * 70); print("Path 4: PrefixSumIF + torch.compile (full Triton)")
    m = build_prefix_sum()
    compiled = torch.compile(m)
    print("  cold compile...")
    t0 = time.perf_counter()
    with torch.no_grad(): compiled(x)
    torch.cuda.synchronize()
    print(f"  cold {time.perf_counter()-t0:.1f}s")
    rows = profile_path("ps_compile", compiled, x, m)
    write_kernel_table(rows, OUT / "kernels_ps_compile.txt", "PrefixSumIF + compile")
    results["4. PrefixSum-compile"] = aggregate(rows)

    # --- 类别对照表 ---
    all_cats = sorted(set().union(*[r.keys() for r in results.values()]))
    print("\n\n========== 4-path kernel category breakdown (us / forward, BATCH=", BATCH, ") ==========")
    headers = ["category"] + list(results.keys())
    col_w = max(len(c) for c in all_cats) + 2
    print(f"{'category':<{col_w}}", " ".join(f"{h:>22s}" for h in headers[1:]))
    print("-" * (col_w + 23 * 4))
    table_str = []
    for cat in all_cats:
        row_vals = []
        for path_name in results:
            v = results[path_name].get(cat, {"us": 0})["us"]
            row_vals.append(v)
        line = f"{cat:<{col_w}}" + " ".join(f"{v:>22.1f}" for v in row_vals)
        print(line); table_str.append(line)

    totals = {p: sum(c["us"] for c in r.values()) for p, r in results.items()}
    total_line = f"{'TOTAL':<{col_w}}" + " ".join(f"{totals[p]:>22.1f}" for p in results)
    print("-" * (col_w + 23 * 4))
    print(total_line); table_str.append(total_line)

    with open(OUT / "breakdown_4paths.txt", "w") as f:
        f.write(f"# 4-path kernel category breakdown (us / forward, BATCH={BATCH}, T={T})\n\n")
        f.write(f"{'category':<{col_w}}" + " ".join(f"{h:>22s}" for h in headers[1:]) + "\n")
        f.write("-" * (col_w + 23 * 4) + "\n")
        for line in table_str:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
