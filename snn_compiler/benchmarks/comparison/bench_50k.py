"""50,000-sample 推理 benchmark：snn_compiler vs SpikingJelly。

测三种后端：
- ``ours``    : snn_compiler 的 zoo `fused=True`
- ``sj_eager``: SJ multi-step LIF + triton 后端
- ``sj_compile``: SJ single-step LIF + torch.compile + max_autotune

测三种网络：
- ``vgg16``   : VGG-16 SNN
- ``resnet18``: ResNet-18 SNN
- ``resnet34``: ResNet-34 SNN

环境变量
--------
- ARCH:    vgg16 | resnet18 | resnet34
- BACKEND: ours | sj_eager | sj_compile
- BATCH:   batch size（默认 32）
- T:       time steps（默认 4）
- TOTAL:   样本数（默认 50000）
- MODE:    fp32 | bf16（默认 bf16，仅 ours 支持 NHWC 加速）
- LAYOUT:  NCHW | NHWC（仅 ours 用 NHWC；sj 路径强制 NCHW）

输出 jsonl 到 ``Document/Benchmark/results/`` 下。
"""
from __future__ import annotations

import os, sys, time, statistics, json, pathlib

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "spikingjelly"))

import torch
import torch.nn as nn


# ---------- Configuration ----------
ARCH    = os.environ.get("ARCH", "vgg16")
BACKEND = os.environ.get("BACKEND", "ours")
BATCH   = int(os.environ.get("BATCH", 32))
T       = int(os.environ.get("T", 4))
TOTAL   = int(os.environ.get("TOTAL", 50000))
WARMUP  = int(os.environ.get("WARMUP", 5))
DTYPE   = {"fp32": torch.float32, "bf16": torch.bfloat16}[os.environ.get("MODE", "bf16")]
LAYOUT  = os.environ.get("LAYOUT", "NHWC")
NEURON  = "lif"
SOFT_RESET = False        # hard reset 全统一
INPUT_H = 224
NUM_CLASSES = 1000
TAG     = os.environ.get("TAG", "")
ITERS   = max(1, (TOTAL + BATCH - 1) // BATCH)
RESULTS_DIR = ROOT / "Document" / "Benchmark" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_ours():
    from snn_compiler.zoo import vgg16_snn, resnet18_snn, resnet34_snn
    factory = {"vgg16": vgg16_snn, "resnet18": resnet18_snn,
               "resnet34": resnet34_snn}[ARCH]
    m = factory(num_classes=NUM_CLASSES, neuron=NEURON, tau=2.0,
                 decay_input=True, soft_reset=SOFT_RESET,
                 v_threshold=1.0, v_reset=0.0,
                 layout=LAYOUT, fused=True, init_bn=True)
    return m


def build_sj(mode):
    """mode: 'eager' | 'compile'。"""
    from snn_compiler.benchmarks.comparison.sj_models import (
        sj_vgg16, sj_resnet18, sj_resnet34,
        init_bn_running_stats, reset_sj_state,
    )
    factory = {"vgg16": sj_vgg16, "resnet18": sj_resnet18,
                "resnet34": sj_resnet34}[ARCH]
    m = factory(num_classes=NUM_CLASSES, tau=2.0, v_threshold=1.0,
                 v_reset=0.0, decay_input=True, mode=mode)
    init_bn_running_stats(m)
    return m


def measure(model, x_seq, *, is_sj):
    """跑 ITERS 次，每次 50000/BATCH 张样本；SJ 模式每 iter 前 reset_net。"""
    if is_sj:
        from spikingjelly.activation_based import functional
        def reset_fn(): functional.reset_net(model)
    else:
        def reset_fn(): pass

    # warmup
    for _ in range(WARMUP):
        reset_fn()
        with torch.no_grad():
            model(x_seq)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # measure
    t_start = time.perf_counter()
    per_iter_ms = []
    for _ in range(ITERS):
        reset_fn()
        ti = time.perf_counter()
        with torch.no_grad():
            model(x_seq)
        torch.cuda.synchronize()
        per_iter_ms.append((time.perf_counter() - ti) * 1000)
    total_s = time.perf_counter() - t_start
    n_samples = ITERS * BATCH
    return dict(
        iters=ITERS, n_samples=n_samples,
        total_s=total_s,
        per_iter_mean_ms=statistics.mean(per_iter_ms),
        per_iter_median_ms=statistics.median(per_iter_ms),
        per_iter_std_ms=statistics.stdev(per_iter_ms) if ITERS > 1 else 0.0,
        per_iter_min_ms=min(per_iter_ms),
        per_iter_max_ms=max(per_iter_ms),
        per_img_ms=statistics.mean(per_iter_ms) / BATCH,
        throughput_ips=n_samples / total_s,
        peak_gib=torch.cuda.max_memory_allocated() / 2**30,
    )


def main():
    torch.manual_seed(42)
    torch.cuda.empty_cache()

    is_sj = BACKEND.startswith("sj_")
    if BACKEND == "ours":
        model = build_ours()
    elif BACKEND == "sj_eager":
        model = build_sj("eager")
    elif BACKEND == "sj_compile":
        model = build_sj("compile")
    else:
        raise ValueError(f"unknown BACKEND: {BACKEND}")

    model = model.eval().cuda()
    if DTYPE != torch.float32:
        model = model.to(DTYPE)

    # NHWC weights layout（仅 ours 走 NHWC）
    use_nhwc = (LAYOUT == "NHWC" and BACKEND == "ours")
    if use_nhwc:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = m.weight.data.to(memory_format=torch.channels_last)

    x = torch.randn(T, BATCH, 3, INPUT_H, INPUT_H, device="cuda", dtype=DTYPE)

    # torch.compile 包装（sj_compile 路径）
    if BACKEND == "sj_compile":
        from torch import _dynamo, _inductor
        _dynamo.config.recompile_limit = 256
        _dynamo.config.cache_size_limit = 256
        _inductor.config.max_autotune = True
        _inductor.config.max_autotune_gemm_backends = "TRITON"
        _inductor.config.max_autotune_conv_backends = "TRITON"
        runnable = torch.compile(model)
    else:
        runnable = model

    print(f"\n[config] BACKEND={BACKEND}  ARCH={ARCH}  BATCH={BATCH}  T={T}  "
          f"TOTAL={TOTAL}  MODE={DTYPE}  LAYOUT={LAYOUT}  ITERS={ITERS}")
    print(f"  input shape: {tuple(x.shape)}  use_nhwc={use_nhwc}")

    # cold start
    t0 = time.perf_counter()
    if BACKEND == "sj_compile":
        from spikingjelly.activation_based import functional
        functional.reset_net(model)
    with torch.no_grad():
        _ = runnable(x)
    torch.cuda.synchronize()
    cold_s = time.perf_counter() - t0
    print(f"[cold] {cold_s:.1f}s")

    r = measure(runnable, x, is_sj=is_sj)

    print(f"\n{'=' * 78}")
    print(f"  {BACKEND} | {ARCH} | B={BATCH} T={T} dt={DTYPE}{' NHWC' if use_nhwc else ' NCHW'}")
    print(f"  iters       = {r['iters']}    n_samples = {r['n_samples']}")
    print(f"  total       = {r['total_s']:.2f} s")
    print(f"  per-iter    = mean {r['per_iter_mean_ms']:.3f} ms | "
          f"median {r['per_iter_median_ms']:.3f} | std {r['per_iter_std_ms']:.3f}")
    print(f"  per-img     = {r['per_img_ms']:.5f} ms")
    print(f"  throughput  = {r['throughput_ips']:.1f} img/s")
    print(f"  peak mem    = {r['peak_gib']:.2f} GiB")
    print(f"  cold        = {cold_s:.1f} s")
    print('=' * 78)

    fname = RESULTS_DIR / f"bench_50k.jsonl"
    with open(fname, "a") as f:
        f.write(json.dumps({
            "tag": TAG, "backend": BACKEND, "arch": ARCH,
            "batch": BATCH, "T": T, "total": TOTAL,
            "dtype": str(DTYPE).replace("torch.", ""),
            "layout": LAYOUT if use_nhwc else "NCHW",
            **r, "cold_s": cold_s,
        }) + "\n")
    print(f"[results appended → {fname}]")


if __name__ == "__main__":
    main()
