"""传统 ANN VGG16-D baseline：作为 SNN 优化对照。

任务：与 SNN 同样在 RTX 5070 Ti 上跑 VGG16 推理，但用标准 ReLU 而非 IF/LIF 神经元，
单步而非多步（无 T 维）。
- 网络：VGG16-D (13 Conv + 13 BN + 13 ReLU + 5 MaxPool + 3 FC + 1000 classes)
- 精度：bf16 (与 SNN 最佳配置一致)
- 内存布局：channels_last (与 SNN 最佳配置一致)
- 模式：eager (我们已证明 compile mode 在该 GPU 上比 eager 慢)
- 输入：随机 fp32→bf16 [B, 3, 224, 224]
- BATCH：在不同 size 下扫一遍，找出 GPU 饱和点
- 任务规模：默认 50000 (= ImageNet val 集大小)，可调到 10000

参考对比：
- SNN 最佳: 1.88 ms/张 (bf16 + NHWC, T=4, BATCH=192) — 含每图 4 时间步
- ANN 单图: 没有 T 维，理论上 conv 工作量只有 SNN 的 1/4
"""
import os, json, time, statistics, pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F


BATCH = int(os.environ.get("BATCH", 192))
TOTAL_SAMPLES = int(os.environ.get("TOTAL_SAMPLES", 50000))   # ImageNet val 集大小
WARMUP = int(os.environ.get("WARMUP", 5))
MODE = os.environ.get("MODE", "eager").lower()                # eager | compile
NET = os.environ.get("NET", "vgg16_bn").lower()               # vgg16_bn (BN+MaxPool+ReLU)
DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
MEASURE_ITERS = (TOTAL_SAMPLES + BATCH - 1) // BATCH

if MODE == "compile":
    import torch._dynamo
    import torch._inductor.config as inductor_cfg
    torch._dynamo.config.recompile_limit = 256
    torch._dynamo.config.cache_size_limit = 256
    inductor_cfg.max_autotune = True
    inductor_cfg.max_autotune_gemm_backends = "TRITON"
    inductor_cfg.max_autotune_conv_backends = "TRITON"
    inductor_cfg.force_disable_caches = True

print(f"[config] BATCH={BATCH} NET={NET} MODE={MODE} MEASURE_ITERS={MEASURE_ITERS} "
      f"目标样本 ≈ {MEASURE_ITERS * BATCH} (≥{TOTAL_SAMPLES})")


VGG16_BN_CFG = [
    (64, "C"), (64, "C"), "M",
    (128, "C"), (128, "C"), "M",
    (256, "C"), (256, "C"), (256, "C"), "M",
    (512, "C"), (512, "C"), (512, "C"), "M",
    (512, "C"), (512, "C"), (512, "C"), "M",
]


def build_vgg16_bn(num_classes=1000):
    """标准 VGG16-D：13 Conv + 13 BN + 13 ReLU + 5 MaxPool + 3 FC + 1000-class"""
    feats, in_ch = [], 3
    for item in VGG16_BN_CFG:
        if item == "M":
            feats.append(nn.MaxPool2d(2, 2))
        else:
            ch, _ = item
            feats.extend([
                nn.Conv2d(in_ch, ch, 3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ])
            in_ch = ch
    classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True), nn.Dropout(0.0),
        nn.Linear(4096, 4096),        nn.ReLU(inplace=True), nn.Dropout(0.0),
        nn.Linear(4096, num_classes),
    )
    return nn.Sequential(nn.Sequential(*feats), classifier)


def build_vgg16_nobn_avgpool(num_classes=1000):
    """ANN-equivalent of our SNN structure: 13 Conv (no BN) + 5 AvgPool + ReLU + 3 FC"""
    feats, in_ch = [], 3
    for item in VGG16_BN_CFG:
        if item == "M":
            feats.append(nn.AvgPool2d(2, 2))
        else:
            ch, _ = item
            feats.extend([
                nn.Conv2d(in_ch, ch, 3, padding=1),
                nn.ReLU(inplace=True),
            ])
            in_ch = ch
    classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True),
        nn.Linear(4096, 4096),        nn.ReLU(inplace=True),
        nn.Linear(4096, num_classes),
    )
    return nn.Sequential(nn.Sequential(*feats), classifier)


def main():
    torch.manual_seed(42)
    if NET == "vgg16_bn":
        m = build_vgg16_bn(1000)
        print(f"[build] VGG16-D (13 Conv + 13 BN + 13 ReLU + 5 MaxPool + 3 FC)")
    elif NET == "vgg16_nobn":
        m = build_vgg16_nobn_avgpool(1000)
        print(f"[build] VGG16 (no BN, AvgPool, ReLU, 3 FC) — matches our SNN structure")
    else:
        raise ValueError(f"unknown NET: {NET}")

    # bf16 + channels_last + cuda + eval
    m = m.eval().to(DTYPE).cuda()
    # weights to channels_last where applicable (4D conv weights)
    for mod in m.modules():
        if isinstance(mod, nn.Conv2d):
            mod.weight.data = mod.weight.data.to(memory_format=torch.channels_last)

    n_params = sum(p.numel() for p in m.parameters())
    print(f"  params: {n_params / 1e6:.1f}M  dtype: {DTYPE}")

    runnable = torch.compile(m) if MODE == "compile" else m

    # 输入：随机 bf16, channels_last
    x = torch.randn(BATCH, 3, 224, 224, device=DEVICE, dtype=DTYPE)
    x = x.to(memory_format=torch.channels_last)
    print(f"  input shape: {tuple(x.shape)}")

    # cold-start
    print(f"\n[cold-start]...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = runnable(x)
    torch.cuda.synchronize()
    cold_s = time.perf_counter() - t0
    print(f"[cold-start] {cold_s:.1f}s  out={tuple(out.shape)}  dtype={out.dtype}")

    torch.cuda.reset_peak_memory_stats()
    for _ in range(WARMUP):
        with torch.no_grad():
            runnable(x)
    torch.cuda.synchronize()

    print(f"[measure] {MEASURE_ITERS} iters × BATCH={BATCH} = {MEASURE_ITERS * BATCH} samples ...")
    per_iter_ms = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(MEASURE_ITERS):
        ti = time.perf_counter()
        with torch.no_grad():
            runnable(x)
        torch.cuda.synchronize()
        per_iter_ms.append((time.perf_counter() - ti) * 1000)
    total_s = time.perf_counter() - t0

    peak_mem = torch.cuda.max_memory_allocated() / 2**30
    n_samples = MEASURE_ITERS * BATCH
    mean_iter = statistics.mean(per_iter_ms)
    std_iter = statistics.stdev(per_iter_ms)
    median_iter = statistics.median(per_iter_ms)
    per_img = mean_iter / BATCH

    print("\n" + "=" * 78)
    print(f"  MODE = ANN-{NET}-{MODE}    BATCH = {BATCH}    dtype = {DTYPE}")
    print(f"  forward 调用次数 = {MEASURE_ITERS}     总样本 = {n_samples}")
    print(f"  总耗时           = {total_s:.4f} s")
    print(f"  每次 forward     = avg {mean_iter:.4f} ms | median {median_iter:.4f} "
          f"| std {std_iter:.4f}")
    print(f"  单张折算         = avg {per_img:.5f} ms / 张")
    print(f"  吞吐             = {n_samples / total_s:.2f} 张/秒")
    print(f"  GPU peak memory  = {peak_mem:.2f} GiB")
    print(f"  cold compile     = {cold_s:.1f} s")
    print("=" * 78)

    with open("/tmp/cold_start_results.jsonl", "a") as f:
        f.write(json.dumps({
            "mode": f"ANN-{NET}-{MODE}",
            "batch": BATCH, "iters": MEASURE_ITERS, "n_samples": n_samples,
            "total_s": total_s, "mean_iter_ms": mean_iter,
            "median_iter_ms": median_iter, "std_iter_ms": std_iter,
            "min_iter_ms": min(per_iter_ms), "max_iter_ms": max(per_iter_ms),
            "mean_per_img_ms": per_img,
            "throughput_imgs_per_s": n_samples / total_s,
            "peak_mem_gib": peak_mem, "compile_s": cold_s,
        }) + "\n")


if __name__ == "__main__":
    main()
