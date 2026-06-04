"""集成手工融合 IF kernel 到 VGG16-SNN，端到端 10024 样本基准。

把 PrefixSumHardResetIFNode 替换为 FusedIFNode（内部调 fused_if Triton kernel
via torch.library.custom_op）。其余结构（13 Conv + 5 AvgPool + 3 FC，无 BN）
与 prefix_sum_snn.py 完全一致。
"""
import os
import sys
import json
import time
import statistics
import pathlib

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "explore"))

import torch
import torch.nn as nn
import torch._dynamo
import torch._inductor.config as inductor_cfg

from fused_if_kernel import fused_if


BATCH = int(os.environ.get("BATCH", 56))
TOTAL_SAMPLES = int(os.environ.get("TOTAL_SAMPLES", 10000))
WARMUP = int(os.environ.get("WARMUP", 5))
MODE = os.environ.get("MODE", "compile").lower()
RESET = os.environ.get("RESET", "hard").lower()
SEED = 42
T = 4
NUM_CLASSES = 1000
DEVICE = torch.device("cuda")
MEASURE_ITERS = (TOTAL_SAMPLES + BATCH - 1) // BATCH

if MODE == "compile":
    torch._dynamo.config.recompile_limit = 256
    torch._dynamo.config.cache_size_limit = 256
    inductor_cfg.max_autotune = True
    inductor_cfg.max_autotune_gemm_backends = "TRITON"
    inductor_cfg.max_autotune_conv_backends = "TRITON"
    inductor_cfg.force_disable_caches = True


# ============ 把 fused_if 注册为 torch.library.custom_op，让 dynamo 看到它 ============
from torch.library import custom_op, register_fake


@custom_op("xarlley::fused_if", mutates_args=())
def fused_if_op(x_seq: torch.Tensor, soft_reset: bool, v_threshold: float) -> torch.Tensor:
    """x_seq: [T, B, ...] contiguous fp32，返回同形 spike_seq。"""
    return fused_if(x_seq.contiguous(), soft_reset=soft_reset, v_threshold=v_threshold)


@register_fake("xarlley::fused_if")
def _fused_if_fake(x_seq, soft_reset, v_threshold):
    return torch.empty_like(x_seq)


class FusedIFNode(nn.Module):
    """用手工融合 Triton kernel 实现的 IF 神经元（替换 PrefixSumIFNode）。

    在 eager 路径下直接调 fused_if；torch.compile 下通过 custom_op 暴露给 dynamo
    作为黑盒 launcher（不会被 Inductor 重新 codegen）。
    """

    def __init__(self, v_threshold: float = 1.0, soft_reset: bool = False):
        super().__init__()
        self.v_threshold = v_threshold
        self.soft_reset = soft_reset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.xarlley.fused_if(x, self.soft_reset, self.v_threshold)


class TimeBatchWrapper(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    def forward(self, x):
        T, B = x.shape[0], x.shape[1]
        y = self.layer(x.flatten(0, 1))
        return y.view(T, B, *y.shape[1:])


VGG16_CFG = [64, 64, "P", 128, 128, "P", 256, 256, 256, "P",
             512, 512, 512, "P", 512, 512, 512, "P"]


def build_fused_if_vgg16(num_classes: int = NUM_CLASSES):
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    use_soft = (RESET == "soft")

    feats, in_ch = [], 3
    for v in VGG16_CFG:
        if v == "P":
            feats.append(TimeBatchWrapper(nn.AvgPool2d(2, 2)))
        else:
            feats.append(TimeBatchWrapper(nn.Conv2d(in_ch, v, 3, padding=1)))
            feats.append(FusedIFNode(v_threshold=1.0, soft_reset=use_soft))
            in_ch = v
    classifier = nn.Sequential(
        TimeBatchWrapper(nn.Flatten()),
        TimeBatchWrapper(nn.Linear(512 * 7 * 7, 4096)),
        FusedIFNode(v_threshold=1.0, soft_reset=use_soft),
        TimeBatchWrapper(nn.Linear(4096, 4096)),
        FusedIFNode(v_threshold=1.0, soft_reset=use_soft),
        TimeBatchWrapper(nn.Linear(4096, num_classes)),
    )
    model = nn.Sequential(nn.Sequential(*feats), classifier)
    return model.eval().cuda()


def main():
    print(f"[config] MODE={MODE}  RESET={RESET}  BATCH={BATCH}  T={T}  "
          f"MEASURE_ITERS={MEASURE_ITERS}  WARMUP={WARMUP}")
    print(f"\n[build] FusedIF VGG16-SNN ({'soft' if RESET == 'soft' else 'hard'}-reset)")

    model = build_fused_if_vgg16(NUM_CLASSES)
    runnable = torch.compile(model) if MODE == "compile" else model
    x = torch.randn(T, BATCH, 3, 224, 224, device=DEVICE)
    print(f"  input shape: {tuple(x.shape)}")

    label = "compile + autotune" if MODE == "compile" else "首次 forward"
    print(f"\n[cold-start] {label}...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = runnable(x)
    torch.cuda.synchronize()
    cold_s = time.perf_counter() - t0
    print(f"[cold-start] {cold_s:.1f}s  out={tuple(out.shape)}")

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
    print(f"  MODE = FusedIF-{MODE}-{RESET}    BATCH = {BATCH}")
    print(f"  forward 调用次数 = {MEASURE_ITERS}     总样本 = {n_samples}")
    print(f"  总耗时           = {total_s:.4f} s")
    print(f"  每次 forward     = avg {mean_iter:.4f} ms | median {median_iter:.4f} "
          f"| std {std_iter:.4f}")
    print(f"  单张折算         = avg {per_img:.5f} ms / 张")
    print(f"  吞吐             = {n_samples / total_s:.2f} 张/秒")
    print(f"  GPU peak memory  = {peak_mem:.2f} GiB")
    print(f"  cold compile     = {cold_s:.1f} s")
    print("=" * 78)

    result = {
        "mode": f"FusedIF-{MODE}-{RESET}",
        "batch": BATCH, "iters": MEASURE_ITERS, "n_samples": n_samples,
        "total_s": total_s, "mean_iter_ms": mean_iter,
        "median_iter_ms": median_iter, "std_iter_ms": std_iter,
        "min_iter_ms": min(per_iter_ms), "max_iter_ms": max(per_iter_ms),
        "mean_per_img_ms": per_img,
        "throughput_imgs_per_s": n_samples / total_s,
        "peak_mem_gib": peak_mem, "compile_s": cold_s,
    }
    with open("/tmp/cold_start_results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
