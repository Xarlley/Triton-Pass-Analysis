"""集成 fused_bias_if kernel：Conv2d(bias=False) + FusedBiasIF 替代分离的 conv+bias_add+IF。

仅 conv 后的 IF 用 FusedBiasIF；fc 后的 IF 仍用 FusedIF（fc 的 bias 由 cuBLAS addmm 处理）。
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
import torch.nn.functional as F
import torch._dynamo
import torch._inductor.config as inductor_cfg

from fused_if_kernel import fused_if
from fused_bias_if_kernel import fused_bias_if


BATCH = int(os.environ.get("BATCH", 56))
TOTAL_SAMPLES = int(os.environ.get("TOTAL_SAMPLES", 10000))
WARMUP = int(os.environ.get("WARMUP", 5))
MODE = os.environ.get("MODE", "eager").lower()
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


# ========== 自定义 ConvBiasIF 层：F.conv2d(bias=None) + fused_bias_if ==========
class ConvBiasIFNode(nn.Module):
    """Conv2d (bias 留在外面) + FusedBiasIF —— 一个层算完整套 conv+IF。

    forward(x:[T,B,C_in,H,W]) → spike:[T,B,C_out,H_out,W_out]
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 v_threshold=1.0, soft_reset=False):
        super().__init__()
        # 复用 nn.Conv2d 的参数初始化，但 bias=False 阻止它自己加 bias
        c = nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=padding, bias=True)
        self.weight = c.weight
        self.bias = c.bias
        self.stride = c.stride
        self.padding = c.padding
        self.dilation = c.dilation
        self.groups = c.groups
        self.v_threshold = v_threshold
        self.soft_reset = soft_reset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B = x.shape[0], x.shape[1]
        # conv (no bias), [T·B, C_out, H_out, W_out]
        y = F.conv2d(x.flatten(0, 1), self.weight, bias=None,
                     stride=self.stride, padding=self.padding,
                     dilation=self.dilation, groups=self.groups)
        # reshape back to [T, B, C_out, H_out, W_out]
        y5 = y.view(T, B, *y.shape[1:]).contiguous()
        # 调用 fused_bias_if_op (custom_op below) 算完 IF
        return torch.ops.xarlley.fused_bias_if(y5, self.bias, self.soft_reset, self.v_threshold)


# ========== FusedIFNode for FC layers (no bias arg needed; bias 在 Linear 内) ==========
class FusedIFNode(nn.Module):
    def __init__(self, v_threshold=1.0, soft_reset=False):
        super().__init__()
        self.v_threshold = v_threshold
        self.soft_reset = soft_reset
    def forward(self, x):
        return torch.ops.xarlley.fused_if(x.contiguous(), self.soft_reset, self.v_threshold)


# ========== custom_op 注册 ==========
from torch.library import custom_op, register_fake


@custom_op("xarlley::fused_if", mutates_args=())
def _fused_if_op(x: torch.Tensor, soft_reset: bool, v_threshold: float) -> torch.Tensor:
    return fused_if(x.contiguous(), soft_reset=soft_reset, v_threshold=v_threshold)

@register_fake("xarlley::fused_if")
def _fused_if_fake(x, soft_reset, v_threshold):
    return torch.empty_like(x)


@custom_op("xarlley::fused_bias_if", mutates_args=())
def _fused_bias_if_op(x: torch.Tensor, bias: torch.Tensor,
                       soft_reset: bool, v_threshold: float) -> torch.Tensor:
    return fused_bias_if(x.contiguous(), bias, soft_reset=soft_reset, v_threshold=v_threshold)

@register_fake("xarlley::fused_bias_if")
def _fused_bias_if_fake(x, bias, soft_reset, v_threshold):
    return torch.empty_like(x)


# ========== Wrapper for stateless layers (AvgPool, Flatten, Linear) ==========
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


def build():
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    use_soft = (RESET == "soft")

    feats, in_ch = [], 3
    for v in VGG16_CFG:
        if v == "P":
            feats.append(TimeBatchWrapper(nn.AvgPool2d(2, 2)))
        else:
            # Conv2d + FusedBiasIF 合二为一
            feats.append(ConvBiasIFNode(in_ch, v, 3, padding=1,
                                         v_threshold=1.0, soft_reset=use_soft))
            in_ch = v
    classifier = nn.Sequential(
        TimeBatchWrapper(nn.Flatten()),
        TimeBatchWrapper(nn.Linear(512 * 7 * 7, 4096)),
        FusedIFNode(v_threshold=1.0, soft_reset=use_soft),
        TimeBatchWrapper(nn.Linear(4096, 4096)),
        FusedIFNode(v_threshold=1.0, soft_reset=use_soft),
        TimeBatchWrapper(nn.Linear(4096, num_classes := 1000)),
    )
    m = nn.Sequential(nn.Sequential(*feats), classifier)
    return m.eval().cuda()


def main():
    print(f"[config] MODE={MODE}  RESET={RESET}  BATCH={BATCH}  T={T}  MEASURE_ITERS={MEASURE_ITERS}")
    print(f"\n[build] ConvBiasIF VGG16-SNN")
    m = build()
    runnable = torch.compile(m) if MODE == "compile" else m
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

    print(f"[measure] {MEASURE_ITERS} iters × BATCH={BATCH}={MEASURE_ITERS * BATCH} samples ...")
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
    per_img = mean_iter / BATCH

    print("\n" + "=" * 78)
    print(f"  MODE = ConvBiasIF-{MODE}-{RESET}    BATCH = {BATCH}")
    print(f"  forward 调用次数 = {MEASURE_ITERS}     总样本 = {n_samples}")
    print(f"  总耗时           = {total_s:.4f} s")
    print(f"  每次 forward     = avg {mean_iter:.4f} ms | std {std_iter:.4f}")
    print(f"  单张折算         = avg {per_img:.5f} ms / 张")
    print(f"  吞吐             = {n_samples / total_s:.2f} 张/秒")
    print(f"  GPU peak memory  = {peak_mem:.2f} GiB")
    print(f"  cold compile     = {cold_s:.1f} s")
    print("=" * 78)

    result = {
        "mode": f"ConvBiasIF-{MODE}-{RESET}",
        "batch": BATCH, "iters": MEASURE_ITERS, "n_samples": n_samples,
        "total_s": total_s, "mean_iter_ms": mean_iter, "std_iter_ms": std_iter,
        "mean_per_img_ms": per_img, "throughput_imgs_per_s": n_samples / total_s,
        "peak_mem_gib": peak_mem, "compile_s": cold_s,
    }
    with open("/tmp/cold_start_results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
