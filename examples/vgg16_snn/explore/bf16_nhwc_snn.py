"""bf16 + channels_last 组合：把两个独立优化叠起来。

bf16 主要省 conv 计算时间（tensor core）+ memory bandwidth；
channels_last 主要省 layout xform scratch buffer + 显存峰值。
两者效果应该叠加（不冲突）。
"""
import os, sys, json, time, statistics, pathlib
HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "explore"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

BATCH = int(os.environ.get("BATCH", 56))
TOTAL_SAMPLES = int(os.environ.get("TOTAL_SAMPLES", 10000))
WARMUP = int(os.environ.get("WARMUP", 5))
SEED = 42
T = 4
DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
MEASURE_ITERS = (TOTAL_SAMPLES + BATCH - 1) // BATCH


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": 128}, num_warps=4),
        triton.Config({"BLOCK_NCL": 256}, num_warps=4),
        triton.Config({"BLOCK_NCL": 256}, num_warps=8),
        triton.Config({"BLOCK_NCL": 512}, num_warps=8),
    ],
    key=["T", "NCL", "C", "SOFT_RESET"],
)
@triton.jit
def _fused_bias_if_bf16_nhwc_kernel(
    x_ptr, bias_ptr, spike_ptr,
    T: tl.constexpr, NCL: tl.constexpr, C: tl.constexpr,
    BLOCK_NCL: tl.constexpr, v_threshold: tl.constexpr, SOFT_RESET: tl.constexpr,
):
    pid = tl.program_id(0)
    ncl_idx = pid * BLOCK_NCL + tl.arange(0, BLOCK_NCL)
    mask = ncl_idx < NCL
    # NHWC 布局：c 最内层，bias index = ncl_idx % C
    c_idx = ncl_idx % C
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    v = tl.zeros([BLOCK_NCL], dtype=tl.float32)
    for t in tl.static_range(0, T, 1):
        x_t = tl.load(x_ptr + t * NCL + ncl_idx, mask=mask, other=0.0).to(tl.float32)
        v = v + x_t + bias
        spike = (v >= v_threshold).to(tl.float32)
        if SOFT_RESET: v = v - spike * v_threshold
        else:          v = v * (1.0 - spike)
        tl.store(spike_ptr + t * NCL + ncl_idx, spike.to(tl.bfloat16), mask=mask)


class ConvBiasIFBf16NHWCNode(nn.Module):
    def __init__(self, in_ch, out_ch, k, padding=0, v_thr=1.0, soft=False):
        super().__init__()
        c = nn.Conv2d(in_ch, out_ch, k, padding=padding, bias=True)
        # weight 是 channels_last 4D, dtype bf16
        self.weight = nn.Parameter(
            c.weight.detach().to(DTYPE).to(memory_format=torch.channels_last)
        )
        self.bias = nn.Parameter(c.bias.detach().to(DTYPE))
        self.padding = c.padding
        self.stride = c.stride
        self.v_threshold = v_thr
        self.soft_reset = soft

    def forward(self, x_4d_cl):
        y = F.conv2d(x_4d_cl, self.weight, bias=None,
                     stride=self.stride, padding=self.padding)
        TB, C, H, W = y.shape
        B = TB // T
        NCL = B * C * H * W
        spike = torch.empty_like(y)
        grid = lambda meta: (triton.cdiv(NCL, meta["BLOCK_NCL"]),)
        _fused_bias_if_bf16_nhwc_kernel[grid](
            y, self.bias, spike,
            T=T, NCL=NCL, C=C,
            v_threshold=self.v_threshold,
            SOFT_RESET=self.soft_reset,
        )
        return spike


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": 128}, num_warps=4),
        triton.Config({"BLOCK_NCL": 256}, num_warps=4),
        triton.Config({"BLOCK_NCL": 512}, num_warps=8),
    ],
    key=["T", "NCL", "SOFT_RESET"],
)
@triton.jit
def _fused_if_bf16_kernel(
    x_ptr, spike_ptr,
    T: tl.constexpr, NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr, v_threshold: tl.constexpr, SOFT_RESET: tl.constexpr,
):
    pid = tl.program_id(0)
    ncl_idx = pid * BLOCK_NCL + tl.arange(0, BLOCK_NCL)
    mask = ncl_idx < NCL
    v = tl.zeros([BLOCK_NCL], dtype=tl.float32)
    for t in tl.static_range(0, T, 1):
        x_t = tl.load(x_ptr + t * NCL + ncl_idx, mask=mask, other=0.0).to(tl.float32)
        v = v + x_t
        spike = (v >= v_threshold).to(tl.float32)
        if SOFT_RESET: v = v - spike * v_threshold
        else:          v = v * (1.0 - spike)
        tl.store(spike_ptr + t * NCL + ncl_idx, spike.to(tl.bfloat16), mask=mask)


class FusedIFBf16Node(nn.Module):
    def __init__(self, v_thr=1.0, soft=False):
        super().__init__()
        self.v_threshold = v_thr
        self.soft_reset = soft
    def forward(self, x):
        TB = x.shape[0]
        NCL = x.numel() // T
        spike = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(NCL, meta["BLOCK_NCL"]),)
        _fused_if_bf16_kernel[grid](
            x.contiguous(), spike,
            T=T, NCL=NCL,
            v_threshold=self.v_threshold,
            SOFT_RESET=self.soft_reset,
        )
        return spike


VGG16_CFG = [64, 64, "P", 128, 128, "P", 256, 256, 256, "P",
             512, 512, 512, "P", 512, 512, 512, "P"]


class Bf16NhwcVGG16SNN(nn.Module):
    def __init__(self, num_classes=1000, soft=False):
        super().__init__()
        feats, in_ch = [], 3
        for v in VGG16_CFG:
            if v == "P":
                feats.append(nn.AvgPool2d(2, 2))
            else:
                feats.append(ConvBiasIFBf16NHWCNode(in_ch, v, 3, padding=1, soft=soft))
                in_ch = v
        self.features = nn.Sequential(*feats)
        self.fc1 = nn.Linear(512*7*7, 4096).to(DTYPE)
        self.if1 = FusedIFBf16Node(soft=soft)
        self.fc2 = nn.Linear(4096, 4096).to(DTYPE)
        self.if2 = FusedIFBf16Node(soft=soft)
        self.fc3 = nn.Linear(4096, num_classes).to(DTYPE)

    def forward(self, x_5d):
        T_, B = x_5d.shape[0], x_5d.shape[1]
        x = x_5d.flatten(0, 1).to(memory_format=torch.channels_last)  # bf16 + cl
        x = self.features(x)
        x = x.contiguous().flatten(1)                                  # [T*B, 25088]
        x = self.fc1(x); x = self.if1(x)
        x = self.fc2(x); x = self.if2(x)
        x = self.fc3(x)
        return x.view(T_, B, *x.shape[1:])


def main():
    print(f"[config] bf16+NHWC BATCH={BATCH} T={T}")
    torch.manual_seed(SEED)
    m = Bf16NhwcVGG16SNN(1000).eval().cuda()
    x = torch.randn(T, BATCH, 3, 224, 224, device=DEVICE, dtype=DTYPE)

    print("[cold-start]...")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = m(x)
    torch.cuda.synchronize()
    cold = time.perf_counter() - t0
    print(f"[cold-start] {cold:.1f}s out={tuple(out.shape)}")

    torch.cuda.reset_peak_memory_stats()
    for _ in range(WARMUP):
        with torch.no_grad():
            m(x)
    torch.cuda.synchronize()

    per_iter = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(MEASURE_ITERS):
        ti = time.perf_counter()
        with torch.no_grad():
            m(x)
        torch.cuda.synchronize()
        per_iter.append((time.perf_counter() - ti) * 1000)
    total_s = time.perf_counter() - t0

    peak = torch.cuda.max_memory_allocated() / 2**30
    n = MEASURE_ITERS * BATCH
    mean = statistics.mean(per_iter)
    per_img = mean / BATCH

    print(f"\n========================================")
    print(f"  MODE = ConvBiasIF-bf16-NHWC  BATCH={BATCH}")
    print(f"  per-img = {per_img:.5f} ms  | iter = {mean:.4f} ms")
    print(f"  throughput = {n / total_s:.2f} 张/秒")
    print(f"  peak mem = {peak:.2f} GiB")
    print(f"  cold = {cold:.1f} s")
    print(f"========================================")

    with open("/tmp/cold_start_results.jsonl", "a") as f:
        f.write(json.dumps({
            "mode": "ConvBiasIF-bf16-NHWC-eager",
            "batch": BATCH, "iters": MEASURE_ITERS, "n_samples": n,
            "total_s": total_s, "mean_iter_ms": mean,
            "mean_per_img_ms": per_img,
            "throughput_imgs_per_s": n / total_s,
            "peak_mem_gib": peak, "compile_s": cold,
        }) + "\n")


if __name__ == "__main__":
    main()
