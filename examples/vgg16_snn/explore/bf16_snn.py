"""bf16 (Brain Float 16) VGG16-SNN：让 Blackwell tensor core 全速跑 conv。

bf16 在 sm_120 上 tensor core 吞吐高于 fp32/TF32：
- bf16 与 fp32 同 exponent range，无 overflow 风险
- mantissa 比 fp32 少 16 bit，对 4 步 IF 阈值比较够用
- conv / gemm 的 HBM 流量减半

IF kernel: 输入 bf16, 但 v 累加器保持 fp32（防止 4 步累加溢精度）。
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
    key=["T", "NCL", "HW", "SOFT_RESET"],
)
@triton.jit
def _fused_bias_if_bf16_kernel(
    x_ptr,                       # [T, NCL] bf16
    bias_ptr,                    # [C] bf16
    spike_ptr,                   # [T, NCL] bf16
    T: tl.constexpr,
    NCL: tl.constexpr,
    HW: tl.constexpr,
    C: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    v_threshold: tl.constexpr,
    SOFT_RESET: tl.constexpr,
):
    pid = tl.program_id(0)
    ncl_idx = pid * BLOCK_NCL + tl.arange(0, BLOCK_NCL)
    mask = ncl_idx < NCL

    c_idx = (ncl_idx // HW) % C
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)

    # ★ v 累加器保持 fp32 (高精度)，输入/输出 bf16 (高带宽)
    v = tl.zeros([BLOCK_NCL], dtype=tl.float32)

    for t in tl.static_range(0, T, 1):
        x_t = tl.load(x_ptr + t * NCL + ncl_idx, mask=mask, other=0.0).to(tl.float32)
        v = v + x_t + bias
        spike = (v >= v_threshold).to(tl.float32)
        if SOFT_RESET:
            v = v - spike * v_threshold
        else:
            v = v * (1.0 - spike)
        tl.store(spike_ptr + t * NCL + ncl_idx, spike.to(tl.bfloat16), mask=mask)


def fused_bias_if_bf16(x_seq, bias, soft_reset=False, v_threshold=1.0):
    assert x_seq.is_cuda and x_seq.dtype == torch.bfloat16
    T, B, C, H, W = x_seq.shape
    NCL = B * C * H * W
    spike = torch.empty_like(x_seq)
    grid = lambda meta: (triton.cdiv(NCL, meta["BLOCK_NCL"]),)
    _fused_bias_if_bf16_kernel[grid](
        x_seq.contiguous(), bias.to(torch.bfloat16), spike,
        T=T, NCL=NCL, HW=H * W, C=C,
        v_threshold=v_threshold,
        SOFT_RESET=soft_reset,
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
    BLOCK_NCL: tl.constexpr,
    v_threshold: tl.constexpr,
    SOFT_RESET: tl.constexpr,
):
    pid = tl.program_id(0)
    ncl_idx = pid * BLOCK_NCL + tl.arange(0, BLOCK_NCL)
    mask = ncl_idx < NCL
    v = tl.zeros([BLOCK_NCL], dtype=tl.float32)
    for t in tl.static_range(0, T, 1):
        x_t = tl.load(x_ptr + t * NCL + ncl_idx, mask=mask, other=0.0).to(tl.float32)
        v = v + x_t
        spike = (v >= v_threshold).to(tl.float32)
        if SOFT_RESET:
            v = v - spike * v_threshold
        else:
            v = v * (1.0 - spike)
        tl.store(spike_ptr + t * NCL + ncl_idx, spike.to(tl.bfloat16), mask=mask)


def fused_if_bf16(x_seq, soft_reset=False, v_threshold=1.0):
    assert x_seq.dtype == torch.bfloat16
    T = x_seq.shape[0]
    NCL = x_seq[0].numel()
    spike = torch.empty_like(x_seq)
    grid = lambda meta: (triton.cdiv(NCL, meta["BLOCK_NCL"]),)
    _fused_if_bf16_kernel[grid](
        x_seq.contiguous(), spike,
        T=T, NCL=NCL,
        v_threshold=v_threshold,
        SOFT_RESET=soft_reset,
    )
    return spike


class ConvBiasIFBf16Node(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 v_threshold=1.0, soft_reset=False):
        super().__init__()
        c = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=True)
        self.weight = nn.Parameter(c.weight.detach().to(DTYPE))
        self.bias = nn.Parameter(c.bias.detach().to(DTYPE))
        self.padding = c.padding
        self.stride = c.stride
        self.v_threshold = v_threshold
        self.soft_reset = soft_reset

    def forward(self, x):
        # x: [T, B, C_in, H, W] bf16
        T_, B = x.shape[0], x.shape[1]
        y = F.conv2d(x.flatten(0, 1), self.weight, bias=None,
                     stride=self.stride, padding=self.padding)
        y5 = y.view(T_, B, *y.shape[1:]).contiguous()
        return fused_bias_if_bf16(y5, self.bias, self.soft_reset, self.v_threshold)


class FusedIFBf16Node(nn.Module):
    def __init__(self, v_threshold=1.0, soft_reset=False):
        super().__init__()
        self.v_threshold = v_threshold
        self.soft_reset = soft_reset
    def forward(self, x):
        return fused_if_bf16(x.contiguous(), soft_reset=self.soft_reset, v_threshold=self.v_threshold)


class TimeBatchWrapper(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    def forward(self, x):
        T_, B = x.shape[0], x.shape[1]
        y = self.layer(x.flatten(0, 1))
        return y.view(T_, B, *y.shape[1:])


VGG16_CFG = [64, 64, "P", 128, 128, "P", 256, 256, 256, "P",
             512, 512, 512, "P", 512, 512, 512, "P"]


def build():
    torch.manual_seed(SEED)
    feats, in_ch = [], 3
    for v in VGG16_CFG:
        if v == "P":
            feats.append(TimeBatchWrapper(nn.AvgPool2d(2, 2)))
        else:
            feats.append(ConvBiasIFBf16Node(in_ch, v, 3, padding=1,
                                             v_threshold=1.0, soft_reset=False))
            in_ch = v
    cls = nn.Sequential(
        TimeBatchWrapper(nn.Flatten()),
        TimeBatchWrapper(nn.Linear(512*7*7, 4096).to(DTYPE)),
        FusedIFBf16Node(v_threshold=1.0, soft_reset=False),
        TimeBatchWrapper(nn.Linear(4096, 4096).to(DTYPE)),
        FusedIFBf16Node(v_threshold=1.0, soft_reset=False),
        TimeBatchWrapper(nn.Linear(4096, 1000).to(DTYPE)),
    )
    m = nn.Sequential(nn.Sequential(*feats), cls)
    return m.eval().cuda()


def main():
    print(f"[config] bf16 BATCH={BATCH} T={T}")
    model = build()
    x = torch.randn(T, BATCH, 3, 224, 224, device=DEVICE, dtype=DTYPE)

    print("[cold-start] 首次 forward...")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(x)
    torch.cuda.synchronize()
    cold = time.perf_counter() - t0
    print(f"[cold-start] {cold:.1f}s out={tuple(out.shape)} dtype={out.dtype}")

    torch.cuda.reset_peak_memory_stats()
    for _ in range(WARMUP):
        with torch.no_grad():
            model(x)
    torch.cuda.synchronize()

    print(f"[measure] {MEASURE_ITERS} iters × BATCH={BATCH}")
    per_iter = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(MEASURE_ITERS):
        ti = time.perf_counter()
        with torch.no_grad():
            model(x)
        torch.cuda.synchronize()
        per_iter.append((time.perf_counter() - ti) * 1000)
    total_s = time.perf_counter() - t0

    peak = torch.cuda.max_memory_allocated() / 2**30
    n = MEASURE_ITERS * BATCH
    mean = statistics.mean(per_iter)
    per_img = mean / BATCH
    print(f"\n========================================")
    print(f"  MODE = ConvBiasIF-bf16-eager-hard  BATCH={BATCH}")
    print(f"  per-img = {per_img:.5f} ms  | mean iter = {mean:.4f} ms")
    print(f"  throughput = {n / total_s:.2f} 张/秒")
    print(f"  peak mem = {peak:.2f} GiB")
    print(f"  cold = {cold:.1f} s")
    print(f"========================================")

    with open("/tmp/cold_start_results.jsonl", "a") as f:
        f.write(json.dumps({
            "mode": "ConvBiasIF-bf16-eager-hard",
            "batch": BATCH, "iters": MEASURE_ITERS, "n_samples": n,
            "total_s": total_s, "mean_iter_ms": mean,
            "mean_per_img_ms": per_img,
            "throughput_imgs_per_s": n / total_s,
            "peak_mem_gib": peak, "compile_s": cold,
        }) + "\n")


if __name__ == "__main__":
    main()
