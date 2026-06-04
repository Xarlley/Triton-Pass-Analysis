"""NHWC (channels_last) 全程的 VGG16-SNN：消除 cuDNN 的 layout xform 开销。

模型内部所有 conv 输入/输出都是 4D [T·B, C, H, W] channels_last 内存布局，
跳过 5D view（直接在 4D 上完成 T 维语义）。IF kernel 用 fused_bias_if_nhwc 处理
channels_last 内存。

预期：消除 ~15 ms / forward (BATCH=32) 的 nchwToNhwcKernel/nhwcToNchwKernel 开销，
对应 ~0.5 ms / 张 (BATCH=56) 节省。
"""
import os, sys, json, time, statistics, pathlib
HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "explore"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from fused_if_kernel import fused_if, _fused_if_forward_kernel
from fused_bias_if_nhwc_kernel import _fused_bias_if_nhwc_kernel

import triton

BATCH = int(os.environ.get("BATCH", 56))
TOTAL_SAMPLES = int(os.environ.get("TOTAL_SAMPLES", 10000))
WARMUP = int(os.environ.get("WARMUP", 5))
SEED = 42
T = 4
DEVICE = torch.device("cuda")
MEASURE_ITERS = (TOTAL_SAMPLES + BATCH - 1) // BATCH


class ConvBiasIFNHWCNode(nn.Module):
    """Conv2d (channels_last) + FusedBiasIF (NHWC kernel) 全 4D。

    forward(x_4d_cl: [T·B, C_in, H_in, W_in] channels_last)
        → spike_4d_cl: [T·B, C_out, H_out, W_out] channels_last
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 v_threshold=1.0, soft_reset=False):
        super().__init__()
        c = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=True)
        # weight 转 channels_last
        self.weight = nn.Parameter(c.weight.detach().to(memory_format=torch.channels_last))
        self.bias = c.bias
        self.padding = c.padding
        self.stride = c.stride
        self.dilation = c.dilation
        self.groups = c.groups
        self.v_threshold = v_threshold
        self.soft_reset = soft_reset
        self.out_channels = out_channels

    def forward(self, x_4d_cl: torch.Tensor) -> torch.Tensor:
        """x_4d_cl: [T·B, C_in, H, W] in channels_last memory.
        Returns: [T·B, C_out, H_out, W_out] in channels_last memory.
        """
        # conv (cuDNN 看到 channels_last 输入会选 NHWC 算法，无 layout xform)
        y = F.conv2d(x_4d_cl, self.weight, bias=None,
                     stride=self.stride, padding=self.padding,
                     dilation=self.dilation, groups=self.groups)
        # y 是 channels_last 4D
        # FusedBiasIF NHWC kernel：T 是常量，从 y 的 batch 维（T*B）反推
        TB, C, H, W = y.shape
        # 我们知道 T=4 是网络外层时间步数
        B = TB // T
        NCL = B * C * H * W
        assert TB == T * B
        spike = torch.empty_like(y)   # 同 channels_last 内存
        grid = lambda meta: (triton.cdiv(NCL, meta["BLOCK_NCL"]),)
        _fused_bias_if_nhwc_kernel[grid](
            y, self.bias, spike,
            T=T, NCL=NCL, C=C,
            v_threshold=self.v_threshold,
            SOFT_RESET=self.soft_reset,
        )
        return spike


class FusedIFNHWCNode(nn.Module):
    """Plain IF without bias add, also using NHWC kernel for FC输出 [T·B, C]."""
    def __init__(self, v_threshold=1.0, soft_reset=False):
        super().__init__()
        self.v_threshold = v_threshold
        self.soft_reset = soft_reset

    def forward(self, x):
        # x: [T·B, C] from Linear, no broadcast issue
        # call fused_if kernel with NCL = numel per timestep
        TB = x.shape[0]
        B = TB // T
        NCL = x.numel() // T
        spike = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(NCL, meta["BLOCK_NCL"]),)
        _fused_if_forward_kernel[grid](
            x.contiguous(), spike,
            T=T, NCL=NCL,
            v_threshold=self.v_threshold,
            SOFT_RESET=self.soft_reset,
        )
        return spike


VGG16_CFG = [64, 64, "P", 128, 128, "P", 256, 256, 256, "P",
             512, 512, 512, "P", 512, 512, 512, "P"]


class NHWC_VGG16SNN(nn.Module):
    """完全 4D channels_last 的 VGG16-SNN。"""
    def __init__(self, num_classes=1000, soft_reset=False):
        super().__init__()
        feats, in_ch = [], 3
        for v in VGG16_CFG:
            if v == "P":
                feats.append(nn.AvgPool2d(2, 2))
            else:
                feats.append(ConvBiasIFNHWCNode(in_ch, v, 3, padding=1,
                                                v_threshold=1.0, soft_reset=soft_reset))
                in_ch = v
        self.features = nn.Sequential(*feats)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512*7*7, 4096)
        self.if1 = FusedIFNHWCNode(v_threshold=1.0, soft_reset=soft_reset)
        self.fc2 = nn.Linear(4096, 4096)
        self.if2 = FusedIFNHWCNode(v_threshold=1.0, soft_reset=soft_reset)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x_5d):
        """x_5d: [T, B, 3, 224, 224]"""
        T_, B = x_5d.shape[0], x_5d.shape[1]
        # flatten to 4D, set channels_last
        x = x_5d.flatten(0, 1).to(memory_format=torch.channels_last)  # [T*B, 3, 224, 224] cl
        # features: 全 4D channels_last
        x = self.features(x)                                          # [T*B, 512, 7, 7] cl
        # back to NCHW for Flatten/Linear
        x = x.contiguous()                                            # 强制 NCHW
        x = self.flatten(x)                                           # [T*B, 25088]
        x = self.fc1(x); x = self.if1(x)
        x = self.fc2(x); x = self.if2(x)
        x = self.fc3(x)
        # view back to 5D
        return x.view(T_, B, *x.shape[1:])


def main():
    print(f"[config] BATCH={BATCH} T={T} MEASURE_ITERS={MEASURE_ITERS}")
    torch.manual_seed(SEED)
    model = NHWC_VGG16SNN(1000, soft_reset=False).eval().cuda()
    x = torch.randn(T, BATCH, 3, 224, 224, device=DEVICE)
    print("[cold-start] 首次 forward...")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(x)
    torch.cuda.synchronize()
    cold = time.perf_counter() - t0
    print(f"[cold-start] {cold:.1f}s out={tuple(out.shape)}")

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
    std = statistics.stdev(per_iter)
    per_img = mean / BATCH

    print("\n" + "=" * 78)
    print(f"  MODE = NHWC-ConvBiasIF-eager-hard    BATCH = {BATCH}")
    print(f"  forward avg = {mean:.4f} ms | std = {std:.4f}")
    print(f"  单张折算 = {per_img:.5f} ms / 张")
    print(f"  吞吐 = {n / total_s:.2f} 张/秒")
    print(f"  peak mem = {peak:.2f} GiB")
    print(f"  cold = {cold:.1f} s")
    print("=" * 78)

    with open("/tmp/cold_start_results.jsonl", "a") as f:
        f.write(json.dumps({
            "mode": f"NHWC-ConvBiasIF-eager-hard", "batch": BATCH,
            "iters": MEASURE_ITERS, "n_samples": n,
            "total_s": total_s, "mean_iter_ms": mean, "std_iter_ms": std,
            "mean_per_img_ms": per_img,
            "throughput_imgs_per_s": n / total_s,
            "peak_mem_gib": peak, "compile_s": cold,
        }) + "\n")


if __name__ == "__main__":
    main()
