"""prefix_sum_snn.py — 前缀和形式的 IF 神经元 VGG16-SNN 推理基准。

算法：用户提议的"prefix-sum + 顺序阈值比对 + reset"重写 LIF 为 IF：

  对每一层 forward(x:[T,B,...]):
    1) 用 cumsum(x, dim=0) 一次性算出每时刻累积输入  ── 这是 prefix-sum 部分
    2) 顺序 t=0..T-1 比对阈值；超过即发 spike，并把累积 reset bias +threshold
    3) v[t] = cumsum(x)[t] - threshold * (spike_count_to_t)
       spike[t] = (v[t] >= threshold) ? 1 : 0

  注：用 soft-reset（spike 后减一份 threshold）—— 这是 prefix-sum 形式自然支持的语义；
      hard-reset（spike 后 v=0）在 prefix-sum 下需要追踪"上次 reset 时的 cumsum 值"，
      和 soft-reset 不等价但实现复杂度类似。本文采用 soft-reset 与用户描述对齐。

网络结构：
  - 与 NIR 路径等价：13 Conv (无 BN, 已 fold) + 5 AvgPool + 3 FC + 15 IF
  - 时间步 T=4
  - 输入 [T, B, 3, 224, 224]
  - 全 Triton 编译路径 (torch.compile + max_autotune + force_disable_caches)

测量协议（与 cold_start_10k_compare.py 对齐，便于跨路径比较）：
  - 冷启动 ~/.triton/cache 一次性编译
  - WARMUP=5 次预热
  - MEASURE=ceil(10000/BATCH) 次 forward，每次单独计时

输出：
  - 单次 forward 平均延迟
  - 单张折算延迟
  - 吞吐 (张/秒)
  - 峰值显存
  - 同步追加到 /tmp/cold_start_results.jsonl 便于与 path B / NIR-compile / SJ-direct 对照
"""
import os
import sys
import json
import time
import statistics
import pathlib

import torch
import torch.nn as nn
import torch._dynamo
import torch._inductor.config as inductor_cfg


# ------------------------------- 配置 -------------------------------
BATCH = int(os.environ.get("BATCH", 56))
TOTAL_SAMPLES = int(os.environ.get("TOTAL_SAMPLES", 10000))
WARMUP = int(os.environ.get("WARMUP", 5))
MODE = os.environ.get("MODE", "compile").lower()   # "compile" | "eager"
RESET = os.environ.get("RESET", "soft").lower()    # "soft" | "hard"
SEED = 42
T = 4
NUM_CLASSES = 1000
DEVICE = torch.device("cuda")
MEASURE_ITERS = (TOTAL_SAMPLES + BATCH - 1) // BATCH

if MODE not in ("compile", "eager"):
    raise ValueError(f"MODE must be 'compile' or 'eager', got {MODE!r}")
if RESET not in ("soft", "hard"):
    raise ValueError(f"RESET must be 'soft' or 'hard', got {RESET!r}")

# 全 Triton 编译配置 —— 仅 compile 模式生效
if MODE == "compile":
    torch._dynamo.config.recompile_limit = 256
    torch._dynamo.config.cache_size_limit = 256
    inductor_cfg.max_autotune = True
    inductor_cfg.max_autotune_gemm_backends = "TRITON"
    inductor_cfg.max_autotune_conv_backends = "TRITON"
    inductor_cfg.force_disable_caches = True

print(f"[config] MODE={MODE}  RESET={RESET}  BATCH={BATCH}  T={T}  "
      f"MEASURE_ITERS={MEASURE_ITERS}  WARMUP={WARMUP}  "
      f"目标样本 ≈ {MEASURE_ITERS * BATCH} (≥{TOTAL_SAMPLES})")


# ------------------------------- 前缀和 IF 神经元 -------------------------------
class PrefixSumIFNode(nn.Module):
    """Integrate-and-Fire 神经元，按 prefix-sum 形式重写。

    forward(x[T,B,...]):
        cum = cumsum(x, dim=0)              # [T,B,...]  prefix-sum，可并行
        spike_count = zeros_like(x[0])
        for t in range(T):                  # 顺序：T=4 在 dynamo 下被完全展开
            v_t = cum[t] - threshold * spike_count
            spike[t] = (v_t >= threshold).float()
            spike_count += spike[t]
        return stack(spike, dim=0)          # [T,B,...]

    与传统 LIF（带 decay、hard-reset）相比：
      - 无 decay：tau = inf 的极限，等价于 IF
      - soft-reset：spike 后 v -= threshold（excess 累计到下一步），等价于"减阈值"重置
      - 数学上：v_t = cumsum(x)[t] - threshold * (累积 spike 数)，无需 fold v 进 state

    设计意图：把 LIF 时间维上的因果递推（v_t = decay*v_{t-1} + x_t）改写为
    "可分离的线性 prefix-sum 部分 + 局部的非线性阈值检查"。理论上 prefix-sum
    部分可走 hardware-optimized parallel scan (log T 深度)，T 维的非线性
    检查只剩 T 步顺序工作 —— 对大 T 有潜在加速。
    """

    def __init__(self, v_threshold: float = 1.0):
        super().__init__()
        self.v_threshold = v_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, ...]
        cum = torch.cumsum(x, dim=0)
        spike_count = torch.zeros_like(x[0])
        spikes = []
        for t in range(x.shape[0]):
            v_t = cum[t] - self.v_threshold * spike_count
            spike_t = (v_t >= self.v_threshold).to(x.dtype)
            spikes.append(spike_t)
            spike_count = spike_count + spike_t
        return torch.stack(spikes, dim=0)


class PrefixSumHardResetIFNode(nn.Module):
    """Hard-reset IF 神经元（v 触发后重置为 0），用 prefix-sum 形式重写。

    forward(x[T,B,...]):
        cum = cumsum(x, dim=0)
        last_cum_at_spike = zeros_like(x[0])   # 上一次 spike 时的 cumsum 值（"reset baseline"）
        for t in range(T):
            v_t = cum[t] - last_cum_at_spike   # 自上次 reset 以来的累积输入
            spike[t] = (v_t >= threshold) ? 1 : 0
            last_cum_at_spike = where(spike[t], cum[t], last_cum_at_spike)
                                # 触发时更新 baseline 为当前 cumsum，下一步起算
        return stack(spike, dim=0)

    数学等价于标准 hard-reset IF（v_t ← 0 after spike，且 v_t = v_{t-1} + x_t before spike）：

        v[t] = cum[t] - cum[last_spike_time]   for t > last_spike_time
        spike[t] = (v[t] >= threshold)

    与 soft-reset 的唯一差别：reset 后丢弃 overshoot（即超过 threshold 的部分），
    soft-reset 把 overshoot 累加进下一步。
    """

    def __init__(self, v_threshold: float = 1.0):
        super().__init__()
        self.v_threshold = v_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cum = torch.cumsum(x, dim=0)
        last_cum_at_spike = torch.zeros_like(x[0])
        spikes = []
        for t in range(x.shape[0]):
            v_t = cum[t] - last_cum_at_spike
            spike_t = (v_t >= self.v_threshold).to(x.dtype)
            spikes.append(spike_t)
            # 触发时 baseline 更新为当前 cum[t]，否则保持原 baseline
            last_cum_at_spike = torch.where(spike_t > 0, cum[t], last_cum_at_spike)
        return torch.stack(spikes, dim=0)


# ------------------------------- 无状态层多步包装 -------------------------------
class TimeBatchWrapper(nn.Module):
    """把 [T, B, ...] 输入 flatten 成 [T·B, ...] 让 nn.Conv2d / nn.Linear / nn.AvgPool2d / nn.Flatten 接收。

    与 SJ layer.Conv2d 的 step_mode='m' 多步包装等价（详见
    spikingjelly/.../layer/stateless_wrapper.py:176-190）。直接用原生 nn.* + 此包装，
    既保留 Inductor 全图 trace 友好性，也避免引入 SJ 的 step_mode 状态机。
    """

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B = x.shape[0], x.shape[1]
        y = self.layer(x.flatten(0, 1))
        return y.view(T, B, *y.shape[1:])


# ------------------------------- VGG16 配置 -------------------------------
VGG16_CFG = [
    64, 64, "P",
    128, 128, "P",
    256, 256, 256, "P",
    512, 512, 512, "P",
    512, 512, 512, "P",
]


def build_prefix_sum_vgg16(num_classes: int = NUM_CLASSES) -> nn.Module:
    """VGG16-SNN with PrefixSum{Soft|Hard}ResetIFNode + AvgPool (no BN, no MaxPool).

    神经元按 RESET 环境变量选择 soft 还是 hard。
    """
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    Neuron = PrefixSumIFNode if RESET == "soft" else PrefixSumHardResetIFNode

    feats, in_ch = [], 3
    for v in VGG16_CFG:
        if v == "P":
            feats.append(TimeBatchWrapper(nn.AvgPool2d(kernel_size=2, stride=2)))
        else:
            feats.append(TimeBatchWrapper(nn.Conv2d(in_ch, v, kernel_size=3, padding=1)))
            feats.append(Neuron(v_threshold=1.0))
            in_ch = v
    classifier = nn.Sequential(
        TimeBatchWrapper(nn.Flatten()),
        TimeBatchWrapper(nn.Linear(512 * 7 * 7, 4096)),
        Neuron(v_threshold=1.0),
        TimeBatchWrapper(nn.Linear(4096, 4096)),
        Neuron(v_threshold=1.0),
        TimeBatchWrapper(nn.Linear(4096, num_classes)),
    )
    model = nn.Sequential(nn.Sequential(*feats), classifier)
    return model.eval().cuda()


# ------------------------------- 实测 -------------------------------
def main():
    print(f"\n[build] PrefixSum-IF VGG16-SNN: "
          f"13 Conv + 15 PrefixSumIFNode + 5 AvgPool + 3 Linear")
    model = build_prefix_sum_vgg16(NUM_CLASSES)
    print(f"  params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    if MODE == "compile":
        runnable = torch.compile(model)
    else:
        runnable = model
    x = torch.randn(T, BATCH, 3, 224, 224, device=DEVICE, generator=None)
    print(f"  input shape: {tuple(x.shape)}")

    # ---- 冷启动：compile 触发 dynamo + Inductor + Triton autotune；eager 只是 warmup 卷积 ----
    label = "compile + autotune（一次性，60-150s）" if MODE == "compile" else "首次 forward（cuDNN heuristic warmup）"
    print(f"\n[cold-start] {label} ...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = runnable(x)
    torch.cuda.synchronize()
    cold_s = time.perf_counter() - t0
    print(f"[cold-start] {cold_s:.1f}s   out shape={tuple(out.shape)}")

    torch.cuda.reset_peak_memory_stats()

    # ---- Warmup ----
    print(f"\n[warmup] {WARMUP} iters ...")
    for _ in range(WARMUP):
        with torch.no_grad():
            runnable(x)
    torch.cuda.synchronize()

    # ---- Measure ----
    print(f"[measure] {MEASURE_ITERS} iters × BATCH={BATCH} "
          f"= {MEASURE_ITERS * BATCH} samples ...")
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
    print(f"  MODE = PrefixSumIF-{MODE}-{RESET}    BATCH = {BATCH}    cold-start = YES")
    print(f"  forward 调用次数 = {MEASURE_ITERS}     总样本 = {n_samples}")
    print(f"  总耗时           = {total_s:.4f} s")
    print(f"  每次 forward     = avg {mean_iter:.4f} ms | median {median_iter:.4f} "
          f"| std {std_iter:.4f} | min {min(per_iter_ms):.4f} | max {max(per_iter_ms):.4f}")
    print(f"  单张折算         = avg {per_img:.5f} ms / 张  | median {median_iter / BATCH:.5f}")
    print(f"  吞吐             = {n_samples / total_s:.2f} 张/秒")
    print(f"  GPU peak memory  = {peak_mem:.2f} GiB")
    print(f"  cold compile     = {cold_s:.1f} s")
    print("=" * 78)

    # ---- 落 JSON 行供跨路径聚合 ----
    result = {
        "mode": f"PrefixSumIF-{MODE}-{RESET}",
        "batch": BATCH,
        "iters": MEASURE_ITERS,
        "n_samples": n_samples,
        "total_s": total_s,
        "mean_iter_ms": mean_iter,
        "median_iter_ms": median_iter,
        "std_iter_ms": std_iter,
        "min_iter_ms": min(per_iter_ms),
        "max_iter_ms": max(per_iter_ms),
        "mean_per_img_ms": per_img,
        "throughput_imgs_per_s": n_samples / total_s,
        "peak_mem_gib": peak_mem,
        "compile_s": cold_s,
    }
    with open("/tmp/cold_start_results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"\n[stat] result appended to /tmp/cold_start_results.jsonl")


if __name__ == "__main__":
    main()
