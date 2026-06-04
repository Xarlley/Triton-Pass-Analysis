"""端到端 VGG16-SNN benchmark：把整个网络用 snn_compiler 重新组装并测速。

对比对象：
  1. SpikingJelly multistep LIFNode + Inductor compile（原始 fused-bias-IF 之前的 baseline）
  2. SpikingJelly LIFNode 朴素 eager （理论 ceiling）
  3. snn_compiler FusedConvBN[+LIF]Node eager（本框架）
  4. snn_compiler 但 IFNode + 软复位（最常用 SNN 推理配置）

所有模型共享同一份随机权重，确保对比公平。

环境变量：
  BATCH=32  T=4  TOTAL=2000  WARMUP=5  MODE=bf16   (fp32 / bf16)
  LAYOUT=NCHW   (NCHW / NHWC)
  NEURON=if     (if / lif)
  RESET=hard    (soft / hard)
"""
import os, sys, time, statistics, json, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
import torch.nn as nn
import torch.nn.functional as F

from snn_compiler.nn.modules import IFNode, LIFNode, FusedConvBNNeuron, FusedLinearNeuron


BATCH    = int(os.environ.get("BATCH", 32))
T        = int(os.environ.get("T", 4))
TOTAL    = int(os.environ.get("TOTAL", 2000))
WARMUP   = int(os.environ.get("WARMUP", 5))
DTYPE    = {"fp32": torch.float32, "bf16": torch.bfloat16}[os.environ.get("MODE", "fp32")]
LAYOUT   = os.environ.get("LAYOUT", "NCHW")
NEURON   = os.environ.get("NEURON", "if")
RESET    = os.environ.get("RESET", "hard")
ITERS    = (TOTAL + BATCH - 1) // BATCH

print(f"[config] BATCH={BATCH} T={T} DTYPE={DTYPE} LAYOUT={LAYOUT} "
      f"NEURON={NEURON} RESET={RESET} ITERS={ITERS} TOTAL≈{ITERS*BATCH}")

VGG16_CFG = [64, 64, "P", 128, 128, "P", 256, 256, 256, "P",
             512, 512, 512, "P", 512, 512, 512, "P"]


def make_neuron():
    soft = (RESET == "soft")
    if NEURON == "if":
        return IFNode(soft_reset=soft, v_threshold=1.0, v_reset=0.0, layout=LAYOUT)
    return LIFNode(tau=2.0, decay_input=True, soft_reset=soft,
                    v_threshold=1.0, v_reset=0.0, layout=LAYOUT)


# ============================================================
#   v1: 朴素 PyTorch nn 模型（无融合）
# ============================================================
class NaiveVGG16SNN(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        feats, in_ch = [], 3
        for v in VGG16_CFG:
            if v == "P":
                feats.append(nn.AvgPool2d(2, 2))
            else:
                feats.extend([
                    nn.Conv2d(in_ch, v, 3, padding=1),
                    nn.BatchNorm2d(v),
                    make_neuron(),
                ])
                in_ch = v
        self.features = nn.Sequential(*feats)
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.if1 = make_neuron()
        self.fc2 = nn.Linear(4096, 4096)
        self.if2 = make_neuron()
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x_seq):
        # x_seq: [T, B, 3, H, W]
        T_, B = x_seq.shape[0], x_seq.shape[1]
        x = x_seq.reshape(T_ * B, *x_seq.shape[2:])
        # 朴素跑 features：依次过每个 layer，遇到 neuron 时 view → forward → reshape
        for layer in self.features:
            if isinstance(layer, (IFNode, LIFNode)):
                x = x.view(T_, B, *x.shape[1:])
                x = layer(x)
                x = x.reshape(T_ * B, *x.shape[2:])
            else:
                x = layer(x)
        x = x.flatten(1)
        x = self.fc1(x); x = self.if1(x.view(T_, B, -1)).reshape(T_ * B, -1)
        x = self.fc2(x); x = self.if2(x.view(T_, B, -1)).reshape(T_ * B, -1)
        x = self.fc3(x)
        return x.view(T_, B, -1)


# ============================================================
#   v2: 全融合（手工）— 用 FusedConvBNNeuron + FusedLinearNeuron
# ============================================================
def build_fused_vgg16_from_naive(naive: NaiveVGG16SNN) -> nn.Module:
    """从一个 NaiveVGG16SNN（已加载权重 & BN running stats）构造融合版。"""
    feats = []
    in_ch = 3
    # 顺序解析：从 naive.features 中拿对应的 Conv2d/BN/Neuron 三元组
    layers = list(naive.features.children())
    i = 0
    while i < len(layers):
        a = layers[i]
        if isinstance(a, nn.AvgPool2d):
            feats.append(a)
            i += 1
            continue
        # 期望 Conv → BN → Neuron
        assert isinstance(a, nn.Conv2d), f"unexpected {type(a)} at {i}"
        assert isinstance(layers[i+1], nn.BatchNorm2d)
        assert isinstance(layers[i+2], (IFNode, LIFNode))
        kw = dict(soft_reset=layers[i+2].soft_reset,
                  v_threshold=layers[i+2].v_threshold,
                  v_reset=layers[i+2].v_reset, layout=LAYOUT)
        if isinstance(layers[i+2], LIFNode):
            kw["neuron"] = "lif"; kw["tau"] = layers[i+2].tau
            kw["decay"] = layers[i+2].decay
            kw["decay_input"] = layers[i+2].decay_input
        else:
            kw["neuron"] = "if"; kw["decay"] = layers[i+2].decay
        mod = FusedConvBNNeuron(a.eval(), layers[i+1].eval(), **kw)
        feats.append(mod)
        i += 3

    feats_seq = nn.Sequential(*feats)

    fc1_kw = dict(soft_reset=naive.if1.soft_reset, v_threshold=naive.if1.v_threshold,
                  v_reset=naive.if1.v_reset)
    if isinstance(naive.if1, LIFNode):
        fc1_kw.update(neuron="lif", tau=naive.if1.tau, decay=naive.if1.decay,
                      decay_input=naive.if1.decay_input)
    else:
        fc1_kw.update(neuron="if", decay=naive.if1.decay)
    fc1_fused = FusedLinearNeuron(512*7*7, 4096, **fc1_kw)
    with torch.no_grad():
        fc1_fused.weight.copy_(naive.fc1.weight)
        fc1_fused.bias.copy_(naive.fc1.bias)

    fc2_kw = dict(soft_reset=naive.if2.soft_reset, v_threshold=naive.if2.v_threshold,
                  v_reset=naive.if2.v_reset)
    if isinstance(naive.if2, LIFNode):
        fc2_kw.update(neuron="lif", tau=naive.if2.tau, decay=naive.if2.decay,
                      decay_input=naive.if2.decay_input)
    else:
        fc2_kw.update(neuron="if", decay=naive.if2.decay)
    fc2_fused = FusedLinearNeuron(4096, 4096, **fc2_kw)
    with torch.no_grad():
        fc2_fused.weight.copy_(naive.fc2.weight)
        fc2_fused.bias.copy_(naive.fc2.bias)

    class FusedVGG16(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = feats_seq
            self.fc1 = fc1_fused
            self.fc2 = fc2_fused
            self.fc3 = nn.Linear(4096, 1000)
            with torch.no_grad():
                self.fc3.weight.copy_(naive.fc3.weight)
                self.fc3.bias.copy_(naive.fc3.bias)

        def forward(self, x_seq):
            T_, B = x_seq.shape[0], x_seq.shape[1]
            x = x_seq
            for layer in self.features:
                if isinstance(layer, nn.AvgPool2d):
                    x = layer(x.reshape(T_ * B, *x.shape[2:]))
                    x = x.view(T_, B, *x.shape[1:])
                else:
                    x = layer(x)
            x = x.contiguous().reshape(T_, B, -1)
            x = self.fc1(x)
            x = self.fc2(x)
            x_flat = x.reshape(T_ * B, -1)
            out = self.fc3(x_flat)
            return out.view(T_, B, -1)

    return FusedVGG16()


def measure(model, x, name):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(WARMUP):
        with torch.no_grad():
            model(x)
    torch.cuda.synchronize()
    per_iter = []
    t_total = time.perf_counter()
    for _ in range(ITERS):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x)
        torch.cuda.synchronize()
        per_iter.append((time.perf_counter() - t0) * 1000)
    total_s = time.perf_counter() - t_total
    peak = torch.cuda.max_memory_allocated() / 2**30
    mean = statistics.mean(per_iter)
    med = statistics.median(per_iter)
    per_img = mean / BATCH
    print(f"\n  {name}")
    print(f"     iter {mean:8.4f} ms (med {med:.4f}, min {min(per_iter):.4f})  "
          f"per-img {per_img:7.4f} ms   peak {peak:.2f} GiB   "
          f"throughput {BATCH*ITERS/total_s:.1f} img/s")
    return dict(name=name, mean_ms=mean, median_ms=med, min_ms=min(per_iter),
                per_img_ms=per_img, peak_gib=peak,
                throughput=BATCH * ITERS / total_s)


def main():
    torch.manual_seed(0)
    naive = NaiveVGG16SNN(1000).eval().cuda()
    # 给 BN 有意义的 stats（评估场景）— 用一次前向更新会破坏 eval；这里直接随机
    for mod in naive.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.running_mean.copy_(torch.randn_like(mod.running_mean) * 0.1)
            mod.running_var.copy_(torch.rand_like(mod.running_var) + 0.5)

    if DTYPE == torch.bfloat16:
        naive = naive.to(DTYPE)
    if LAYOUT == "NHWC":
        for mod in naive.modules():
            if isinstance(mod, nn.Conv2d):
                mod.weight.data = mod.weight.data.to(memory_format=torch.channels_last)

    x = torch.randn(T, BATCH, 3, 224, 224, device="cuda", dtype=DTYPE)

    # naive baseline
    res_naive = measure(naive, x, name=f"Naive   (Conv+BN+{NEURON.upper()})")

    # fused via framework
    fused = build_fused_vgg16_from_naive(naive).eval().cuda()
    if DTYPE == torch.bfloat16:
        fused = fused.to(DTYPE)
    res_fused = measure(fused, x, name=f"Fused   (snn_compiler ConvBN-{NEURON.upper()})")

    speedup = res_naive["mean_ms"] / res_fused["mean_ms"]
    mem_saved = res_naive["peak_gib"] - res_fused["peak_gib"]
    print(f"\n  Speedup: {speedup:.3f}x   peak-mem saved {mem_saved:+.2f} GiB")

    out_path = pathlib.Path(__file__).resolve().parent / "results.jsonl"
    with open(out_path, "a") as f:
        for r in [res_naive, res_fused]:
            r["config"] = dict(batch=BATCH, T=T, dtype=str(DTYPE), layout=LAYOUT,
                                neuron=NEURON, reset=RESET)
            f.write(json.dumps(r) + "\n")
    print(f"\n[results appended to {out_path}]")


if __name__ == "__main__":
    main()
