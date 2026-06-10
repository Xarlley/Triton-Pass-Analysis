"""验证现象：较大网络 + 较多时间步 T → 全 T 融合（triton 高性能）路径的激活 [T,B,...] 显存 ∝ T
→ 在 16GiB 卡上 OOM → 只能回落到「按 chunk / 逐时间步的 python 循环」低显存路径才能跑通，
但该回落路径无法享受全 T 融合的高性能（更慢）。

本实验是 snn_compiler 之外的独立实验代码（不修改、不写入 snn_compiler 开源项目目录），
只把 snn_compiler 当库调用其**公开 API**：
  - `fused_bias_if_lif`          —— 全 T 融合 LIF（一次处理整段 [T,B,...]）
  - `fused_bias_if_lif_stateful` —— 带 v_init/v_final 的有状态 LIF（供 chunked/逐步 python 循环串接膜电位）
  - `snn_compiler.zoo.vgg16_snn` —— 整网 baseline

本机：RTX 5070 Ti（16 GiB），是复现该现象的标准硬件（snn_compiler 的 large_T 基准即在此卡）。

Part A：整网 VGG-16 SNN（融合 + bf16 + NHWC）扫 T → 峰值显存与 OOM 点（"larger net + more T → OOM"）。
Part B：自建多层 conv-bn-LIF 栈，同一计算两条执行路径：
        - full-T 融合：每层持 [T,B,C,H,W]，峰值 ∝ T → 大 T OOM；
        - chunked（逐 chunk python 循环 + 每层膜电位状态）：每层只持 [chunk,B,C,H,W]，峰值 ∝ chunk → 存活但更慢。
        记录峰值显存随 T 的标度、存活性、速度、两路逐位一致。
跑法：python experiments/large-T-oom-fallback/oom_fallback_demo.py
"""
import sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))   # 仓库根，便于 `import snn_compiler`

import torch
import torch.nn as nn
import torch.nn.functional as F

from snn_compiler.kernels.fused import fused_bias_if_lif, fused_bias_if_lif_stateful

dev = "cuda"
GiB = 1024 ** 3
TOTAL_GiB = torch.cuda.get_device_properties(0).total_memory / GiB


def _peak_gib():
    return torch.cuda.max_memory_allocated() / GiB


def _reset():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _bench(fn, warmup=1, iters=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters


# ============================================================
#   Part A：整网融合路径扫 T —— 峰值显存 ∝ T 与 OOM
# ============================================================
def part_A():
    from snn_compiler.zoo import vgg16_snn
    print("\n" + "=" * 80)
    print(f"Part A — 整网 VGG-16 SNN（融合, bf16, NHWC）扫 T；显卡 {torch.cuda.get_device_name(0)} {TOTAL_GiB:.1f} GiB")
    print("=" * 80)
    B = 8
    model = vgg16_snn(num_classes=1000, neuron="lif", tau=2.0, soft_reset=False,
                      layout="NHWC", fused=True).cuda().eval().to(torch.bfloat16)
    wgib = sum(p.numel() * p.element_size() for p in model.parameters()) / GiB
    print(f"模型权重常驻显存 = {wgib:.3f} GiB（与 T 无关）；BATCH={B}")
    print(f"{'T':>5} | {'状态':>5} | {'峰值显存(GiB)':>13} | {'每图(ms)':>9} | 备注")
    prev_peak, prev_T = None, None
    for T in (4, 16, 32, 64, 128, 192, 256, 384):
        _reset()
        try:
            x = torch.randn(T, B, 3, 224, 224, device=dev, dtype=torch.bfloat16)
            with torch.no_grad():
                for _ in range(2):
                    y = model(x)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                y = model(x)
                torch.cuda.synchronize(); dt = (time.perf_counter() - t0) * 1e3
            peak = _peak_gib()
            note = "" if prev_peak is None else f"Δpeak/ΔT≈{(peak-prev_peak)/(T-prev_T)*1024:.0f} MiB/步"
            print(f"{T:>5} | {'OK':>5} | {peak:>13.2f} | {dt/B:>9.2f} | {note}")
            prev_peak, prev_T = peak, T
            del x, y
        except torch.cuda.OutOfMemoryError:
            print(f"{T:>5} | {'OOM':>5} | {'> '+f'{TOTAL_GiB:.0f}':>12} | {'—':>9} | 全 T 融合路径 OOM（激活 [T,B,...] ∝ T）")
            torch.cuda.empty_cache()
            break
    del model
    torch.cuda.empty_cache()


# ============================================================
#   Part B：多层栈 full-T vs chunked —— 显存律 + 存活 + 速度 + 正确性
# ============================================================
class ConvSNNStack(nn.Module):
    """conv-bn-LIF 多层栈 + 全局池化 + 分类头（输出 [T,B,num_classes] 很小）。
    大空间分辨率使中间激活 [T,B,C,H,W] 主导显存。两条前向共享同一权重。"""
    def __init__(self, in_ch=3, num_classes=100):
        super().__init__()
        spec = [(in_ch, 64, False), (64, 64, True), (64, 128, False),
                (128, 128, True), (128, 256, True)]
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.pool = []
        c = in_ch
        for (i, o, p) in spec:
            self.convs.append(nn.Conv2d(c, o, 3, padding=1, bias=False))
            bn = nn.BatchNorm2d(o)
            bn.running_mean.normal_(0, 0.1); bn.running_var.uniform_(0.5, 1.0)
            bn.weight.data.uniform_(0.5, 1.0); bn.bias.data.normal_(0.2, 0.1)
            self.bns.append(bn)
            self.pool.append(p); c = o
        self.fc = nn.Linear(c, num_classes)
        self.eval()

    def _cb(self, li, x4):
        y = self.bns[li](self.convs[li](x4))
        if self.pool[li]:
            y = F.max_pool2d(y, 2)
        return y

    @torch.no_grad()
    def full_T(self, x):                                   # x:[T,B,3,H,W]，每层持 [T,B,...]
        B = x.shape[1]
        for li in range(len(self.convs)):
            Tc = x.shape[0]
            y = self._cb(li, x.reshape(Tc * B, *x.shape[2:]))
            y = y.view(Tc, B, *y.shape[1:]).contiguous()
            x = fused_bias_if_lif(y, None, neuron="lif", tau=2.0, decay_input=True,
                                  soft_reset=False, v_threshold=0.5, v_reset=0.0)
        Tc = x.shape[0]
        x4 = F.adaptive_avg_pool2d(x.reshape(Tc * B, *x.shape[2:]).float(), 1).flatten(1)
        return self.fc(x4.to(x.dtype)).view(Tc, B, -1)

    @torch.no_grad()
    def chunked(self, gen, T, B, chunk):                   # gen(i,c)->[c,B,3,H,W]，每层只持 [chunk,B,...]
        v = [None] * len(self.convs)
        outs = []
        for i in range(0, T, chunk):
            c = min(chunk, T - i)
            x = gen(i, c)
            for li in range(len(self.convs)):
                cc = x.shape[0]
                y = self._cb(li, x.reshape(cc * B, *x.shape[2:]))
                y = y.view(cc, B, *y.shape[1:]).contiguous()
                x, v[li] = fused_bias_if_lif_stateful(
                    y, None, v_init=v[li], return_v=True, neuron="lif", tau=2.0,
                    decay_input=True, soft_reset=False, v_threshold=0.5, v_reset=0.0)
            cc = x.shape[0]
            x4 = F.adaptive_avg_pool2d(x.reshape(cc * B, *x.shape[2:]).float(), 1).flatten(1)
            outs.append(self.fc(x4.to(x.dtype)).view(cc, B, -1))
        return torch.cat(outs, 0)


def part_B():
    print("\n" + "=" * 80)
    print("Part B — 多层 conv-bn-LIF 栈：full-T 融合 vs chunked（逐 chunk python 循环 + 膜电位状态）")
    print("=" * 80)
    torch.manual_seed(0)
    B, H = 16, 112
    model = ConvSNNStack(in_ch=3, num_classes=100).cuda().eval().to(torch.bfloat16)

    # 正确性：小 T 下 full-T 与 chunked 逐位一致
    Tsmall = 24
    xs = (torch.rand(Tsmall, B, 3, H, H, device=dev) < 0.5).to(torch.bfloat16)
    out_full = model.full_T(xs)
    out_ck = model.chunked(lambda i, c: xs[i:i + c].contiguous(), Tsmall, B, chunk=8)
    bit_eq = torch.equal(out_full, out_ck)
    print(f"正确性(T={Tsmall}, chunk=8)：full-T 与 chunked 逐位一致 = {bit_eq}  "
          f"max|Δ|={(out_full.float()-out_ck.float()).abs().max().item():.3e}")
    del xs, out_full, out_ck
    # 大 T 等价性：在 full-T 仍跑得通的较大 T（=64）上，用**同一输入**再确认逐位一致
    Tbig = 64
    xb = (torch.rand(Tbig, B, 3, H, H, device=dev) < 0.5).to(torch.bfloat16)
    of = model.full_T(xb)
    oc = model.chunked(lambda i, c: xb[i:i + c].contiguous(), Tbig, B, chunk=16)
    print(f"正确性(T={Tbig}, chunk=16, 同一输入)：full-T 与 chunked 逐位一致 = {torch.equal(of, oc)}  "
          f"max|Δ|={(of.float()-oc.float()).abs().max().item():.3e}\n")
    del xb, of, oc; torch.cuda.empty_cache()

    print(f"BATCH={B} H=W={H}；显卡 {TOTAL_GiB:.1f} GiB；chunk_t=16")
    print(f"{'T':>5} | {'full-T峰值':>10} {'full':>5} {'full(ms)':>9} | "
          f"{'chunked峰值':>11} {'ck':>4} {'chunk(ms)':>10} | {'chunk慢':>8}")
    CHUNK = 16
    for T in (16, 32, 64, 128):
        _reset()
        full_peak, full_ms, full_state = None, None, "OOM"
        try:
            x = torch.randn(T, B, 3, H, H, device=dev, dtype=torch.bfloat16)
            model.full_T(x)
            full_ms = _bench(lambda: model.full_T(x))
            full_peak = _peak_gib(); full_state = "OK"
            del x
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
        _reset()
        ck_peak, ck_ms, ck_state = None, None, "OOM"
        try:
            gen = lambda i, c: torch.randn(c, B, 3, H, H, device=dev, dtype=torch.bfloat16)
            model.chunked(gen, T, B, CHUNK)
            ck_ms = _bench(lambda: model.chunked(gen, T, B, CHUNK))
            ck_peak = _peak_gib(); ck_state = "OK"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
        fp = f"{full_peak:.2f}GiB" if full_peak else ">16"
        cp = f"{ck_peak:.2f}GiB" if ck_peak else ">16"
        fm = f"{full_ms:.1f}" if full_ms else "—"
        cm = f"{ck_ms:.1f}" if ck_ms else "—"
        if full_ms and ck_ms:
            ratio = f"{ck_ms/full_ms:.2f}×"
        elif full_state == "OOM" and ck_state == "OK":
            ratio = "仅chunked存活"
        else:
            ratio = "—"
        print(f"{T:>5} | {fp:>10} {full_state:>5} {fm:>9} | {cp:>11} {ck_state:>4} {cm:>10} | {ratio:>8}")
    print(f"\n说明：chunked 峰值显存与 T 基本无关（∝ chunk_t），full-T ∝ T；故大 T 时只有 chunked 能跑通，但更慢。")


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}  total={TOTAL_GiB:.1f} GiB  torch={torch.__version__}")
    part_B()
    part_A()
    print("\nOOM_FALLBACK_DEMO_DONE")


if __name__ == "__main__":
    main()
