"""Phase D: 跨 T 与跨架构的端到端 + 分项基准。

测三组：
1. baseline:       zoo `fused=True`（i64 修复后大 T 可正确运行）
2. rate_classifier: 把 classifier 末尾的 nn.Linear(...→num_classes) 后追加一个
                   RateCodedLIFNode 作为 spike-count 投票头（合理的 rate-coding SNN
                   架构）。这样 rate-coded LIF 真的接到了网络输出，效果可比。
3. lif_kernel:    单独跑等效 shape 的 LIF kernel 时间，用于诊断"LIF 在端到端
                   占比"。

每架构 × T ∈ {4, 16, 64, 128} × B = ${BATCH=4}。
"""
import os, sys, pathlib, time, statistics, json, gc

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
import torch.nn as nn

from snn_compiler.zoo import vgg16_snn, resnet18_snn, resnet34_snn
from snn_compiler.nn import RateCodedLIFNode, LIFNode
from snn_compiler.kernels.fused import fused_bias_if_lif, fused_bias_if_lif_rate


def measure(model, x, *, warm=3, iters=10):
    for _ in range(warm):
        with torch.no_grad():
            model(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    per_iter = []
    for _ in range(iters):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x)
        torch.cuda.synchronize()
        per_iter.append((time.perf_counter() - t0) * 1000)
    return dict(
        mean_ms=statistics.mean(per_iter),
        median_ms=statistics.median(per_iter),
        peak_gib=torch.cuda.max_memory_allocated() / 2**30,
    )


class RateClassifierWrapper(nn.Module):
    """把一个 zoo 模型包成 rate-coded SNN：
       fused_model(x_seq) → [T, B, num_classes] logits
       此 wrapper：把 logits 当 LIF 输入，过 RateCodedLIFNode → [B, num_classes] 投票分数
    """
    def __init__(self, model, *, num_classes=1000, tau=2.0, v_threshold=1.0):
        super().__init__()
        self.model = model
        self.rate_head = RateCodedLIFNode(
            tau=tau, decay_input=True, soft_reset=False,
            v_threshold=v_threshold, v_reset=0.0, layout='NCHW',
        )

    def forward(self, x_seq):
        logits = self.model(x_seq)            # [T, B, num_classes]
        return self.rate_head(logits)          # [B, num_classes] spike count


def measure_lif_kernel_share(arch_name, factory, T, B, INPUT_H=224, layout="NHWC"):
    """单独估算在该 (arch, T, B) 下，所有 LIF kernel 跑一次的累计时间，与端到端
    比较得到"LIF 占比"。
    """
    # 估算手段：拿网络中各 conv 输出 shape，按 NCL 累加；分别跑独立 fused_bias_if_lif
    # 计算耗时，求和。
    torch.cuda.empty_cache(); gc.collect()
    torch.manual_seed(0)
    m = factory(num_classes=1000, neuron='lif', tau=2.0, decay_input=True,
                  soft_reset=False, v_threshold=1.0, v_reset=0.0,
                  layout=layout, fused=True, init_bn=True).cuda().eval().to(torch.bfloat16)
    for mod in m.modules():
        if isinstance(mod, nn.Conv2d):
            mod.weight.data = mod.weight.data.to(memory_format=torch.channels_last)
    # 列出各 FusedConvBNNeuron 的 (out_C, out_H, out_W)
    shapes = []
    x = torch.randn(T, B, 3, INPUT_H, INPUT_H, device='cuda', dtype=torch.bfloat16)
    hooks = []
    def hook(mod, _in, out):
        # out: [T, B, C, H, W]
        shapes.append(tuple(out.shape))
    from snn_compiler.nn import FusedConvBNNeuron, FusedConvBNAddNeuron, FusedLinearNeuron
    for mod in m.modules():
        if isinstance(mod, (FusedConvBNNeuron, FusedConvBNAddNeuron, FusedLinearNeuron)):
            hooks.append(mod.register_forward_hook(hook))
    with torch.no_grad():
        m(x)
    for h in hooks: h.remove()
    # 每个 LIF kernel 等效时间：用 fused_bias_if_lif 直接跑一遍同 shape 的 y → spike
    total_ms = 0.0
    for sh in shapes:
        y = torch.randn(*sh, device='cuda', dtype=torch.bfloat16).contiguous()
        bias_c = y.shape[2] if y.ndim >= 3 else y.shape[-1]
        bias = torch.randn(bias_c, device='cuda')
        # warm + measure
        for _ in range(3):
            fused_bias_if_lif(y, bias, neuron='lif', tau=2.0, decay_input=True,
                                soft_reset=False, v_threshold=1.0, v_reset=0.0, layout=layout)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            fused_bias_if_lif(y, bias, neuron='lif', tau=2.0, decay_input=True,
                                soft_reset=False, v_threshold=1.0, v_reset=0.0, layout=layout)
        torch.cuda.synchronize()
        total_ms += (time.perf_counter() - t0) * 1000 / 5
        del y, bias
    return total_ms, len(shapes)


def run_one(arch_name, factory, T, B, *, INPUT_H=224, layout="NHWC"):
    print(f"\n== {arch_name}  T={T}  BATCH={B}  layout={layout} ==", flush=True)
    torch.cuda.empty_cache(); gc.collect()
    torch.manual_seed(0)
    m = factory(num_classes=1000, neuron='lif', tau=2.0, decay_input=True,
                  soft_reset=False, v_threshold=1.0, v_reset=0.0,
                  layout=layout, fused=True, init_bn=True).cuda().eval().to(torch.bfloat16)
    for mod in m.modules():
        if isinstance(mod, nn.Conv2d):
            mod.weight.data = mod.weight.data.to(memory_format=torch.channels_last)

    x = torch.randn(T, B, 3, INPUT_H, INPUT_H, device='cuda', dtype=torch.bfloat16)

    try:
        r_base = measure(m, x)
        print(f"  baseline:        iter {r_base['mean_ms']:8.3f} ms   "
              f"per-img {r_base['mean_ms']/B:7.4f} ms   peak {r_base['peak_gib']:5.2f} GiB", flush=True)
    except (torch.AcceleratorError, RuntimeError) as e:
        print(f"  baseline:   FAIL  {str(e)[:60]}", flush=True)
        r_base = None

    # rate-coded classifier 包装
    wrapped = RateClassifierWrapper(m, num_classes=1000, tau=2.0, v_threshold=1.0).cuda().eval()
    try:
        r_rate = measure(wrapped, x)
        print(f"  rate_classifier: iter {r_rate['mean_ms']:8.3f} ms   "
              f"per-img {r_rate['mean_ms']/B:7.4f} ms   peak {r_rate['peak_gib']:5.2f} GiB",
              flush=True)
        if r_base is not None:
            delta_ms = r_rate['mean_ms'] - r_base['mean_ms']
            print(f"  → rate-coded head overhead = {delta_ms:+.3f} ms / iter "
                  f"({100*delta_ms/r_base['mean_ms']:+.2f}%)", flush=True)
    except (torch.AcceleratorError, RuntimeError) as e:
        print(f"  rate_classifier: FAIL  {str(e)[:80]}", flush=True)
        r_rate = None

    return r_base, r_rate


def main():
    cases = [
        ('VGG-16 SNN',    vgg16_snn),
        ('ResNet-18 SNN', resnet18_snn),
        ('ResNet-34 SNN', resnet34_snn),
    ]
    Ts = [4, 16, 64, 128]
    B = int(os.environ.get('BATCH', 4))
    results = []
    for name, fac in cases:
        for T in Ts:
            r_base, r_rate = run_one(name, fac, T, B)
            # LIF kernel 占比
            try:
                t_lif, n = measure_lif_kernel_share(name, fac, T, B)
                share = (t_lif / r_base['mean_ms'] * 100) if r_base else None
                print(f"  LIF kernel sum:  {t_lif:.3f} ms across {n} layers"
                      f"   = {share:.1f}% of total" if share is not None else "", flush=True)
            except Exception as e:
                t_lif = None
                print(f"  LIF kernel share: skip ({str(e)[:50]})", flush=True)
            results.append(dict(arch=name, T=T, B=B,
                                 baseline=r_base, rate_classifier=r_rate,
                                 lif_kernel_ms=t_lif))

    out = pathlib.Path(__file__).resolve().parent / "largeT_results.jsonl"
    with open(out, 'a') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\n[results appended to {out}]", flush=True)

    # Summary
    print("\n" + "=" * 100, flush=True)
    print(f"{'Arch':<18s} {'T':>4s} {'base ms/img':>13s} {'rate ms/img':>13s} "
          f"{'rate Δ%':>8s} {'LIF share':>10s} {'peak base':>10s} {'peak rate':>10s}", flush=True)
    print("=" * 100, flush=True)
    for r in results:
        b = r['baseline']; rt = r['rate_classifier']
        if b is None:
            print(f"{r['arch']:<18s} {r['T']:>4d}   FAIL", flush=True)
            continue
        pb = b['mean_ms'] / r['B']
        pr = rt['mean_ms'] / r['B'] if rt else float('nan')
        d = ((rt['mean_ms']/b['mean_ms'])*100 - 100) if rt else float('nan')
        if r['lif_kernel_ms'] is not None and b is not None:
            share = f"{r['lif_kernel_ms']/b['mean_ms']*100:5.1f}%"
        else:
            share = "  -  "
        pb_g = b['peak_gib']; pr_g = rt['peak_gib'] if rt else 0
        print(f"{r['arch']:<18s} {r['T']:>4d}   {pb:>10.4f}    {pr:>10.4f}    "
              f"{d:>+6.2f}%   {share:>9s}  {pb_g:>8.2f}G  {pr_g:>8.2f}G", flush=True)
    print("=" * 100, flush=True)


if __name__ == "__main__":
    main()
