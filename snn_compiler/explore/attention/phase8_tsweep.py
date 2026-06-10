"""P8 T-sweep：脉冲注意力的融合收益随 T 如何变化（T 越大 eager LIF 的 python 时间循环越吃亏，
预期融合赢更多）。也顺带验证 spike_av_lif / popcount 路径对 T=8,16 仍逐位一致。

跑法（A100, triton-src）：cd ~/charlley/snn_compiler_attn && SJ_NEURON_BACKEND=torch python snn_compiler/explore/attention/phase8_tsweep.py
"""
import os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.expanduser("~/charlley/snn_infer_triton"))
import timm_compat  # noqa
import sj_compat    # noqa
import _bench_util as BU
from snn_compiler.nn.attention import FusedSpikeAttention
from snn_compiler.verify import compare_models

dev = "cuda"


def build_ssa(dim=768, heads=8):
    sys.path.insert(0, os.path.expanduser("~/charlley/snn_infer/repos/Spikingformer/imagenet"))
    import model as SF
    return SF.SpikingSelfAttention(dim=dim, num_heads=heads).to(dev).eval()


def main():
    from spikingjelly.activation_based import functional
    BU.gpu_guard(tag="P8-start")
    block = build_ssa(768, 8)
    fused = FusedSpikeAttention.from_reference(block, fold_bn=False).to(dev).eval()
    fused_pc = FusedSpikeAttention.from_reference(block, fold_bn=False, ktv_mode="popcount").to(dev).eval()
    C, H, W, B = 768, 14, 14, 16
    print(f"=== Spikingformer SSA  C={C} B={B} N={H*W}  T-sweep ===")
    print(f"{'T':>3} | {'eager':>9} {'fused':>9} {'fused_pc':>9} | {'f/eager':>8} {'pc/eager':>8} | bit-exact")
    for T in (4, 8, 16):
        x = (torch.rand(T, B, C, H, W, device=dev) < 0.2).float()
        # 正确性（bmm + popcount 两路）
        functional.reset_net(block)
        be = compare_models(block, fused, x)["bit_exact"]
        be_pc = compare_models(block, fused_pc, x)["bit_exact"]
        re = BU.bench(lambda: (functional.reset_net(block), block(x))[1], warmup=12, iters=40)
        rf = BU.bench(lambda: fused(x), warmup=20, iters=60)
        rp = BU.bench(lambda: fused_pc(x), warmup=20, iters=60)
        print(f"{T:>3} | {re['median_ms']:>9.3f} {rf['median_ms']:>9.3f} {rp['median_ms']:>9.3f} | "
              f"{re['median_ms']/rf['median_ms']:>7.2f}x {re['median_ms']/rp['median_ms']:>7.2f}x | "
              f"{be} / {be_pc}")
    BU.gpu_guard(tag="P8-end")
    print("\nP8_DONE")


if __name__ == "__main__":
    main()
