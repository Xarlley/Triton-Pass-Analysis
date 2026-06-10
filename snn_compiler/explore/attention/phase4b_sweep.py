"""P4b 速度扫表：两类脉冲注意力块，扫 B，对比 eager / FusedSpikeAttention / torch.compile。

速度与权重数值无关（matmul/LIF/conv 的耗时不依赖发放率），故用随机权重 + Bernoulli 脉冲输入，
免加载 checkpoint。GPU-guard + 冷启动感知。
跑法（A100, triton-src）：cd ~/charlley/snn_compiler_attn && SJ_NEURON_BACKEND=torch python snn_compiler/explore/attention/phase4b_sweep.py
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

dev = "cuda"


def build_ssa(dim=768, heads=8):
    sys.path.insert(0, os.path.expanduser("~/charlley/snn_infer/repos/Spikingformer/imagenet"))
    import model as SF
    return SF.SpikingSelfAttention(dim=dim, num_heads=heads).to(dev).eval()


def build_ms(dim=512, heads=8):
    sys.path.insert(0, os.path.expanduser("~/charlley/snn_infer/repos/Spike-Driven-Transformer-V2/classification"))
    import models as M
    return M.MS_Attention_RepConv_qkv_id(dim=dim, num_heads=heads).to(dev).eval()


def sweep(tag, block, C, T=4, H=14, W=14, Bs=(8, 16, 32, 64), compile_Bs=(16, 64)):
    from spikingjelly.activation_based import functional
    fused = FusedSpikeAttention.from_reference(block, fold_bn=False).to(dev).eval()
    print(f"\n=== {tag}  C={C} T={T} N={H*W} ===")
    print(f"{'B':>4} | {'eager(ms)':>10} {'fused(ms)':>10} {'compile(ms)':>11} | {'fused/eager':>11} {'fused/compile':>13}")
    for B in Bs:
        x = (torch.rand(T, B, C, H, W, device=dev) < 0.2).float()
        re = BU.bench(lambda: (functional.reset_net(block), block(x))[1], warmup=12, iters=50)
        rf = BU.bench(lambda: fused(x), warmup=20, iters=80)
        cs = ""
        sp_c = ""
        if B in compile_Bs:
            try:
                import torch._inductor.config as ic
                ic.compile_threads = 1
                cb = torch.compile(block, mode="max-autotune-no-cudagraphs")
                rc = BU.bench(lambda: (functional.reset_net(block), cb(x))[1], warmup=15, iters=40)
                cs = f"{rc['median_ms']:.3f}"
                sp_c = f"{rc['median_ms']/rf['median_ms']:.2f}x"
            except Exception as e:
                cs = "FAIL"
        print(f"{B:>4} | {re['median_ms']:>10.3f} {rf['median_ms']:>10.3f} {cs:>11} | "
              f"{re['median_ms']/rf['median_ms']:>10.2f}x {sp_c:>13}")


def main():
    BU.gpu_guard(tag="P4b-start")
    print(f"{torch.cuda.get_device_name(0)}")
    sweep("Spikingformer SSA", build_ssa(768, 8), C=768)
    sweep("SDT-V2 MS_Attention", build_ms(512, 8), C=512)
    BU.gpu_guard(tag="P4b-end")
    print("\nP4b_DONE")


if __name__ == "__main__":
    main()
