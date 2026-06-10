"""P1 刻画：对两个参考脉冲注意力，测 (a) 端到端 eager 耗时、(b) 各段耗时占比、
(c) Q/K/V 脉冲发放率（决定二值/门控优化能省多少）。

跑法（A100, triton-src）：
    cd ~/charlley/snn_compiler_attn && python snn_compiler/explore/attention/phase1_characterize.py
"""
import os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                       # _bench_util
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))   # snn_compiler repo root
# 参考模型 + shim
sys.path.insert(0, os.path.expanduser("~/charlley/snn_infer_triton"))   # sj_compat, timm_compat
import timm_compat   # noqa: F401
import sj_compat     # noqa: F401  (forces neuron backend per SJ_NEURON_BACKEND, default torch)

import _bench_util as BU

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
dev = "cuda"


def build_spikingformer_attn(dim=768, heads=8):
    sys.path.insert(0, os.path.expanduser("~/charlley/snn_infer/repos/Spikingformer/imagenet"))
    import model as SF
    return SF.SpikingSelfAttention(dim=dim, num_heads=heads).to(dev).eval()


def build_sdtv2_attn(dim=512, heads=8):
    sys.path.insert(0, os.path.expanduser("~/charlley/snn_infer/repos/Spike-Driven-Transformer-V2/classification"))
    import models as M
    return M.MS_Attention_RepConv_qkv_id(dim=dim, num_heads=heads).to(dev).eval()


def firing_rates(attn, x, names=("q_lif", "k_lif", "v_lif", "attn_lif")):
    """挂 forward hook 记录各 LIF 输出的发放率（脉冲均值）。"""
    rates = {}
    handles = []
    for nm in names:
        mod = getattr(attn, nm, None)
        if mod is None:
            continue
        def mk(nm):
            def hook(m, i, o):
                t = o if isinstance(o, torch.Tensor) else o[0]
                rates[nm] = float(t.float().mean().item())
            return hook
        handles.append(mod.register_forward_hook(mk(nm)))
    with torch.no_grad():
        attn(x)
    for h in handles:
        h.remove()
    return rates


def run_one(name, attn, T, B, C, H, W):
    x = torch.randn(T, B, C, H, W, device=dev)
    N = H * W
    with torch.no_grad():
        rates = firing_rates(attn, x)
        fn = lambda: attn(x)
        res = BU.bench(fn, warmup=25, iters=100)
    print(f"\n[{name}]  T={T} B={B} C={C} H=W={H} N={N} heads={attn.num_heads} d={C//attn.num_heads}")
    print(f"   eager forward: {BU.fmt(res)}")
    print(f"   firing rates : " + "  ".join(f"{k}={v:.3f}" for k, v in rates.items()))
    return res, rates


def main():
    print("=" * 70)
    g = BU.gpu_guard(tag="P1-start")
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # 代表性维度：ImageNet 224/patch16 → 14×14=196 tokens
    sf = build_spikingformer_attn(dim=768, heads=8)
    run_one("Spikingformer SSA", sf, T=4, B=16, C=768, H=14, W=14)

    sd = build_sdtv2_attn(dim=512, heads=8)
    run_one("SDT-V2 MS_Attention", sd, T=4, B=16, C=512, H=14, W=14)

    BU.gpu_guard(tag="P1-end")
    print("\nP1_DONE")


if __name__ == "__main__":
    main()
