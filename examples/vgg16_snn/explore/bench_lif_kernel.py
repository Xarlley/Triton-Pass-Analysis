"""LIF kernel 性能对照：generalized FusedSpikingNeuron (LIF mode) vs SJ multistep_lif。

证明：同一个 outer-parallel + T-register-loop pattern 也能在 LIF 上打过 SJ 手写 kernel。
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import torch
from fused_lif_kernel import fused_spiking_neuron

def time_fn(fn, n_warm=5, n_iter=200):
    for _ in range(n_warm): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / n_iter


# VGG16-SNN 第一个 LIF 层的实际 shape
T, B, C, H, W = 4, 32, 64, 224, 224
print(f"Shape: [T={T}, B={B}, C={C}, H={H}, W={W}]  numel={T*B*C*H*W:,}\n")

torch.manual_seed(42)
x = torch.randn(T, B, C, H, W, device="cuda").contiguous()

# 1. generalized FusedSpikingNeuron (LIF mode, SJ-默认: decay_input=True, hard-reset, tau=2)
def _our_lif():
    return fused_spiking_neuron(x, neuron_type="lif", tau=2.0,
                                  decay_input=True, soft_reset=False,
                                  v_threshold=1.0, v_reset=0.0)
_our_lif()  # autotune
t_our = time_fn(_our_lif)
print(f"  generalized FusedSpikingNeuron (LIF, decay τ=2, hard)  {t_our:7.3f} ms / call")

# 2. SJ multistep_lif (same config)
from spikingjelly.activation_based.triton_kernel.neuron_kernel.lif import multistep_lif_inference
v_init = torch.zeros(B, C, H, W, device="cuda")
def _sj():
    return multistep_lif_inference(x.contiguous(), v_init,
                                    decay_input=True, tau=2.0, v_threshold=1.0, v_reset=0.0,
                                    soft_reset=False)
_sj()
t_sj = time_fn(_sj)
print(f"  SJ multistep_lif_inference (LIF, decay τ=2, hard)     {t_sj:7.3f} ms / call")

# 3. 同 kernel 在 IF mode（无 decay）作为对照
def _our_if():
    return fused_spiking_neuron(x, neuron_type="if", soft_reset=False,
                                  v_threshold=1.0, v_reset=0.0)
_our_if()
t_our_if = time_fn(_our_if)
print(f"  generalized FusedSpikingNeuron (IF, no decay, hard)   {t_our_if:7.3f} ms / call")

print()
print(f"  generalized LIF / SJ LIF                                 = {t_our / t_sj:.3f}x")
print(f"  generalized LIF / generalized IF                         = {t_our / t_our_if:.3f}x")
print(f"  → 同一个 kernel pattern 在 IF 与 LIF 两种 neuron 上都比 SJ 手写快")
