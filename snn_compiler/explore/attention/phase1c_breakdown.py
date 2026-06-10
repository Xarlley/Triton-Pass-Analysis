"""P1c 组件耗时拆解：用实测发放率的 Bernoulli 脉冲，分别测注意力各段成本，
确定优化优先级（Amdahl）。重点对比：两个 matmul vs eager LIF vs snn_compiler 的 Triton LIF。

跑法（A100, triton-src）：cd ~/charlley/snn_compiler_attn && python snn_compiler/explore/attention/phase1c_breakdown.py
"""
import os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))   # repo root for snn_compiler
sys.path.insert(0, os.path.expanduser("~/charlley/snn_infer_triton"))
import sj_compat  # noqa
import _bench_util as BU

from snn_compiler.kernels.fused import fused_bias_if_lif

dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True


def spikes(shape, rate):
    return (torch.rand(*shape, device=dev) < rate).float()


def run(tag, T, B, heads, N, d, rq, rk, rv, scale=0.125, tau=2.0, vth=0.5):
    C = heads * d
    q = spikes((T, B, heads, N, d), rq)
    k = spikes((T, B, heads, N, d), rk)
    v = spikes((T, B, heads, N, d), rv)

    def kv_mm():     # KᵀV : [d,N]@[N,d] -> [d,d]
        return k.transpose(-2, -1) @ v
    def qkv_mm():    # full: (q @ (kᵀ@v)) * scale
        return (q @ (k.transpose(-2, -1) @ v)) * scale

    kv = kv_mm()
    # attn map a -> reshape 成 LIF 输入 [T,B,C,N]
    a = (q @ kv) * scale                                   # [T,B,heads,N,d]
    a_map = a.transpose(3, 4).reshape(T, B, C, N).contiguous()   # [T,B,C,N]

    # eager LIF（参考实现用的 MultiStepLIFNode, torch 后端）
    from spikingjelly.activation_based import neuron, functional
    lif = neuron.LIFNode(tau=tau, v_threshold=vth, v_reset=0.0, step_mode="m",
                         backend="torch", detach_reset=True).to(dev).eval()
    def eager_lif():
        functional.reset_net(lif)
        return lif(a_map)

    # snn_compiler 的 Triton LIF（hard reset, scale 已折进输入；这里直接喂 a_map）
    a_lifin = a_map.unsqueeze(-1)                          # [T,B,C,N,1] -> 当 H=N,W=1 的 5D
    def triton_lif():
        return fused_bias_if_lif(a_lifin, None, neuron="lif", tau=tau,
                                 soft_reset=False, v_threshold=vth, v_reset=0.0, layout="NCHW")

    r_kv   = BU.bench(kv_mm,   warmup=25, iters=100)
    r_full = BU.bench(qkv_mm,  warmup=25, iters=100)
    r_el   = BU.bench(eager_lif, warmup=15, iters=60)
    r_tl   = BU.bench(triton_lif, warmup=25, iters=100)

    print(f"\n[{tag}]  T={T} B={B} heads={heads} N={N} d={d} C={C}  rates q/k/v={rq}/{rk}/{rv}")
    print(f"   KᵀV matmul only      : {BU.fmt(r_kv)}")
    print(f"   full (q@(kᵀv))*scale : {BU.fmt(r_full)}")
    print(f"   attn_lif  EAGER (SJ) : {BU.fmt(r_el)}")
    print(f"   attn_lif  TRITON(ours): {BU.fmt(r_tl)}")
    print(f"   => matmul/full = {r_kv['median_ms']/r_full['median_ms']*100:.0f}% ; "
          f"eager-LIF / matmul = {r_el['median_ms']/r_full['median_ms']:.1f}x ; "
          f"LIF speedup eager->triton = {r_el['median_ms']/r_tl['median_ms']:.1f}x")
    return dict(kv=r_kv, full=r_full, eager_lif=r_el, triton_lif=r_tl)


def main():
    BU.gpu_guard(tag="P1c-start")
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
    # Spikingformer-ish: d=96, rates q0.07/k0.026/v0.047 ; SDT-V2-ish: d=64, q0.17/k0.034/v0.078
    run("SF-like",  T=4, B=16, heads=8, N=196, d=96, rq=0.072, rk=0.026, rv=0.047)
    run("SDT-like", T=4, B=16, heads=8, N=196, d=64, rq=0.166, rk=0.034, rv=0.078)
    # 大 batch 看 matmul 是否变主导
    run("SF-like B64", T=4, B=64, heads=8, N=196, d=96, rq=0.072, rk=0.026, rv=0.047)
    BU.gpu_guard(tag="P1c-end")
    print("\nP1c_DONE")


if __name__ == "__main__":
    main()
