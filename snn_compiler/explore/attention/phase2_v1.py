"""P2 v1：脉冲感知 matmul kernel 的正确性 + 测速。

对比：
- v0 路径：kv=bmm(kᵀ,v)；a=(q@kv)*scale（bmm，落 [T,B,heads,N,d] 显存）；triton LIF。
- v1 路径：kv=spike_ktv；spike=spike_av_lif（融合 q@kv+scale+LIF，不落注意力图）。
- v1b：kv=bmm；spike_av_lif（隔离「av-lif 融合」单独的贡献）。

正确性：spike_av_lif 与 v0 的 bmm+triton-LIF 应**逐位一致**（q∈{0,1}、kv 小整数→ matmul 精确）。
跑法（A100, triton-src）：cd ~/charlley/snn_compiler_attn && python snn_compiler/explore/attention/phase2_v1.py
"""
import os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
import _bench_util as BU
from snn_compiler.kernels.fused import fused_bias_if_lif
from snn_compiler.kernels.attention import spike_av_lif, spike_ktv

dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True


def spikes(shape, rate):
    return (torch.rand(*shape, device=dev) < rate).float()


def v0_full(q, k, v, scale, tau, vth):
    kv = k.transpose(-2, -1) @ v
    a = (q @ kv) * scale                            # [T,B,heads,N,d]
    return fused_bias_if_lif(a.contiguous(), None, neuron="lif", tau=tau,
                             soft_reset=False, v_threshold=vth, v_reset=0.0, layout="NCHW")


def run(tag, T, B, heads, N, d, rq, rk, rv, scale=0.125, tau=2.0, vth=0.5):
    q = spikes((T, B, heads, N, d), rq)
    k = spikes((T, B, heads, N, d), rk)
    v = spikes((T, B, heads, N, d), rv)

    # --- correctness ---
    kv_bmm = k.transpose(-2, -1) @ v
    kv_ker = spike_ktv(k, v)
    kv_err = (kv_bmm - kv_ker).abs().max().item()

    s_v0 = v0_full(q, k, v, scale, tau, vth)
    s_v1 = spike_av_lif(q, kv_ker, scale=scale, tau=tau, v_threshold=vth, v_reset=0.0)
    spike_match = torch.equal(s_v0, s_v1)
    sdiff = (s_v0 - s_v1).abs().max().item()

    # --- speed ---
    r_v0  = BU.bench(lambda: v0_full(q, k, v, scale, tau, vth), warmup=25, iters=100)
    r_v1  = BU.bench(lambda: spike_av_lif(q, spike_ktv(k, v), scale=scale, tau=tau, v_threshold=vth), warmup=25, iters=100)
    r_v1b = BU.bench(lambda: spike_av_lif(q, k.transpose(-2, -1) @ v, scale=scale, tau=tau, v_threshold=vth), warmup=25, iters=100)

    print(f"\n[{tag}] T={T} B={B} heads={heads} N={N} d={d}  rates q/k/v={rq}/{rk}/{rv}")
    print(f"   correctness: KᵀV max|bmm-kernel|={kv_err:.3e}   spike bit-exact(v1==v0)={spike_match} (max|Δ|={sdiff:.1e})")
    print(f"   v0  (bmm+bmm+map+lif)   : {BU.fmt(r_v0)}")
    print(f"   v1  (ktv-ker+fused-av)  : {BU.fmt(r_v1)}")
    print(f"   v1b (bmm-kv+fused-av)   : {BU.fmt(r_v1b)}")
    print(f"   => v1/v0 = {r_v0['median_ms']/r_v1['median_ms']:.2f}x ; v1b/v0 = {r_v0['median_ms']/r_v1b['median_ms']:.2f}x")
    return dict(v0=r_v0, v1=r_v1, v1b=r_v1b, match=spike_match)


def main():
    BU.gpu_guard(tag="P2v1-start")
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
    run("SF-like",     T=4, B=16, heads=8, N=196, d=96, rq=0.072, rk=0.026, rv=0.047)
    run("SDT-like",    T=4, B=16, heads=8, N=196, d=64, rq=0.166, rk=0.034, rv=0.078)
    run("SF-like B64", T=4, B=64, heads=8, N=196, d=96, rq=0.072, rk=0.026, rv=0.047)
    BU.gpu_guard(tag="P2v1-end")
    print("\nP2v1_DONE")


if __name__ == "__main__":
    main()
