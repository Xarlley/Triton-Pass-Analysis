"""Standalone benchmark: 手工融合 IF kernel vs Inductor autogen IF vs SJ multistep_lif。

只测 IF/LIF kernel 本身在 VGG16-SNN layer 1 输出 shape ([T=4, B=32, C=64, H=W=224])
上的 wall-clock 时间。
"""
import sys, pathlib, time, gc
HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "explore"))

import torch
import torch._dynamo
import torch._inductor.config as inductor_cfg

from fused_if_kernel import fused_if, naive_if


def time_fn(fn, n_warmup=5, n_iters=200):
    """Wall-clock per call, ms."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / n_iters


torch.manual_seed(42)
T, B, C, H, W = 4, 32, 64, 224, 224
print(f"Test shape: [T={T}, B={B}, C={C}, H={H}, W={W}]")
print(f"  numel = {T*B*C*H*W:,}  bytes = {T*B*C*H*W*4/2**30:.2f} GiB\n")

x = torch.randn(T, B, C, H, W, device="cuda").contiguous()

# -------- 1. 手工融合 kernel (soft) --------
def _f_soft():
    return fused_if(x, soft_reset=True, v_threshold=1.0)
_f_soft()  # warmup + autotune
t_soft = time_fn(_f_soft, n_iters=200)
print(f"  fused_if (soft, hand-written)            {t_soft:7.3f} ms / call")

# -------- 2. 手工融合 kernel (hard) --------
def _f_hard():
    return fused_if(x, soft_reset=False, v_threshold=1.0)
_f_hard()
t_hard = time_fn(_f_hard, n_iters=200)
print(f"  fused_if (hard, hand-written)            {t_hard:7.3f} ms / call")

# -------- 3. 朴素 PyTorch eager (soft) --------
def _e_soft():
    return naive_if(x, soft_reset=True, v_threshold=1.0)
t_eager_soft = time_fn(_e_soft, n_iters=50)
print(f"  naive_if (soft, eager pytorch)           {t_eager_soft:7.3f} ms / call")

# -------- 4. 朴素 PyTorch eager (hard) --------
def _e_hard():
    return naive_if(x, soft_reset=False, v_threshold=1.0)
t_eager_hard = time_fn(_e_hard, n_iters=50)
print(f"  naive_if (hard, eager pytorch)           {t_eager_hard:7.3f} ms / call")

# -------- 5. Inductor compile of naive (soft) --------
# 跟 prefix_sum_snn.py 一样的 inductor 配置
torch._dynamo.config.recompile_limit = 256
inductor_cfg.max_autotune = True
inductor_cfg.max_autotune_gemm_backends = "TRITON"
inductor_cfg.max_autotune_conv_backends = "TRITON"
inductor_cfg.force_disable_caches = True

def naive_if_soft(x):
    return naive_if(x, soft_reset=True, v_threshold=1.0)
def naive_if_hard(x):
    return naive_if(x, soft_reset=False, v_threshold=1.0)

print(f"  Inductor-compiling naive_if (soft)... ", end="", flush=True)
t0 = time.perf_counter()
c_soft = torch.compile(naive_if_soft, mode="max-autotune")
c_soft(x)  # cold compile + autotune
torch.cuda.synchronize()
print(f"cold {time.perf_counter() - t0:.1f}s")
t_ind_soft = time_fn(lambda: c_soft(x), n_iters=100)
print(f"  naive_if (soft, Inductor compile)        {t_ind_soft:7.3f} ms / call")

print(f"  Inductor-compiling naive_if (hard)... ", end="", flush=True)
t0 = time.perf_counter()
c_hard = torch.compile(naive_if_hard, mode="max-autotune")
c_hard(x)
torch.cuda.synchronize()
print(f"cold {time.perf_counter() - t0:.1f}s")
t_ind_hard = time_fn(lambda: c_hard(x), n_iters=100)
print(f"  naive_if (hard, Inductor compile)        {t_ind_hard:7.3f} ms / call")

# -------- 6. SJ multistep_lif (hard, decay tau→∞ ≈ IF) --------
try:
    from spikingjelly.activation_based.triton_kernel.neuron_kernel.lif import multistep_lif
    x_seq = x.contiguous()
    v_init = torch.zeros(B, C, H, W, device="cuda")
    def _sj_call():
        return multistep_lif(
            x_seq, v_init,
            decay_input=False, tau=1e10, v_threshold=1.0, v_reset=0.0,
            detach_reset=True, surrogate_function=None,
        )
    # 该接口只在 training=True 路径下接受 sg；inference 路径用 inference 版本：
    from spikingjelly.activation_based.triton_kernel.neuron_kernel.lif import multistep_lif_inference
    def _sj_call_inf():
        return multistep_lif_inference(
            x_seq, v_init,
            decay_input=False, tau=1e10, v_threshold=1.0, v_reset=0.0,
            soft_reset=False,
        )
    try:
        _sj_call_inf()  # warmup / autotune
        t_sj = time_fn(_sj_call_inf, n_iters=200)
        print(f"  SJ multistep_lif_inference (hard, τ→∞)   {t_sj:7.3f} ms / call")
    except Exception as e:
        print(f"  SJ inference call failed: {e}")
except Exception as e:
    print(f"  Skipping SJ multistep_lif: {e}")

print()
print("=" * 60)
print(f"  fused_if (hand) soft  : {t_soft:7.3f} ms")
print(f"  fused_if (hand) hard  : {t_hard:7.3f} ms")
print(f"  naive (eager)   soft  : {t_eager_soft:7.3f} ms")
print(f"  naive (eager)   hard  : {t_eager_hard:7.3f} ms")
print(f"  naive (compile) soft  : {t_ind_soft:7.3f} ms")
print(f"  naive (compile) hard  : {t_ind_hard:7.3f} ms")
print("=" * 60)
