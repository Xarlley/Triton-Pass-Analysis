"""通用版：FusedSpikingNeuron Triton kernel，IF 与 LIF 在同一模板下实现。

同一个 outer-parallel + T-register-loop 模式，按 constexpr 切换：
  - NEURON_TYPE=0 (IF):   v_t = v_{t-1} + x_t                         (无 decay)
  - NEURON_TYPE=1 (LIF):  v_t = (1 - 1/tau) * v_{t-1} + x_t          (decay_input=False)
  - NEURON_TYPE=2 (LIF):  v_t = (1 - 1/tau) * v_{t-1} + x_t / tau    (decay_input=True)
                                                                       (SJ LIFNode 默认)
RESET 也是 constexpr 控制（soft / hard）。v_reset 作 runtime 参数（不限于 0）。

证明同一个 MLIR-level pattern 对 IF / LIF 都成立。
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": 64},  num_warps=2),
        triton.Config({"BLOCK_NCL": 128}, num_warps=4),
        triton.Config({"BLOCK_NCL": 256}, num_warps=4),
        triton.Config({"BLOCK_NCL": 256}, num_warps=8),
        triton.Config({"BLOCK_NCL": 512}, num_warps=8),
    ],
    key=["T", "NCL", "NEURON_TYPE", "SOFT_RESET"],
)
@triton.jit
def _fused_spiking_neuron_kernel(
    x_ptr,                       # [T, NCL]
    spike_ptr,                   # [T, NCL]
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    v_threshold: tl.constexpr,
    v_reset_val: tl.constexpr,
    decay_factor: tl.constexpr,  # for IF=1.0 (no decay); for LIF=(1 - 1/tau)
    input_scale: tl.constexpr,   # for IF=1.0; for LIF decay_input=True 是 1/tau, decay_input=False 是 1.0
    SOFT_RESET: tl.constexpr,
):
    """统一 spiking neuron kernel:

       Per t step:
         v_t = decay_factor * v_{t-1} + input_scale * x_t
         spike = (v_t >= v_threshold) ? 1 : 0
         if SOFT_RESET: v_t = v_t - spike * v_threshold
         else (hard):   v_t = spike ? v_reset_val : v_t       (= where(spike, v_reset, v))
    """
    pid = tl.program_id(0)
    ncl_idx = pid * BLOCK_NCL + tl.arange(0, BLOCK_NCL)
    mask = ncl_idx < NCL

    v = tl.zeros([BLOCK_NCL], dtype=tl.float32)

    for t in tl.static_range(0, T, 1):
        x_t = tl.load(x_ptr + t * NCL + ncl_idx, mask=mask, other=0.0).to(tl.float32)
        # ★ 唯一与 IF 不同的一行：加 decay 与 input_scale
        v = decay_factor * v + input_scale * x_t
        spike = (v >= v_threshold).to(tl.float32)
        if SOFT_RESET:
            v = v - spike * v_threshold
        else:
            # hard reset：spike 时 v 替换为 v_reset_val
            v = v * (1.0 - spike) + spike * v_reset_val
        tl.store(spike_ptr + t * NCL + ncl_idx, spike, mask=mask)


def fused_spiking_neuron(
    x_seq: torch.Tensor,
    neuron_type: str = "if",       # "if" | "lif"
    tau: float = 2.0,
    decay_input: bool = True,
    soft_reset: bool = False,
    v_threshold: float = 1.0,
    v_reset: float = 0.0,
) -> torch.Tensor:
    """统一 IF / LIF 调用入口。

    Args:
        x_seq: [T, B, ...] fp32 contiguous
        neuron_type: 'if' or 'lif'
        tau: LIF 时间常数（仅 lif 用）
        decay_input: 对应 SJ LIFNode 的 decay_input
        soft_reset: True 软复位，False 硬复位
        v_threshold: 发放阈值
        v_reset: 硬复位时重置到的电位（默认 0）
    """
    assert x_seq.is_cuda and x_seq.dtype == torch.float32
    assert x_seq.is_contiguous()
    T = x_seq.shape[0]
    NCL = x_seq[0].numel()

    if neuron_type == "if":
        decay_factor = 1.0
        input_scale = 1.0
        NEURON_TYPE = 0
    elif neuron_type == "lif":
        decay_factor = 1.0 - 1.0 / tau
        if decay_input:
            input_scale = 1.0 / tau           # SJ 默认: v = (1-1/τ)v + x/τ
            NEURON_TYPE = 2
        else:
            input_scale = 1.0                 # v = (1-1/τ)v + x
            NEURON_TYPE = 1
    else:
        raise ValueError(f"unknown neuron_type: {neuron_type!r}")

    spike_seq = torch.empty_like(x_seq)
    grid = lambda meta: (triton.cdiv(NCL, meta["BLOCK_NCL"]),)
    _fused_spiking_neuron_kernel[grid](
        x_seq, spike_seq,
        T=T, NCL=NCL,
        v_threshold=v_threshold,
        v_reset_val=v_reset,
        decay_factor=decay_factor,
        input_scale=input_scale,
        SOFT_RESET=soft_reset,
    )
    return spike_seq


# -------------------- 参考实现 --------------------
def naive_spiking(x_seq, neuron_type="if", tau=2.0, decay_input=True,
                  soft_reset=False, v_threshold=1.0, v_reset=0.0):
    """朴素顺序参考，用于 bit-equal 验证。"""
    T = x_seq.shape[0]
    if neuron_type == "if":
        decay_factor, input_scale = 1.0, 1.0
    elif neuron_type == "lif":
        decay_factor = 1.0 - 1.0 / tau
        input_scale = 1.0 / tau if decay_input else 1.0
    v = torch.zeros_like(x_seq[0])
    spikes = []
    for t in range(T):
        v = decay_factor * v + input_scale * x_seq[t]
        spike = (v >= v_threshold).to(x_seq.dtype)
        spikes.append(spike)
        if soft_reset:
            v = v - spike * v_threshold
        else:
            v = torch.where(spike > 0, torch.full_like(v, v_reset), v)
    return torch.stack(spikes, dim=0)


# -------------------- selftest: IF + LIF 全配置 --------------------
def _selftest():
    torch.manual_seed(0)
    cases = []
    for shape in [(4, 16), (4, 1024)]:
        for ntype in ["if", "lif"]:
            for soft in [True, False]:
                for di in ([True, False] if ntype == "lif" else [True]):
                    cases.append((shape, ntype, soft, di))

    print(f"{'shape':<14s} {'neuron':<8s} {'reset':<5s} {'decay_input':<12s} {'bit-eq':<8s}")
    print("-" * 60)
    for shape, ntype, soft, di in cases:
        x = torch.randn(*shape, device="cuda").contiguous()
        kwargs = dict(neuron_type=ntype, tau=2.0, decay_input=di,
                       soft_reset=soft, v_threshold=1.0, v_reset=0.0)
        r_naive = naive_spiking(x, **kwargs)
        r_fused = fused_spiking_neuron(x, **kwargs)
        eq = torch.equal(r_naive, r_fused)
        diff = (r_naive != r_fused).sum().item()
        total = r_naive.numel()
        print(f"{str(shape):<14s} {ntype:<8s} {'soft' if soft else 'hard':<5s} "
              f"{str(di):<12s} {str(eq):<8s} ({diff}/{total})")


if __name__ == "__main__":
    _selftest()
