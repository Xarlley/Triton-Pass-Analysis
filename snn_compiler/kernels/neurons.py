"""统一 spiking neuron kernel 库（generalized IF/LIF/CubaLIF/EIF）。

设计原则
=========
1. 所有 kernel 共用同一个 outer-parallel + T-register-loop 模式：
     grid = (ceil(NCL / BLOCK_NCL),)              # NCL = B*C*H*W (or B*N for 1D)
     for t in tl.static_range(T):                  # T 在寄存器循环
         v = step(v, x_t, params)
         spike = (v >= v_th)
         v = reset(v, spike, v_reset, soft_reset)
         store(spike_t)

2. 静态属性走 constexpr（neuron 类型、reset 模式、is_dynamic_threshold），
   动态量走 runtime tensor（v_threshold tensor、bias、weight）。

3. dtype：输入可 fp32/bf16/fp16，state 始终在 fp32 累加（避免长 T 下漂移）。

4. 不持久化 v：本框架只做推理，每条样本独立，T 步内完成后 v 即丢弃。
   如需训练或时序持久化，扩展时把 v 作为 in/out tensor 即可。

支持的 neuron 模型
==================
- IF       : v_t = v_{t-1} + x_t                            (decay=0 退化)
- IF/decay : v_t = decay * v_{t-1} + x_t                    (一般化 IF)
- LIF      : v_t = (1-1/τ) v_{t-1} + s * x_t                (s=1 或 1/τ)
- CubaLIF  : i_t = α i_{t-1} + x_t; v_t = β v_{t-1} + i_t  (双状态)
- EIF      : v_t = β v_{t-1} + ΔT exp((v-V_T)/ΔT) + x_t    (指数 IF)

支持的 reset 模式
==================
- soft : v ← v - threshold * spike
- hard : v ← v_reset where spike, else v

支持的阈值模式
==============
- SCALAR     : 一个 constexpr float（最快）
- PER_CHANNEL: tensor[C]，按 c_idx 加载
- PER_NEURON : tensor[NCL]，按 ncl_idx 加载（per-position dynamic）

布局兼容
========
- NCHW 与 NHWC 均通过同一 kernel 处理：因为 spike kernel 是 elementwise+前缀依赖型，
  布局不影响每个 neuron 的 v 状态计算（v 沿 T 维累积，不沿 spatial）。
  唯一差异是 per-channel threshold 的 c_idx 推断方式（NCHW: ncl_idx//(H*W)%C；
  NHWC: ncl_idx%C），用 constexpr CHANNEL_LAST 切换。
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ============================================================
#   constexpr code 编号
#   Python 侧使用普通 int 即可（按值传给 @triton.jit 时自动作 constexpr）。
# ============================================================
NEURON_IF = 0
NEURON_LIF = 1
NEURON_CUBA = 2
NEURON_EIF = 3

RESET_SOFT = 0
RESET_HARD = 1

THR_SCALAR = 0
THR_PER_CHANNEL = 1
THR_PER_NEURON = 2


# ============================================================
#   IF / LIF 统一 kernel（单状态）
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": 128}, num_warps=4),
        triton.Config({"BLOCK_NCL": 256}, num_warps=4),
        triton.Config({"BLOCK_NCL": 256}, num_warps=8),
        triton.Config({"BLOCK_NCL": 512}, num_warps=8),
        triton.Config({"BLOCK_NCL": 1024}, num_warps=8),
    ],
    key=["T", "NCL", "THR_MODE", "RESET_MODE", "CHANNEL_LAST"],
)
@triton.jit
def _if_lif_kernel(
    x_ptr,                              # [T, NCL]  fp16/bf16/fp32
    spike_ptr,                          # [T, NCL]  same dtype as x
    v_th_ptr,                           # [] / [C] / [NCL]
    T: tl.constexpr,
    NCL: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,                   # H*W；只用于 NCHW per-channel index
    BLOCK_NCL: tl.constexpr,
    decay_factor: tl.constexpr,         # IF 默认 1.0；LIF=(1-1/τ)
    input_scale: tl.constexpr,          # IF=1.0；LIF decay_input=True 时=1/τ
    v_threshold_const: tl.constexpr,    # THR_MODE=SCALAR 时使用
    v_reset_val: tl.constexpr,          # hard reset 时使用
    RESET_MODE: tl.constexpr,           # 0=soft, 1=hard
    THR_MODE: tl.constexpr,             # 0=scalar, 1=per-channel, 2=per-neuron
    CHANNEL_LAST: tl.constexpr,         # 仅 PER_CHANNEL 时区分 NCHW/NHWC
):
    """通用 IF/LIF kernel。

    数学：
       v_t = decay_factor * v_{t-1} + input_scale * x_t
       spike = (v_t >= v_threshold)
       if soft: v_t -= spike * v_threshold
       else:    v_t = where(spike, v_reset, v_t)
    """
    pid = tl.program_id(0)
    ncl_idx = pid.to(tl.int64) * BLOCK_NCL + tl.arange(0, BLOCK_NCL).to(tl.int64)
    mask = ncl_idx < NCL
    NCL_i64 = tl.full([], NCL, dtype=tl.int64)

    # 加载阈值：0=scalar, 1=per-channel, 2=per-neuron
    if THR_MODE == 0:
        v_th = v_threshold_const
    elif THR_MODE == 1:
        if CHANNEL_LAST:
            c_idx = ncl_idx % C
        else:
            c_idx = (ncl_idx // HW) % C
        v_th = tl.load(v_th_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    else:
        v_th = tl.load(v_th_ptr + ncl_idx, mask=mask, other=0.0).to(tl.float32)

    v = tl.zeros([BLOCK_NCL], dtype=tl.float32)

    for t in tl.static_range(0, T, 1):
        t_off = tl.full([], t, dtype=tl.int64) * NCL_i64
        x_t = tl.load(x_ptr + t_off + ncl_idx, mask=mask, other=0.0).to(tl.float32)
        v = decay_factor * v + input_scale * x_t
        spike = (v >= v_th).to(tl.float32)
        if RESET_MODE == 0:  # soft
            v = v - spike * v_th
        else:                # hard
            v = v * (1.0 - spike) + spike * v_reset_val
        tl.store(spike_ptr + t_off + ncl_idx, spike, mask=mask)


# ============================================================
#   CubaLIF kernel（双状态：突触电流 i + 膜电位 v）
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": 128}, num_warps=4),
        triton.Config({"BLOCK_NCL": 256}, num_warps=4),
        triton.Config({"BLOCK_NCL": 512}, num_warps=8),
        triton.Config({"BLOCK_NCL": 1024}, num_warps=8),
    ],
    key=["T", "NCL", "RESET_MODE"],
)
@triton.jit
def _cuba_lif_kernel(
    x_ptr, spike_ptr, v_th_ptr,
    T: tl.constexpr, NCL: tl.constexpr, C: tl.constexpr, HW: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    alpha: tl.constexpr,                # 突触衰减 = exp(-dt/τ_syn)
    beta: tl.constexpr,                 # 膜衰减   = exp(-dt/τ_mem)
    input_scale: tl.constexpr,          # 输入到突触电流的尺度（通常 1.0）
    v_threshold_const: tl.constexpr,
    v_reset_val: tl.constexpr,
    RESET_MODE: tl.constexpr,
    THR_MODE: tl.constexpr,
    CHANNEL_LAST: tl.constexpr,
):
    """CubaLIF: 双状态线性。
       i_t = alpha * i_{t-1} + input_scale * x_t
       v_t = beta  * v_{t-1} + i_t
       spike + reset 同 IF/LIF
    """
    pid = tl.program_id(0)
    ncl_idx = pid.to(tl.int64) * BLOCK_NCL + tl.arange(0, BLOCK_NCL).to(tl.int64)
    mask = ncl_idx < NCL
    NCL_i64 = tl.full([], NCL, dtype=tl.int64)

    if THR_MODE == 0:
        v_th = v_threshold_const
    elif THR_MODE == 1:
        if CHANNEL_LAST:
            c_idx = ncl_idx % C
        else:
            c_idx = (ncl_idx // HW) % C
        v_th = tl.load(v_th_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    else:
        v_th = tl.load(v_th_ptr + ncl_idx, mask=mask, other=0.0).to(tl.float32)

    i_syn = tl.zeros([BLOCK_NCL], dtype=tl.float32)
    v = tl.zeros([BLOCK_NCL], dtype=tl.float32)

    for t in tl.static_range(0, T, 1):
        t_off = tl.full([], t, dtype=tl.int64) * NCL_i64
        x_t = tl.load(x_ptr + t_off + ncl_idx, mask=mask, other=0.0).to(tl.float32)
        i_syn = alpha * i_syn + input_scale * x_t
        v = beta * v + i_syn
        spike = (v >= v_th).to(tl.float32)
        if RESET_MODE == 0:
            v = v - spike * v_th
        else:
            v = v * (1.0 - spike) + spike * v_reset_val
        tl.store(spike_ptr + t_off + ncl_idx, spike, mask=mask)


# ============================================================
#   EIF kernel（指数 IF：非线性项）
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": 128}, num_warps=4),
        triton.Config({"BLOCK_NCL": 256}, num_warps=4),
        triton.Config({"BLOCK_NCL": 512}, num_warps=8),
    ],
    key=["T", "NCL", "RESET_MODE"],
)
@triton.jit
def _eif_kernel(
    x_ptr, spike_ptr, v_th_ptr,
    T: tl.constexpr, NCL: tl.constexpr, C: tl.constexpr, HW: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    decay_factor: tl.constexpr,         # (1 - 1/τ) 膜泄漏
    input_scale: tl.constexpr,
    v_threshold_const: tl.constexpr,
    v_reset_val: tl.constexpr,
    delta_T: tl.constexpr,              # 指数斜率
    v_rh: tl.constexpr,                 # 软阈值（指数项中心）
    RESET_MODE: tl.constexpr,
    THR_MODE: tl.constexpr,
    CHANNEL_LAST: tl.constexpr,
):
    """EIF: v_t = decay * v_{t-1} + ΔT exp((v_{t-1}-v_rh)/ΔT) + input_scale * x_t
       发放阈值仍是 v_threshold（通常 v_th > v_rh）。
    """
    pid = tl.program_id(0)
    ncl_idx = pid.to(tl.int64) * BLOCK_NCL + tl.arange(0, BLOCK_NCL).to(tl.int64)
    mask = ncl_idx < NCL
    NCL_i64 = tl.full([], NCL, dtype=tl.int64)

    if THR_MODE == 0:
        v_th = v_threshold_const
    elif THR_MODE == 1:
        if CHANNEL_LAST:
            c_idx = ncl_idx % C
        else:
            c_idx = (ncl_idx // HW) % C
        v_th = tl.load(v_th_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    else:
        v_th = tl.load(v_th_ptr + ncl_idx, mask=mask, other=0.0).to(tl.float32)

    v = tl.zeros([BLOCK_NCL], dtype=tl.float32)

    for t in tl.static_range(0, T, 1):
        t_off = tl.full([], t, dtype=tl.int64) * NCL_i64
        x_t = tl.load(x_ptr + t_off + ncl_idx, mask=mask, other=0.0).to(tl.float32)
        e = tl.exp(tl.minimum((v - v_rh) / delta_T, 16.0))
        v = decay_factor * v + delta_T * e + input_scale * x_t
        spike = (v >= v_th).to(tl.float32)
        if RESET_MODE == 0:
            v = v - spike * v_th
        else:
            v = v * (1.0 - spike) + spike * v_reset_val
        tl.store(spike_ptr + t_off + ncl_idx, spike, mask=mask)


# ============================================================
#   Python entrypoints
# ============================================================
def _resolve_thr(v_threshold, x_seq, layout):
    """解析 v_threshold (float / Tensor) → (THR_MODE, ptr, scalar, channel_last)。

    框架约定：x_seq.shape 始终按 NCHW 标记 [T, B, C, H, W]；layout 仅影响
    kernel 内 c_idx 的访存模式（channels_last 时 ncl_idx % C）。
    """
    if isinstance(v_threshold, (int, float)):
        return THR_SCALAR, None, float(v_threshold), layout == "NHWC"
    assert isinstance(v_threshold, torch.Tensor), type(v_threshold)
    assert v_threshold.is_cuda and v_threshold.dtype == torch.float32 and v_threshold.is_contiguous()

    if x_seq.ndim >= 4:
        C = x_seq.shape[2]
    elif x_seq.ndim == 3:
        C = x_seq.shape[-1]
    else:
        C = 1
    NCL = x_seq[0].numel()
    if v_threshold.numel() == C:
        return THR_PER_CHANNEL, v_threshold, 0.0, layout == "NHWC"
    if v_threshold.numel() == NCL:
        return THR_PER_NEURON, v_threshold, 0.0, False
    raise ValueError(
        f"v_threshold shape {tuple(v_threshold.shape)} matches neither C={C} "
        f"nor NCL={NCL}"
    )


def _infer_C_HW(x_seq, layout):
    """从 x_seq 推断 C 与 HW（用于 PER_CHANNEL 索引）。

    PyTorch 约定：shape 标记始终是 NCHW（[T, B, C, H, W]），channels_last 仅是
    内存格式不是 shape 重排。本框架沿用此约定 — layout='NHWC' 仅控制 kernel
    内 c_idx 的计算方式（与 channels_last 内存格式配套）。
    """
    if x_seq.ndim == 5:
        return x_seq.shape[2], x_seq.shape[3] * x_seq.shape[4]
    if x_seq.ndim == 4:
        return x_seq.shape[2], x_seq.shape[3]
    if x_seq.ndim == 3:
        return x_seq.shape[-1], 1
    return 1, 1


def if_lif(
    x_seq: torch.Tensor,
    *,
    neuron: str = "if",                 # 'if' | 'lif'
    tau: float = 2.0,
    decay: float | None = None,         # IF/通用衰减；None 则按 neuron 推断
    decay_input: bool = True,
    soft_reset: bool = False,
    v_threshold=1.0,                    # float | tensor[C] | tensor[NCL]
    v_reset: float = 0.0,
    layout: str = "NCHW",
) -> torch.Tensor:
    """统一 IF/LIF 推理入口。

    Args:
        x_seq: [T, B, ...] tensor。dtype fp16/bf16/fp32。内存连续。
        neuron: 'if' 或 'lif'。
        tau: LIF 时间常数；对 IF 无效。
        decay: 自定义衰减系数（覆盖 neuron 内置）。
            - 当 neuron='if' 默认 decay=1.0；可传 0.9 等表示 leaky IF。
            - 当 neuron='lif' 默认 decay=1-1/τ；不建议手动覆盖。
        decay_input: LIF 模式下控制是否对输入除以 τ。SJ LIFNode 默认 True。
        soft_reset: True=软复位（v -= θ * spike），False=硬复位（v ← v_reset）。
        v_threshold: 标量、[C] 或 [NCL]。
        v_reset: 硬复位电位（默认 0）。
        layout: 'NCHW' 或 'NHWC'，仅在 v_threshold 是 [C] 时影响 c_idx 推断。
    """
    assert x_seq.is_cuda
    # channels_last 5D 视图的 is_contiguous() 返回 False，但底层内存仍然连续。
    if not x_seq.is_contiguous():
        assert layout == "NHWC", \
            f"non-contiguous x_seq requires layout='NHWC', got {layout}"
    assert x_seq.dtype in (torch.float32, torch.float16, torch.bfloat16)
    T = x_seq.shape[0]
    NCL = x_seq[0].numel()
    C, HW = _infer_C_HW(x_seq, layout)

    # neuron-specific 参数
    if neuron == "if":
        decay_factor = 1.0 if decay is None else float(decay)
        input_scale = 1.0
    elif neuron == "lif":
        decay_factor = (1.0 - 1.0 / tau) if decay is None else float(decay)
        input_scale = (1.0 / tau) if decay_input else 1.0
    else:
        raise ValueError(f"unknown neuron: {neuron!r}")

    THR_MODE, thr_ptr, thr_const, channel_last = _resolve_thr(
        v_threshold, x_seq, layout
    )
    RESET_MODE = RESET_SOFT if soft_reset else RESET_HARD

    spike_seq = torch.empty_like(x_seq)
    grid = lambda meta: (triton.cdiv(NCL, meta["BLOCK_NCL"]),)
    _if_lif_kernel[grid](
        x_seq, spike_seq, thr_ptr if thr_ptr is not None else x_seq,
        T=T, NCL=NCL, C=C, HW=HW,
        decay_factor=decay_factor,
        input_scale=input_scale,
        v_threshold_const=thr_const,
        v_reset_val=v_reset,
        RESET_MODE=RESET_MODE,
        THR_MODE=THR_MODE,
        CHANNEL_LAST=channel_last,
    )
    return spike_seq


def cuba_lif(
    x_seq: torch.Tensor,
    *,
    tau_syn: float = 2.0,
    tau_mem: float = 4.0,
    dt: float = 1.0,
    decay_syn: float | None = None,    # 显式覆盖 α；None 时 = exp(-dt/τ_syn)
    decay_mem: float | None = None,    # 显式覆盖 β；None 时 = exp(-dt/τ_mem)
    input_scale: float = 1.0,
    soft_reset: bool = False,
    v_threshold=1.0,
    v_reset: float = 0.0,
    layout: str = "NCHW",
) -> torch.Tensor:
    """CubaLIF 推理入口。

    默认  alpha = exp(-dt/τ_syn);  beta = exp(-dt/τ_mem)
    可通过 decay_syn / decay_mem 直接覆盖 alpha / beta（例如想让 alpha=0 退化成
    纯 LIF，或想用与 τ 推导不一致的非物理 decay 做训练实验）。
    """
    assert x_seq.is_cuda and x_seq.is_contiguous()
    T = x_seq.shape[0]
    NCL = x_seq[0].numel()
    C, HW = _infer_C_HW(x_seq, layout)

    import math
    alpha = math.exp(-dt / tau_syn) if decay_syn is None else float(decay_syn)
    beta  = math.exp(-dt / tau_mem) if decay_mem is None else float(decay_mem)

    THR_MODE, thr_ptr, thr_const, channel_last = _resolve_thr(
        v_threshold, x_seq, layout
    )
    RESET_MODE = RESET_SOFT if soft_reset else RESET_HARD
    spike_seq = torch.empty_like(x_seq)
    grid = lambda meta: (triton.cdiv(NCL, meta["BLOCK_NCL"]),)
    _cuba_lif_kernel[grid](
        x_seq, spike_seq, thr_ptr if thr_ptr is not None else x_seq,
        T=T, NCL=NCL, C=C, HW=HW,
        alpha=alpha, beta=beta, input_scale=input_scale,
        v_threshold_const=thr_const, v_reset_val=v_reset,
        RESET_MODE=RESET_MODE, THR_MODE=THR_MODE, CHANNEL_LAST=channel_last,
    )
    return spike_seq


def eif(
    x_seq: torch.Tensor,
    *,
    tau: float = 2.0,
    decay: float | None = None,        # 显式覆盖膜衰减；None 时 = 1 - 1/τ
    delta_T: float = 1.0,
    v_rh: float = 0.5,
    input_scale: float = 1.0,
    soft_reset: bool = False,
    v_threshold=1.0,
    v_reset: float = 0.0,
    layout: str = "NCHW",
) -> torch.Tensor:
    """指数 IF 推理入口。

    默认 decay_factor = 1 - 1/τ；可通过 decay 参数直接覆盖（例如设 decay=1.0 关
    掉线性泄漏，仅保留指数非线性项，对应 quadratic IF 风格的极限）。
    """
    assert x_seq.is_cuda and x_seq.is_contiguous()
    T = x_seq.shape[0]
    NCL = x_seq[0].numel()
    C, HW = _infer_C_HW(x_seq, layout)
    decay_factor = (1.0 - 1.0 / tau) if decay is None else float(decay)
    THR_MODE, thr_ptr, thr_const, channel_last = _resolve_thr(
        v_threshold, x_seq, layout
    )
    RESET_MODE = RESET_SOFT if soft_reset else RESET_HARD
    spike_seq = torch.empty_like(x_seq)
    grid = lambda meta: (triton.cdiv(NCL, meta["BLOCK_NCL"]),)
    _eif_kernel[grid](
        x_seq, spike_seq, thr_ptr if thr_ptr is not None else x_seq,
        T=T, NCL=NCL, C=C, HW=HW,
        decay_factor=decay_factor, input_scale=input_scale,
        v_threshold_const=thr_const, v_reset_val=v_reset,
        delta_T=delta_T, v_rh=v_rh,
        RESET_MODE=RESET_MODE, THR_MODE=THR_MODE, CHANNEL_LAST=channel_last,
    )
    return spike_seq


# ============================================================
#   朴素参考实现：用于 bit-equal 测试
# ============================================================
def naive_if_lif(x_seq, *, neuron="if", tau=2.0, decay=None, decay_input=True,
                  soft_reset=False, v_threshold=1.0, v_reset=0.0, layout="NCHW"):
    T = x_seq.shape[0]
    if neuron == "if":
        df = 1.0 if decay is None else float(decay)
        sc = 1.0
    else:
        df = (1.0 - 1.0 / tau) if decay is None else float(decay)
        sc = (1.0 / tau) if decay_input else 1.0
    if isinstance(v_threshold, torch.Tensor):
        per_neuron = v_threshold.numel() == x_seq[0].numel()
        if per_neuron:
            v_th = v_threshold.view_as(x_seq[0])
        elif x_seq.ndim >= 4:                   # 5D [T,B,C,H,W] → broadcast at C
            view = [1] * (x_seq.ndim - 1)
            view[1] = x_seq.shape[2]
            v_th = v_threshold.view(*view)
        elif x_seq.ndim == 3:                   # 3D [T,B,C]
            view = [1, x_seq.shape[-1]]
            v_th = v_threshold.view(*view)
        else:
            v_th = v_threshold
    else:
        v_th = float(v_threshold)
    v = torch.zeros_like(x_seq[0], dtype=torch.float32)
    spikes = []
    for t in range(T):
        v = df * v + sc * x_seq[t].to(torch.float32)
        spike = (v >= v_th).to(torch.float32)
        spikes.append(spike)
        if soft_reset:
            v = v - spike * v_th
        else:
            v = torch.where(spike > 0, torch.full_like(v, float(v_reset)), v)
    out = torch.stack(spikes, dim=0)
    return out.to(x_seq.dtype)


def naive_cuba_lif(x_seq, *, tau_syn=2.0, tau_mem=4.0, dt=1.0,
                    decay_syn=None, decay_mem=None,
                    input_scale=1.0, soft_reset=False, v_threshold=1.0,
                    v_reset=0.0, layout="NCHW"):
    import math
    T = x_seq.shape[0]
    alpha = math.exp(-dt / tau_syn) if decay_syn is None else float(decay_syn)
    beta  = math.exp(-dt / tau_mem) if decay_mem is None else float(decay_mem)
    if isinstance(v_threshold, torch.Tensor):
        if layout == "NCHW" and x_seq.ndim >= 4 and v_threshold.numel() == x_seq.shape[2]:
            view = [1] * (x_seq.ndim - 1)
            view[1] = x_seq.shape[2]
            v_th = v_threshold.view(*view)
        else:
            v_th = v_threshold.view_as(x_seq[0])
    else:
        v_th = float(v_threshold)
    i_syn = torch.zeros_like(x_seq[0], dtype=torch.float32)
    v = torch.zeros_like(x_seq[0], dtype=torch.float32)
    spikes = []
    for t in range(T):
        i_syn = alpha * i_syn + input_scale * x_seq[t].to(torch.float32)
        v = beta * v + i_syn
        spike = (v >= v_th).to(torch.float32)
        spikes.append(spike)
        if soft_reset:
            v = v - spike * v_th
        else:
            v = torch.where(spike > 0, torch.full_like(v, float(v_reset)), v)
    return torch.stack(spikes, dim=0).to(x_seq.dtype)


def naive_eif(x_seq, *, tau=2.0, decay=None, delta_T=1.0, v_rh=0.5,
              input_scale=1.0, soft_reset=False, v_threshold=1.0,
              v_reset=0.0, layout="NCHW"):
    T = x_seq.shape[0]
    df = (1.0 - 1.0 / tau) if decay is None else float(decay)
    if isinstance(v_threshold, torch.Tensor):
        v_th = v_threshold.view_as(x_seq[0]) if v_threshold.numel() == x_seq[0].numel() else v_threshold
    else:
        v_th = float(v_threshold)
    v = torch.zeros_like(x_seq[0], dtype=torch.float32)
    spikes = []
    for t in range(T):
        e = torch.exp(torch.clamp((v - v_rh) / delta_T, max=16.0))
        v = df * v + delta_T * e + input_scale * x_seq[t].to(torch.float32)
        spike = (v >= v_th).to(torch.float32)
        spikes.append(spike)
        if soft_reset:
            v = v - spike * v_th
        else:
            v = torch.where(spike > 0, torch.full_like(v, float(v_reset)), v)
    return torch.stack(spikes, dim=0).to(x_seq.dtype)
