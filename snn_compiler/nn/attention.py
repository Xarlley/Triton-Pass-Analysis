"""脉冲注意力融合 module —— snn_compiler 对「脉冲注意力块」的优化产物。

覆盖两类常见脉冲注意力（共有形态：无 softmax 的脉冲 Q/K/V + 线性序矩阵乘
``(q@(kᵀ@v))*scale`` + LIF）：

- ``ssa``：Spikingformer ``SpikingSelfAttention`` —— Conv1d-1×1 投影 + BatchNorm1d + 多步 LIF。
- ``ms`` ：SDT-V2 ``MS_Attention_RepConv_qkv_id`` —— Sequential(RepConv, BN2d) 投影（2D 卷积链）+ 多步 LIF。

两者**核心完全相同**，只是投影算子形态不同。优化动作：
- 各 ``*_lif`` → snn_compiler 的 Triton LIF（与 SpikingJelly LIF 逐位一致）；
- ``kᵀ@v`` → torch.bmm（cutlass，实测优于朴素 Triton 二值 GEMM）；
- ``(q@kv)*scale → attn_lif`` → 融合 ``spike_av_lif``（膜电位寄存器跨 T，不落注意力图，scale 折进输入尺度）。

投影：``ssa`` 可折 BN（``fold_bn``）；``ms`` 的 RepConv 链保持 eager（逐位一致；折叠留作后续 micro-opt）。
``fold_bn=False``（默认）→ 逐位一致。只把 triton 当库用，不改 triton 源码。
"""
from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels.fused import fused_bias_if_lif
from ..kernels.attention import spike_av_lif, spike_ktv_popcount, HAS_POPC


def fold_conv1d_bn(conv: nn.Conv1d, bn: nn.BatchNorm1d):
    inv = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + bn.eps)
    w = conv.weight.detach() * inv.view(-1, 1, 1)
    cb = conv.bias.detach() if conv.bias is not None else torch.zeros_like(bn.bias)
    b = (cb - bn.running_mean.detach()) * inv + bn.bias.detach()
    return w, b.float().contiguous()


def _lif_pv(node):
    return float(getattr(node, "tau", 2.0)), float(getattr(node, "v_threshold", 1.0))


def _has_lifs_and_core(m):
    return all(hasattr(m, n) for n in
               ("q_lif", "k_lif", "v_lif", "attn_lif", "q_conv", "k_conv",
                "v_conv", "proj_conv", "num_heads", "scale"))


def is_spiking_self_attention(m: nn.Module) -> bool:
    """Spikingformer SSA：Conv1d 投影 + 独立 BatchNorm1d + proj_lif。"""
    return (_has_lifs_and_core(m) and isinstance(getattr(m, "q_conv"), nn.Conv1d)
            and hasattr(m, "q_bn") and hasattr(m, "proj_lif"))


def is_ms_attention(m: nn.Module) -> bool:
    """SDT-V2 MS_Attention：Sequential(RepConv,BN) 投影 + head_lif。"""
    return (_has_lifs_and_core(m) and isinstance(getattr(m, "q_conv"), nn.Sequential)
            and hasattr(m, "head_lif"))


def is_spiking_attention(m: nn.Module) -> bool:
    return is_spiking_self_attention(m) or is_ms_attention(m)


class FusedSpikeAttention(nn.Module):
    """优化后的脉冲注意力块（variant ∈ {ssa, ms}）。forward 接 [T,B,C,H,W]。"""

    def __init__(self, num_heads, scale, variant, fold_bn=False, ktv_mode="bmm"):
        super().__init__()
        self.num_heads = num_heads
        self.scale = float(scale)
        self.variant = variant
        self.fold_bn = fold_bn and variant == "ssa"   # ms 暂不折 RepConv 链
        # KᵀV：'bmm'(cutlass，默认稳) 或 'popcount'(bit-pack+popcount，逐位一致且实测 1.4–2.2× 更快)
        if ktv_mode == "popcount" and not HAS_POPC:
            ktv_mode = "bmm"
        self.ktv_mode = ktv_mode

    @classmethod
    def from_reference(cls, ref: nn.Module, *, fold_bn: bool = False, ktv_mode: str = "bmm"):
        if is_spiking_self_attention(ref):
            variant = "ssa"
        elif is_ms_attention(ref):
            variant = "ms"
        else:
            raise TypeError(f"not a recognized spiking attention block: {type(ref).__name__}")
        self = cls(ref.num_heads, ref.scale, variant, fold_bn=fold_bn, ktv_mode=ktv_mode)

        in_lif = ref.proj_lif if variant == "ssa" else ref.head_lif
        self.in_tau, self.in_vth = _lif_pv(in_lif)
        self.q_tau, self.q_vth = _lif_pv(ref.q_lif)
        self.k_tau, self.k_vth = _lif_pv(ref.k_lif)
        self.v_tau, self.v_vth = _lif_pv(ref.v_lif)
        self.a_tau, self.a_vth = _lif_pv(ref.attn_lif)

        if variant == "ssa" and self.fold_bn:
            for nm in ("q", "k", "v"):
                w, b = fold_conv1d_bn(getattr(ref, f"{nm}_conv"), getattr(ref, f"{nm}_bn"))
                self.register_buffer(f"{nm}_w", w)
                self.register_buffer(f"{nm}_b", b)
            pw, pb = fold_conv1d_bn(ref.proj_conv, ref.proj_bn)
            self.register_buffer("proj_w", pw)
            self.register_buffer("proj_b", pb)
        elif variant == "ssa":
            for nm in ("q", "k", "v", "proj"):
                setattr(self, f"{nm}_conv", copy.deepcopy(getattr(ref, f"{nm}_conv")).eval())
                setattr(self, f"{nm}_bn", copy.deepcopy(getattr(ref, f"{nm}_bn")).eval())
        else:  # ms：保留整条投影（Sequential RepConv+BN）为 eager 子模块
            for nm in ("q", "k", "v", "proj"):
                setattr(self, f"{nm}_conv", copy.deepcopy(getattr(ref, f"{nm}_conv")).eval())
        return self

    def _lif(self, x, bias, tau, vth, premul=1.0):
        if premul != 1.0:
            x = x * premul
        return fused_bias_if_lif(x.contiguous(), bias, neuron="lif", tau=tau,
                                 decay_input=True, soft_reset=False,
                                 v_threshold=vth, v_reset=0.0, layout="NCHW")

    def _proj(self, xf, nm, tau, vth, T, B, C, N, H, W):
        """投影 + LIF → 脉冲 [T,B,C,N]。"""
        if self.variant == "ssa":
            if self.fold_bn:
                y = F.conv1d(xf, getattr(self, f"{nm}_w"), bias=None)
                bias = getattr(self, f"{nm}_b")
            else:
                y = getattr(self, f"{nm}_bn")(getattr(self, f"{nm}_conv")(xf))
                bias = None
            s = self._lif(y.reshape(T, B, C, N).unsqueeze(-1), bias, tau, vth)
            return s.squeeze(-1)
        else:  # ms：xf 是 [TB,C,H,W]
            y = getattr(self, f"{nm}_conv")(xf).reshape(T, B, C, H, W)
            s = self._lif(y, None, tau, vth)        # [T,B,C,H,W]
            return s.flatten(3)                     # [T,B,C,N]

    def forward(self, x):                           # [T,B,C,H,W]
        T, B, C, H, W = x.shape
        N = H * W
        heads = self.num_heads
        d = C // heads
        x = self._lif(x, None, self.in_tau, self.in_vth).reshape(T, B, C, H, W)
        xf = x.flatten(3).flatten(0, 1) if self.variant == "ssa" else x.flatten(0, 1)

        def to_heads(s):                            # [T,B,C,N] -> [T,B,heads,N,d]
            return (s.transpose(-1, -2).reshape(T, B, N, heads, d)
                     .permute(0, 1, 3, 2, 4).contiguous())

        q = to_heads(self._proj(xf, "q", self.q_tau, self.q_vth, T, B, C, N, H, W))
        k = to_heads(self._proj(xf, "k", self.k_tau, self.k_vth, T, B, C, N, H, W))
        v = to_heads(self._proj(xf, "v", self.v_tau, self.v_vth, T, B, C, N, H, W))

        kv = spike_ktv_popcount(k, v) if self.ktv_mode == "popcount" else (k.transpose(-2, -1) @ v)
        s = spike_av_lif(q, kv, scale=self.scale, tau=self.a_tau,
                         v_threshold=self.a_vth, v_reset=0.0)          # [T,B,heads,N,d]
        s = s.transpose(3, 4).reshape(T, B, C, N)

        if self.variant == "ssa":
            sf = s.flatten(0, 1)
            if self.fold_bn:
                out = F.conv1d(sf, self.proj_w, bias=self.proj_b)
            else:
                out = self.proj_bn(self.proj_conv(sf))
            return out.reshape(T, B, C, H, W)
        else:  # ms：proj_conv 是 2D Sequential，输入 [TB,C,H,W]
            sf = s.reshape(T, B, C, H, W).flatten(0, 1)
            out = self.proj_conv(sf)
            return out.reshape(T, B, C, H, W)
