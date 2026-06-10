"""脉冲注意力融合的正确性测试（自包含，不依赖外部模型/权重）。

构造迷你的 SSA 形态（Conv1d 投影）与 MS 形态（Conv2d Sequential 投影）参考块，
参考神经元用 naive_if_lif（= snn_compiler Triton LIF 的逐位参考），断言
FusedSpikeAttention.from_reference 与参考**逐位一致**，并验证探测 + pass。
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import pytest
import torch
import torch.nn as nn

from snn_compiler.kernels.neurons import naive_if_lif
from snn_compiler.nn.attention import (
    FusedSpikeAttention, is_spiking_self_attention, is_ms_attention, is_spiking_attention,
)
from snn_compiler.passes import fuse_spiking_attention
from snn_compiler.kernels.attention import spike_ktv_popcount, HAS_POPC
from snn_compiler import assert_equivalent


class _RefLIF(nn.Module):
    """参考多步 LIF（与 snn_compiler 的 fused_bias_if_lif 逐位等价：hard, v_reset=0, decay_input）。"""
    def __init__(self, tau=2.0, v_threshold=1.0):
        super().__init__()
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = 0.0

    def forward(self, x):
        return naive_if_lif(x, neuron="lif", tau=self.tau, decay_input=True,
                            soft_reset=False, v_threshold=self.v_threshold, v_reset=0.0)


def _to_heads(s, T, B, N, heads, d):
    return (s.transpose(-1, -2).reshape(T, B, N, heads, d)
             .permute(0, 1, 3, 2, 4).contiguous())


def _bn1d(c):
    bn = nn.BatchNorm1d(c).eval()
    bn.running_mean.normal_(0, 0.2); bn.running_var.uniform_(0.5, 1.5)
    bn.weight.data.uniform_(0.5, 1.5); bn.bias.data.normal_(0, 0.2)
    return bn


def _bn2d(c):
    bn = nn.BatchNorm2d(c).eval()
    bn.running_mean.normal_(0, 0.2); bn.running_var.uniform_(0.5, 1.5)
    bn.weight.data.uniform_(0.5, 1.5); bn.bias.data.normal_(0, 0.2)
    return bn


class _RefSSA(nn.Module):
    """迷你 Spikingformer SpikingSelfAttention（Conv1d 投影）。"""
    def __init__(self, C=32, heads=4):
        super().__init__()
        self.num_heads = heads; self.scale = 0.125
        self.proj_lif = _RefLIF(v_threshold=0.3)
        for nm in ("q", "k", "v"):
            setattr(self, f"{nm}_conv", nn.Conv1d(C, C, 1, bias=False))
            setattr(self, f"{nm}_bn", _bn1d(C))
            setattr(self, f"{nm}_lif", _RefLIF(v_threshold=0.3))
        self.attn_lif = _RefLIF(v_threshold=0.5)
        self.proj_conv = nn.Conv1d(C, C, 1)
        self.proj_bn = _bn1d(C)

    def forward(self, x):
        T, B, C, H, W = x.shape; N = H * W; heads = self.num_heads; d = C // heads
        x = self.proj_lif(x).flatten(3)              # [T,B,C,N]
        xf = x.flatten(0, 1)
        def proj(nm, lif):
            y = getattr(self, f"{nm}_bn")(getattr(self, f"{nm}_conv")(xf)).reshape(T, B, C, N)
            return _to_heads(lif(y), T, B, N, heads, d)
        q = proj("q", self.q_lif); k = proj("k", self.k_lif); v = proj("v", self.v_lif)
        a = (q @ (k.transpose(-2, -1) @ v)) * self.scale
        a = a.transpose(3, 4).reshape(T, B, C, N)
        s = self.attn_lif(a).flatten(0, 1)
        return self.proj_bn(self.proj_conv(s)).reshape(T, B, C, H, W)


class _RefMS(nn.Module):
    """迷你 SDT-V2 MS_Attention（Conv2d Sequential 投影；用普通 conv 代 RepConv）。"""
    def __init__(self, C=32, heads=4):
        super().__init__()
        self.num_heads = heads; self.scale = 0.125
        self.head_lif = _RefLIF(v_threshold=0.3)
        for nm in ("q", "k", "v", "proj"):
            setattr(self, f"{nm}_conv",
                    nn.Sequential(nn.Conv2d(C, C, 3, padding=1, bias=False), _bn2d(C)))
        for nm in ("q", "k", "v"):
            setattr(self, f"{nm}_lif", _RefLIF(v_threshold=0.3))
        self.attn_lif = _RefLIF(v_threshold=0.5)

    def forward(self, x):
        T, B, C, H, W = x.shape; N = H * W; heads = self.num_heads; d = C // heads
        x = self.head_lif(x)
        def proj(nm, lif):
            y = getattr(self, f"{nm}_conv")(x.flatten(0, 1)).reshape(T, B, C, H, W)
            return _to_heads(lif(y).flatten(3), T, B, N, heads, d)
        q = proj("q", self.q_lif); k = proj("k", self.k_lif); v = proj("v", self.v_lif)
        a = (q @ (k.transpose(-2, -1) @ v)) * self.scale
        a = a.transpose(3, 4).reshape(T, B, C, N)
        s = self.attn_lif(a).reshape(T, B, C, H, W)
        return self.proj_conv(s.flatten(0, 1)).reshape(T, B, C, H, W)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_ssa_detect_and_bit_exact():
    torch.manual_seed(0)
    ref = _RefSSA(C=32, heads=4).cuda().eval()
    assert is_spiking_self_attention(ref) and is_spiking_attention(ref)
    x = (torch.rand(4, 2, 32, 8, 8, device="cuda") < 0.3).float()   # 脉冲输入
    fused = FusedSpikeAttention.from_reference(ref, fold_bn=False).cuda().eval()
    rep = assert_equivalent(ref, fused, x)                          # 默认要求逐位一致
    assert rep["bit_exact"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_ms_detect_and_bit_exact():
    torch.manual_seed(1)
    ref = _RefMS(C=32, heads=4).cuda().eval()
    assert is_ms_attention(ref) and is_spiking_attention(ref)
    x = (torch.rand(4, 2, 32, 8, 8, device="cuda") < 0.3).float()
    fused = FusedSpikeAttention.from_reference(ref, fold_bn=False).cuda().eval()
    rep = assert_equivalent(ref, fused, x)
    assert rep["bit_exact"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fuse_pass_replaces_in_container():
    torch.manual_seed(2)
    model = nn.ModuleList([_RefSSA(C=32, heads=4), _RefMS(C=32, heads=4)]).cuda().eval()
    n = fuse_spiking_attention(model, fold_bn=False)
    assert n == 2
    assert all(type(m).__name__ == "FusedSpikeAttention" for m in model)


@pytest.mark.skipif(not (torch.cuda.is_available() and HAS_POPC),
                    reason="needs CUDA + libdevice.popc")
def test_spike_ktv_popcount_bit_exact():
    torch.manual_seed(3)
    T, B, H, N, d = 4, 2, 4, 70, 16          # N 非 32 的倍数，测 mask
    k = (torch.rand(T, B, H, N, d, device="cuda") < 0.1).float()
    v = (torch.rand(T, B, H, N, d, device="cuda") < 0.15).float()
    kv_bmm = k.transpose(-2, -1) @ v
    assert torch.equal(kv_bmm, spike_ktv_popcount(k, v))


@pytest.mark.skipif(not (torch.cuda.is_available() and HAS_POPC),
                    reason="needs CUDA + libdevice.popc")
def test_ssa_popcount_ktv_still_bit_exact():
    torch.manual_seed(0)
    ref = _RefSSA(C=32, heads=4).cuda().eval()
    x = (torch.rand(4, 2, 32, 8, 8, device="cuda") < 0.3).float()
    fused = FusedSpikeAttention.from_reference(ref, fold_bn=False, ktv_mode="popcount").cuda().eval()
    assert fused.ktv_mode == "popcount"      # 确认没退回 bmm
    rep = assert_equivalent(ref, fused, x)
    assert rep["bit_exact"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
