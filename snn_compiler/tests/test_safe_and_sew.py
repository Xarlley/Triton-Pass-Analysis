"""新增能力的测试：
1. fold_bn=False 的 FusedConvBN(Add)Neuron 与"conv→BN→neuron"逐位一致
2. SEW zoo（sew_resnet18_snn）naive vs fused(fold_bn=False) 逐位一致；ADD/AND/IAND
3. SEW 与标准 ResNet 是不同网络（输出不同）—— 锁定拓扑差异
4. snn_compiler.verify.assert_equivalent：逐位通过、拓扑接错时报错
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import pytest
import torch
import torch.nn as nn

from snn_compiler.kernels.neurons import naive_if_lif
from snn_compiler.nn.modules import FusedConvBNNeuron, FusedConvBNAddNeuron
from snn_compiler.zoo import sew_resnet18_snn, resnet18_snn
from snn_compiler import assert_equivalent, compare_models

IF = dict(neuron="if", soft_reset=False, v_threshold=1.0, v_reset=0.0)


def _rand_bn(c):
    bn = nn.BatchNorm2d(c).cuda().eval()
    bn.running_mean.copy_(torch.randn(c, device="cuda") * 0.2)
    bn.running_var.copy_(torch.rand(c, device="cuda") + 0.5)
    bn.weight.data.copy_(torch.rand(c, device="cuda") + 0.5)
    bn.bias.data.copy_(torch.randn(c, device="cuda") * 0.2)
    return bn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fold_bn_false_is_bit_exact():
    torch.manual_seed(0)
    T, B, Cin, Cout, H, W = 4, 2, 8, 16, 16, 16
    conv = nn.Conv2d(Cin, Cout, 3, padding=1, bias=False).cuda().eval()
    bn = _rand_bn(Cout)
    x = torch.randn(T, B, Cin, H, W, device="cuda")

    ref_pre = bn(conv(x.reshape(T * B, Cin, H, W))).view(T, B, Cout, H, W)
    ref = naive_if_lif(ref_pre, **IF)

    exact = FusedConvBNNeuron(conv, bn, fold_bn=False, layout="NCHW", **IF).cuda()(x)
    assert torch.equal(exact, ref), (exact - ref).abs().max().item()

    # fold_bn=True 跑得通（数值可能不同，不强求逐位）
    folded = FusedConvBNNeuron(conv, bn, fold_bn=True, layout="NCHW", **IF).cuda()(x)
    assert folded.shape == ref.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fold_bn_false_add_is_bit_exact():
    torch.manual_seed(1)
    T, B, Cin, Cout, H, W = 4, 2, 8, 16, 12, 12
    conv = nn.Conv2d(Cin, Cout, 3, padding=1, bias=False).cuda().eval()
    bn = _rand_bn(Cout)
    x = torch.randn(T, B, Cin, H, W, device="cuda")
    r = torch.randn(T, B, Cout, H, W, device="cuda")

    ref_pre = bn(conv(x.reshape(T * B, Cin, H, W))).view(T, B, Cout, H, W)
    ref = naive_if_lif(ref_pre + r, **IF)

    exact = FusedConvBNAddNeuron(conv, bn, fold_bn=False, layout="NCHW", **IF).cuda()(x, r)
    assert torch.equal(exact, ref), (exact - ref).abs().max().item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("cf", ["ADD", "AND", "IAND"])
def test_sew_zoo_fused_bit_exact(cf):
    x = torch.randn(4, 2, 3, 64, 64, device="cuda")
    torch.manual_seed(7)
    m_naive = sew_resnet18_snn(num_classes=10, neuron="if", connect_f=cf,
                               init_bn=True, fused=False).cuda().eval()
    torch.manual_seed(7)
    m_fused = sew_resnet18_snn(num_classes=10, neuron="if", connect_f=cf,
                               init_bn=True, fused=True, fold_bn=False).cuda().eval()
    with torch.no_grad():
        y1 = m_naive(x); y2 = m_fused(x)
    assert (y1 - y2).abs().max().item() < 1e-4, cf


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_sew_is_not_standard_resnet():
    """同样深度/权重种子下，SEW 与标准 ResNet 输出不同 —— 证明拓扑确实不同。"""
    x = torch.randn(4, 2, 3, 64, 64, device="cuda")
    torch.manual_seed(11)
    sew = sew_resnet18_snn(num_classes=10, neuron="if", connect_f="ADD",
                           init_bn=True, fused=False).cuda().eval()
    torch.manual_seed(11)
    std = resnet18_snn(num_classes=10, neuron="if", init_bn=True,
                       fused=False).cuda().eval()
    with torch.no_grad():
        d = (sew(x) - std(x)).abs().max().item()
    assert d > 1e-3, "SEW 与标准 ResNet 输出不应相同"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_verify_passes_on_bitexact():
    x = torch.randn(4, 2, 3, 64, 64, device="cuda")
    torch.manual_seed(3)
    ref = sew_resnet18_snn(num_classes=10, neuron="if", init_bn=True, fused=False).cuda().eval()
    torch.manual_seed(3)
    fast = sew_resnet18_snn(num_classes=10, neuron="if", init_bn=True,
                            fused=True, fold_bn=False).cuda().eval()
    rep = assert_equivalent(ref, fast, x)         # 默认要求逐位一致
    assert rep["bit_exact"] and rep["argmax_agree"] == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_verify_catches_wrong_topology():
    """把标准 ResNet 当作 SEW 的"加速版"——verify 必须报错（防静默用错）。"""
    x = torch.randn(4, 2, 3, 64, 64, device="cuda")
    torch.manual_seed(5)
    sew = sew_resnet18_snn(num_classes=10, neuron="if", init_bn=True, fused=False).cuda().eval()
    torch.manual_seed(5)
    wrong = resnet18_snn(num_classes=10, neuron="if", init_bn=True, fused=True).cuda().eval()
    with pytest.raises(AssertionError):
        assert_equivalent(sew, wrong, x)
    # compare_models 不抛异常，但应报告不一致
    rep = compare_models(sew, wrong, x)
    assert rep["shape_match"] and not rep["bit_exact"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
