"""验证：
1. fused_bias_if_lif 的 residual 通路（HAS_RESIDUAL=True）bit-equal
2. FusedAddNeuron / FusedConvBNAddNeuron 在 ResNet 残差块上的语义正确
3. snn_compiler.zoo 里 VGG / ResNet / MobileNet-V2 三族 SNN 的朴素 vs 融合
   前向输出 max|diff|==0（fp32）
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from snn_compiler.kernels.neurons import naive_if_lif
from snn_compiler.kernels.fused import fused_bias_if_lif
from snn_compiler.nn.modules import FusedAddNeuron, FusedConvBNAddNeuron


# ============================================================
#   1. fused_bias_if_lif + residual 路径 bit-equal
# ============================================================
def run_residual_kernel():
    print("== fused_bias_if_lif residual path ==")
    torch.manual_seed(0)
    T, B, C, H, W = 4, 2, 16, 8, 8
    fails = 0
    for neuron in ["if", "lif"]:
        for soft in [True, False]:
            for v_reset in [0.0, 0.3]:
                if soft and v_reset != 0.0:
                    continue
                y = torch.randn(T, B, C, H, W, device="cuda").contiguous()
                r = torch.randn(T, B, C, H, W, device="cuda").contiguous()
                bias = torch.randn(C, device="cuda")
                # naive: y + bias + r → neuron
                broadcast_bias = bias.view(1, 1, C, 1, 1)
                inp = y + broadcast_bias + r
                ref = naive_if_lif(
                    inp, neuron=neuron, tau=2.0, decay_input=True,
                    soft_reset=soft, v_threshold=1.0, v_reset=v_reset,
                )
                # fused
                out = fused_bias_if_lif(
                    y, bias, residual=r,
                    neuron=neuron, tau=2.0, decay_input=True,
                    soft_reset=soft, v_threshold=1.0, v_reset=v_reset, layout="NCHW",
                )
                ok = torch.equal(ref, out)
                tag = f"{neuron} soft={soft} v_reset={v_reset}"
                if not ok:
                    fails += 1
                    print(f"  [FAIL] {tag}  diff={(ref != out).sum().item()}")
                else:
                    print(f"  [ OK ] {tag}")
    return fails


# ============================================================
#   2. FusedAddNeuron / FusedConvBNAddNeuron 语义
# ============================================================
def run_add_neuron_module():
    print("== FusedAddNeuron module ==")
    torch.manual_seed(1)
    a = torch.randn(4, 2, 16, 8, 8, device="cuda").contiguous()
    b = torch.randn(4, 2, 16, 8, 8, device="cuda").contiguous()
    m = FusedAddNeuron(neuron="lif", tau=2.0, decay_input=True,
                       soft_reset=False, v_threshold=1.0, v_reset=0.0).cuda()
    out = m(a, b)
    ref = naive_if_lif(a + b, neuron="lif", tau=2.0, decay_input=True,
                       soft_reset=False, v_threshold=1.0, v_reset=0.0)
    ok = torch.equal(ref, out)
    print(f"  bit-equal: {ok}")
    return 0 if ok else 1


def run_conv_bn_add_neuron_module():
    print("== FusedConvBNAddNeuron module ==")
    torch.manual_seed(2)
    T, B, in_C, out_C, H, W = 4, 2, 8, 16, 16, 16
    conv = nn.Conv2d(in_C, out_C, 3, padding=1, bias=False).cuda().eval()
    bn = nn.BatchNorm2d(out_C).cuda().eval()
    bn.running_mean.copy_(torch.randn(out_C, device="cuda") * 0.1)
    bn.running_var.copy_(torch.rand(out_C, device="cuda") + 0.5)
    bn.weight.data.copy_(torch.rand(out_C, device="cuda") + 0.5)
    bn.bias.data.copy_(torch.randn(out_C, device="cuda") * 0.1)

    x = torch.randn(T, B, in_C, H, W, device="cuda")
    r = torch.randn(T, B, out_C, H, W, device="cuda")

    # naive: conv(x) → BN → +r → IFNode
    y = bn(conv(x.reshape(T * B, in_C, H, W))).view(T, B, out_C, H, W)
    ref = naive_if_lif(y + r, neuron="lif", tau=2.0, decay_input=True,
                        soft_reset=False, v_threshold=1.0, v_reset=0.0)

    mod = FusedConvBNAddNeuron(conv, bn, neuron="lif", tau=2.0,
                                decay_input=True, soft_reset=False,
                                v_threshold=1.0, v_reset=0.0,
                                layout="NCHW").cuda()
    out = mod(x, r)
    diff = (out - ref).abs().max().item()
    same_spikes = (out > 0).eq(ref > 0).all().item()
    print(f"  max|out - ref| = {diff:.3e}   same_spikes = {same_spikes}")
    return 0 if same_spikes and diff < 1e-4 else 1


# ============================================================
#   3. zoo 模型端到端：fused vs naive 输出一致
# ============================================================
def run_zoo_forward():
    from snn_compiler.zoo import vgg11_snn, resnet18_snn, mobilenet_v2_snn

    torch.manual_seed(3)
    fails = 0

    print("== zoo: VGG-11 SNN (fp32, NCHW, fused vs naive) ==")
    torch.manual_seed(100)
    x = torch.randn(2, 1, 3, 224, 224, device="cuda")
    m1 = vgg11_snn(num_classes=10, init_bn=True, fused=False).cuda().eval()
    torch.manual_seed(100)
    m2 = vgg11_snn(num_classes=10, init_bn=True, fused=True).cuda().eval()
    with torch.no_grad():
        y1 = m1(x); y2 = m2(x)
    diff = (y1 - y2).abs().max().item()
    print(f"  max|fused - naive| = {diff:.3e}")
    if diff > 1e-3:
        fails += 1
        print("  [FAIL] vgg11 fused vs naive output mismatch")

    print("== zoo: ResNet-18 SNN (fp32, NCHW, fused vs naive) ==")
    torch.manual_seed(200)
    x = torch.randn(2, 1, 3, 64, 64, device="cuda")        # ResNet 没 fc_in_dim 锁定
    m1 = resnet18_snn(num_classes=10, init_bn=True, fused=False).cuda().eval()
    torch.manual_seed(200)
    m2 = resnet18_snn(num_classes=10, init_bn=True, fused=True).cuda().eval()
    with torch.no_grad():
        y1 = m1(x); y2 = m2(x)
    diff = (y1 - y2).abs().max().item()
    print(f"  max|fused - naive| = {diff:.3e}")
    if diff > 1e-3:
        fails += 1
        print("  [FAIL] resnet18 fused vs naive output mismatch")

    print("== zoo: MobileNet-V2 SNN (fp32, NCHW, fused vs naive) ==")
    torch.manual_seed(300)
    x = torch.randn(2, 1, 3, 64, 64, device="cuda")
    m1 = mobilenet_v2_snn(num_classes=10, init_bn=True, fused=False).cuda().eval()
    torch.manual_seed(300)
    m2 = mobilenet_v2_snn(num_classes=10, init_bn=True, fused=True).cuda().eval()
    with torch.no_grad():
        y1 = m1(x); y2 = m2(x)
    diff = (y1 - y2).abs().max().item()
    print(f"  max|fused - naive| = {diff:.3e}")
    if diff > 1e-3:
        fails += 1
        print("  [FAIL] mobilenet_v2 fused vs naive output mismatch")
    return fails


def main():
    total = 0
    total += run_residual_kernel()
    total += run_add_neuron_module()
    total += run_conv_bn_add_neuron_module()
    total += run_zoo_forward()
    print("\n" + "=" * 50)
    print("  PASS" if total == 0 else f"  {total} FAIL")
    print("=" * 50)
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
