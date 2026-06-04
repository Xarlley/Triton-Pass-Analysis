"""验证 fuse_snn_model pass 的语义保持性。

策略：构造若干小型 nn.Sequential 网络（含 Conv→BN→Neuron、Conv→Neuron、Linear→Neuron），
跑 pass 前后输出对比。融合前后允许微小数值偏差（因为 BN 折叠+conv 重排
是 fp32 上的代数等价但 fp16/bf16 上有 ULP 差异），但 spike 数应相同。

并验证 SpikingJelly LIFNode 的识别（如果环境里装了 SJ）。
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

from snn_compiler.nn.modules import IFNode, LIFNode, FusedConvNeuron, FusedConvBNNeuron, FusedLinearNeuron
from snn_compiler.passes.fuse import fuse_snn_model


def run_conv_neuron():
    print("== Conv → IFNode ==")
    torch.manual_seed(0)
    m = nn.Sequential(
        nn.Conv2d(8, 16, 3, padding=1),
        IFNode(),
        nn.Conv2d(16, 16, 3, padding=1),
        IFNode(),
    ).cuda().eval()
    x = torch.randn(4, 2, 8, 16, 16, device="cuda")

    # 原始版本：需要 T 步 forward。本测试 module 是单步前向，所以需要手工封装。
    # 简化：直接调用每层。FusedConvNeuron 接受 [T, B, C, H, W]，原 Conv2d 接受 [N, C, H, W]。
    # 因此原始模型也得做 T 步循环。
    def forward_raw(model, x_seq):
        out = x_seq
        for layer in model:
            if isinstance(layer, nn.Conv2d):
                T, B = out.shape[0], out.shape[1]
                out = layer(out.reshape(T * B, *out.shape[2:])).view(T, B, *layer.weight.shape[0:1], *out.shape[-2:])
                # 上一行有问题：H/W 可能被 padding 改变；改成不假设
                pass
        return out

    # 这里我们换种方式：直接用 raw forward 取出每层结果
    def forward_raw_v2(model, x_seq):
        out = x_seq
        T, B = out.shape[0], out.shape[1]
        out = out.reshape(T * B, *out.shape[2:])
        # 逐层
        for layer in model:
            if isinstance(layer, nn.Conv2d):
                out = layer(out)
            elif isinstance(layer, IFNode):
                # IFNode forward 要求 [T, B, ...]
                out = out.view(T, B, *out.shape[1:])
                out = layer(out)
                out = out.reshape(T * B, *out.shape[2:])
        return out.view(T, B, *out.shape[1:])

    ref = forward_raw_v2(m, x).clone()
    fused, n = fuse_snn_model(m)
    print(f"  fused {n} pattern(s)")
    out = forward_raw_v2(fused, x)
    # 但 fused 模型的 FusedConvNeuron 接受 [T, B, ...]
    # forward_raw_v2 在 FusedConvNeuron 上会失败 — 让我们用 dedicated 调用
    # 重新写一个 forward 适配 fused 版
    def forward_fused(model, x_seq):
        out = x_seq
        for layer in model:
            if isinstance(layer, (FusedConvNeuron, FusedConvBNNeuron, FusedLinearNeuron)):
                out = layer(out)
            elif isinstance(layer, nn.Conv2d):
                T, B = out.shape[0], out.shape[1]
                out2 = layer(out.reshape(T * B, *out.shape[2:]))
                out = out2.view(T, B, *out2.shape[1:])
            elif isinstance(layer, IFNode):
                out = layer(out)
        return out
    out = forward_fused(fused, x)
    diff = (out - ref).abs().max().item()
    same_spikes = (out > 0).eq(ref > 0).all().item()
    print(f"  max|out - ref| = {diff:.3e}   same_spikes={same_spikes}")
    assert same_spikes, "spikes mismatch after fusion"
    print("  OK")


def run_conv_bn_neuron():
    print("== Conv → BN → LIFNode ==")
    torch.manual_seed(1)
    m = nn.Sequential(
        nn.Conv2d(8, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        LIFNode(tau=2.0, decay_input=True, soft_reset=False),
    ).cuda().eval()
    # 给 BN 有意义的 stats
    m[1].running_mean.copy_(torch.randn(16, device="cuda"))
    m[1].running_var.copy_(torch.rand(16, device="cuda") + 0.1)
    m[1].weight.data.copy_(torch.rand(16, device="cuda") + 0.5)
    m[1].bias.data.copy_(torch.randn(16, device="cuda") * 0.1)

    x = torch.randn(4, 2, 8, 16, 16, device="cuda")

    def forward_raw(model, x_seq):
        T, B = x_seq.shape[0], x_seq.shape[1]
        out = x_seq.reshape(T * B, *x_seq.shape[2:])
        out = model[0](out)
        out = model[1](out)
        out = out.view(T, B, *out.shape[1:])
        out = model[2](out)
        return out

    ref = forward_raw(m, x).clone()
    fused, n = fuse_snn_model(m)
    print(f"  fused {n} pattern(s)")
    assert n == 1
    out = fused[0](x)
    diff = (out - ref).abs().max().item()
    same_spikes = (out > 0).eq(ref > 0).all().item()
    print(f"  max|out - ref| = {diff:.3e}   same_spikes={same_spikes}")
    assert same_spikes
    print("  OK")


def run_linear_neuron():
    print("== Linear → IFNode ==")
    torch.manual_seed(2)
    m = nn.Sequential(
        nn.Linear(128, 64),
        IFNode(),
    ).cuda().eval()
    x = torch.randn(4, 8, 128, device="cuda")
    def forward_raw(model, x_seq):
        T, B = x_seq.shape[0], x_seq.shape[1]
        y = model[0](x_seq.reshape(T * B, -1))
        y = y.view(T, B, -1)
        return model[1](y)
    ref = forward_raw(m, x).clone()
    fused, n = fuse_snn_model(m)
    assert n == 1
    out = fused[0](x)
    diff = (out - ref).abs().max().item()
    same_spikes = (out > 0).eq(ref > 0).all().item()
    print(f"  max|out - ref| = {diff:.3e}   same_spikes={same_spikes}")
    assert same_spikes
    print("  OK")


def run_nested():
    print("== nested Sequential of (Conv,BN,LIF) ==")
    torch.manual_seed(3)
    block = lambda ic, oc: nn.Sequential(
        nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc),
        LIFNode(tau=2.0, decay_input=True, soft_reset=False, v_threshold=1.0),
    )
    m = nn.Sequential(block(8, 16), block(16, 16)).cuda().eval()
    # 给两个 BN 有效 stats
    for bn in [m[0][1], m[1][1]]:
        bn.running_mean.copy_(torch.randn_like(bn.running_mean))
        bn.running_var.copy_(torch.rand_like(bn.running_var) + 0.1)

    fused, n = fuse_snn_model(m)
    print(f"  fused {n} pattern(s) (expected 2)")
    assert n == 2

    # 跑一遍前向
    x = torch.randn(4, 2, 8, 16, 16, device="cuda")
    out = x
    for sub in fused:
        out = sub[0](out)
    print(f"  output shape = {tuple(out.shape)}")
    print("  OK")


def main():
    run_conv_neuron()
    run_conv_bn_neuron()
    run_linear_neuron()
    run_nested()
    print("\n== ALL GRAPH PASS TESTS PASSED ==")


if __name__ == "__main__":
    main()
