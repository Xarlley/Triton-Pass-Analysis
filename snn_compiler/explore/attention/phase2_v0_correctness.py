"""P2 v0：在真实 Spikingformer 注意力块上验证「快速脉冲注意力前向」与参考逐元素一致 + 测速。

快速实现 = 复用参考权重，但：
  - 每个 *_lif 用 snn_compiler 的 Triton LIF（fused_bias_if_lif）替换 SJ eager LIF；
  - q/k/v/proj 的 BN 折进 conv（fold_bn）→ 卷积无 bias、bias 折进 LIF；
  - attn 的 *scale 折进 attn_lif 的输入尺度（pre-scale，数学等价）；
  - 两个 matmul 暂用 torch.bmm（cutlass），稀疏感知留到 v1。

对拍：捕获真实 block.0.attn 的真实输入 → ref(x) vs fast(x)，用 snn_compiler.verify 比对。
跑法（A100, triton-src）：cd ~/charlley/snn_compiler_attn && SJ_NEURON_BACKEND=torch python snn_compiler/explore/attention/phase2_v0_correctness.py
"""
import os, sys
import torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.expanduser("~/charlley/snn_infer_triton"))
import timm_compat  # noqa
import sj_compat  # noqa
import _bench_util as BU
from snn_compiler.kernels.fused import fused_bias_if_lif, fold_conv_bn
from snn_compiler.verify import compare_models

dev = "cuda"
CK_SF = "/home/liushifeng/lsf/checkpoints/spikingformer_8_768/checkpoint-284.pth.tar"


def lif_params(node):
    """从 SJ LIFNode 抽 tau / v_threshold（v_reset=0, hard, decay_input=True）。"""
    tau = float(getattr(node, "tau", 2.0))
    vth = float(getattr(node, "v_threshold", 1.0))
    return tau, vth


def tlif(x5d, bias, node, input_premul=1.0):
    """对 [T,B,C,N,1] 张量跑 snn_compiler Triton LIF；可选把输入预乘一个常数(=折 scale)。"""
    tau, vth = lif_params(node)
    if input_premul != 1.0:
        x5d = x5d * input_premul
    return fused_bias_if_lif(x5d.contiguous(), bias, neuron="lif", tau=tau,
                             decay_input=True, soft_reset=False,
                             v_threshold=vth, v_reset=0.0, layout="NCHW")


def fold_conv1d_bn(conv, bn):
    """折叠 Conv1d(1×1) + BatchNorm1d → (W'[C,Cin,1], b'[C])。conv 可有/无 bias。"""
    inv = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + bn.eps)   # [C]
    w = conv.weight.detach() * inv.view(-1, 1, 1)                              # [C,Cin,1]
    cb = conv.bias.detach() if conv.bias is not None else torch.zeros_like(bn.bias)
    b = (cb - bn.running_mean.detach()) * inv + bn.bias.detach()              # [C]
    return w, b.float().contiguous()


def folded_conv1d_lif(xf, conv, bn, lif, T, B, C, N, fold_bn=True):
    """xf:[TB,C,N] -> conv1d(+BN) -> LIF -> spike [T,B,C,N]。"""
    if fold_bn:
        w, bias = fold_conv1d_bn(conv, bn)
        y = F.conv1d(xf, w, bias=None)              # [TB,C,N]
    else:
        y = bn(conv(xf))
        bias = None
    y = y.reshape(T, B, C, N).unsqueeze(-1)         # [T,B,C,N,1]
    s = tlif(y, bias, lif)
    return s.squeeze(-1)                            # [T,B,C,N]


@torch.no_grad()
def fast_ssa(ref, x, fold_bn=True):
    T, B, C, H, W = x.shape
    N = H * W
    heads = ref.num_heads
    d = C // heads
    # 输入 LIF（proj_lif）
    x = tlif(x, None, ref.proj_lif).reshape(T, B, C, H, W)
    x = x.flatten(3)                                # [T,B,C,N]
    xf = x.flatten(0, 1)                            # [TB,C,N]

    def to_heads(s):                                # [T,B,C,N] -> [T,B,heads,N,d]
        return (s.transpose(-1, -2).reshape(T, B, N, heads, d)
                 .permute(0, 1, 3, 2, 4).contiguous())

    q = to_heads(folded_conv1d_lif(xf, ref.q_conv, ref.q_bn, ref.q_lif, T, B, C, N, fold_bn))
    k = to_heads(folded_conv1d_lif(xf, ref.k_conv, ref.k_bn, ref.k_lif, T, B, C, N, fold_bn))
    v = to_heads(folded_conv1d_lif(xf, ref.v_conv, ref.v_bn, ref.v_lif, T, B, C, N, fold_bn))

    kv = k.transpose(-2, -1) @ v                    # [T,B,heads,d,d]
    a = q @ kv                                      # [T,B,heads,N,d]  (未乘 scale)
    a_map = a.transpose(3, 4).reshape(T, B, C, N).unsqueeze(-1)   # [T,B,C,N,1]
    # attn_lif：scale 折进输入尺度
    s = tlif(a_map, None, ref.attn_lif, input_premul=ref.scale).squeeze(-1)   # [T,B,C,N]

    sf = s.flatten(0, 1)                            # [TB,C,N]
    if fold_bn:
        w, b = fold_conv1d_bn(ref.proj_conv, ref.proj_bn)
        out = F.conv1d(sf, w, bias=b)               # proj 后无 LIF
    else:
        out = ref.proj_bn(ref.proj_conv(sf))
    return out.reshape(T, B, C, H, W)


def build_sf():
    sys.path.insert(0, os.path.expanduser("~/charlley/snn_infer/repos/Spikingformer/imagenet"))
    import model as SF
    m = SF.vit_snn(img_size_h=224, img_size_w=224, patch_size=16, in_channels=3,
                   num_classes=1000, embed_dims=768, num_heads=8, mlp_ratios=4,
                   qkv_bias=False, depths=8, sr_ratios=1, T=4)
    sd = torch.load(CK_SF, map_location="cpu", weights_only=False)["state_dict"]
    m.load_state_dict(sd, strict=False)
    return m.to(dev).eval()


def main():
    BU.gpu_guard(tag="P2v0-start")
    model = build_sf()
    # 抓 block.0.attn 的真实输入
    attn_blocks = [(n, m) for n, m in model.named_modules()
                   if type(m).__name__ == "SpikingSelfAttention"]
    bname, ref = attn_blocks[0]
    cap = {}
    h = ref.register_forward_hook(lambda m, i, o: cap.__setitem__("x", i[0].detach()))
    from spikingjelly.activation_based import functional
    with torch.no_grad():
        model(torch.randn(8, 3, 224, 224, device=dev))
    h.remove()
    x = cap["x"]
    print(f"[P2v0] captured input to {bname}: shape={tuple(x.shape)}")

    # 参考输出（确保 SNN 状态干净）
    functional.reset_net(model)
    with torch.no_grad():
        ref_out = ref(x)

    for fold_bn in (False, True):
        functional.reset_net(model)
        wrap_ref = lambda xx: ref(xx)
        wrap_fast = lambda xx: fast_ssa(ref, xx, fold_bn=fold_bn)
        # 用 compare_models 比对（它内部各跑一次）
        class W(torch.nn.Module):
            def __init__(s, fn): super().__init__(); s.fn = fn
            def forward(s, xx): return s.fn(xx)
        rep = compare_models(W(wrap_ref), W(wrap_fast), x)
        print(f"\n[P2v0 fold_bn={fold_bn}] max|Δ|={rep['max_abs_diff']:.3e} "
              f"rel={rep['rel_max']:.3e} bit_exact={rep['bit_exact']} "
              f"spike_mismatch={rep['spike_mismatch_frac']}")

    # 测速：ref(eager) vs fast(fold_bn=True)
    functional.reset_net(model)
    r_ref = BU.bench(lambda: (functional.reset_net(ref), ref(x))[1], warmup=15, iters=60)
    r_fast = BU.bench(lambda: fast_ssa(ref, x, fold_bn=True), warmup=25, iters=100)
    print(f"\n[P2v0 speed] ref eager : {BU.fmt(r_ref)}")
    print(f"[P2v0 speed] fast v0   : {BU.fmt(r_fast)}")
    print(f"[P2v0 speed] speedup   : {r_ref['median_ms']/r_fast['median_ms']:.2f}x")
    BU.gpu_guard(tag="P2v0-end")
    print("\nP2v0_DONE")


if __name__ == "__main__":
    main()
