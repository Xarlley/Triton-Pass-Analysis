"""P5 集成验证：FusedSpikeAttention + fuse_spiking_attention 探测 pass。

1. 单块逐位一致：assert_equivalent(ref_block, FusedSpikeAttention.from_reference(ref_block), x)。
2. 全模型替换：fuse_spiking_attention(model) 换掉全部 8 个注意力块 → 端到端输出与原模型对比。
3. 公平测速：单块 eager vs fused vs torch.compile(eager)。
跑法（A100, triton-src）：cd ~/charlley/snn_compiler_attn && SJ_NEURON_BACKEND=torch python snn_compiler/explore/attention/phase5_integration.py
"""
import os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.expanduser("~/charlley/snn_infer_triton"))
import timm_compat  # noqa
import sj_compat    # noqa
import _bench_util as BU

from snn_compiler.nn.attention import FusedSpikeAttention, is_spiking_self_attention
from snn_compiler.passes import fuse_spiking_attention
from snn_compiler.verify import assert_equivalent, compare_models

dev = "cuda"
CK_SF = "/home/liushifeng/lsf/checkpoints/spikingformer_8_768/checkpoint-284.pth.tar"


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
    from spikingjelly.activation_based import functional
    BU.gpu_guard(tag="P5-start")
    model = build_sf()

    blocks = [(n, m) for n, m in model.named_modules() if is_spiking_self_attention(m)]
    print(f"[P5] detected {len(blocks)} spiking-attention blocks (duck-type)")
    bname, ref = blocks[0]

    # 捕获真实输入
    cap = {}
    h = ref.register_forward_hook(lambda m, i, o: cap.__setitem__("x", i[0].detach()))
    with torch.no_grad():
        model(torch.randn(8, 3, 224, 224, device=dev))
    h.remove()
    x = cap["x"]
    print(f"[P5] captured input {tuple(x.shape)} to {bname}")

    # 1) 单块逐位一致
    fused_block = FusedSpikeAttention.from_reference(ref, fold_bn=False).to(dev).eval()
    functional.reset_net(model)
    rep = assert_equivalent(ref, fused_block, x)   # 默认要求逐位一致
    print(f"[P5] single-block bit-exact OK: max|Δ|={rep['max_abs_diff']:.2e}")

    # 2) 全模型替换 → 端到端对比
    xb = torch.randn(8, 3, 224, 224, device=dev)
    functional.reset_net(model)
    with torch.no_grad():
        out_ref = model(xb)
    out_ref = out_ref[0] if isinstance(out_ref, (tuple, list)) else out_ref
    n = fuse_spiking_attention(model, fold_bn=False)
    print(f"[P5] fuse_spiking_attention replaced {n} blocks")
    functional.reset_net(model)
    with torch.no_grad():
        out_fused = model(xb)
    out_fused = out_fused[0] if isinstance(out_fused, (tuple, list)) else out_fused
    dmax = (out_ref - out_fused).abs().max().item()
    agree = (out_ref.argmax(-1) == out_fused.argmax(-1)).float().mean().item()
    print(f"[P5] FULL MODEL (8 blocks fused): max|Δ|={dmax:.3e}  top1-agree={agree*100:.2f}%")

    # 3) 公平测速：单块
    functional.reset_net(model)
    r_eager = BU.bench(lambda: (functional.reset_net(ref), ref(x))[1], warmup=15, iters=60)
    r_fused = BU.bench(lambda: fused_block(x), warmup=25, iters=100)
    print(f"\n[P5 speed single block]")
    print(f"   eager (SJ torch) : {BU.fmt(r_eager)}")
    print(f"   FusedSpikeAttn   : {BU.fmt(r_fused)}  => {r_eager['median_ms']/r_fused['median_ms']:.2f}x")
    try:
        import torch._inductor.config as ic
        ic.compile_threads = 1                 # 源码 triton 在 inductor 子进程编不过（A100 文档）
        cblock = torch.compile(ref, mode="max-autotune-no-cudagraphs")
        functional.reset_net(ref)
        r_comp = BU.bench(lambda: (functional.reset_net(ref), cblock(x))[1], warmup=20, iters=60)
        print(f"   torch.compile    : {BU.fmt(r_comp)}  => fused is {r_comp['median_ms']/r_fused['median_ms']:.2f}x vs compile")
    except Exception as e:
        print(f"   torch.compile    : FAILED ({type(e).__name__}: {str(e)[:80]})")

    BU.gpu_guard(tag="P5-end")
    print("\nP5_DONE")


if __name__ == "__main__":
    main()
