"""P6 泛化到 SDT-V2 MS_Attention + P4 确定性基线。

1. 确定性基线：原模型同输入连跑两次 |ref−ref'|（量化未替换 conv/MLP 的 cutlass 非确定性）。
2. 单块：FusedSpikeAttention 对 MS_Attention 是否逐位一致（与 ref 比，且对照 ref−ref' 噪声）。
3. 全模型替换 → 端到端对比 + top-1。
4. 测速：单块 eager vs fused。
跑法（A100, triton-src）：cd ~/charlley/snn_compiler_attn && SJ_NEURON_BACKEND=torch python snn_compiler/explore/attention/phase6_sdtv2.py
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

from snn_compiler.nn.attention import FusedSpikeAttention, is_ms_attention
from snn_compiler.passes import fuse_spiking_attention
from snn_compiler.verify import compare_models

dev = "cuda"
CK = "/home/liushifeng/lsf/checkpoints/spike_driven_v2_metaspikeformer/55M_kd_T4.pth"


def build():
    sys.path.insert(0, os.path.expanduser("~/charlley/snn_infer/repos/Spike-Driven-Transformer-V2/classification"))
    import models as M
    m = M.metaspikformer_8_512(kd=True)
    m.T = 4
    sd = torch.load(CK, map_location="cpu", weights_only=False)["model"]
    miss, unexp = m.load_state_dict(sd, strict=False)
    print(f"[sdtv2] load missing={len(miss)} unexpected={len(unexp)}")
    return m.to(dev).eval()


def main():
    from spikingjelly.activation_based import functional
    BU.gpu_guard(tag="P6sdt-start")
    model = build()
    blocks = [(n, m) for n, m in model.named_modules() if is_ms_attention(m)]
    print(f"[P6] detected {len(blocks)} MS_Attention blocks")
    if not blocks:
        print("NO MS_ATTENTION FOUND"); return
    bname, ref = blocks[0]

    cap = {}
    h = ref.register_forward_hook(lambda m, i, o: cap.__setitem__("x", i[0].detach()))
    with torch.no_grad():
        model(torch.randn(4, 3, 224, 224, device=dev))
    h.remove()
    x = cap["x"]
    print(f"[P6] captured input {tuple(x.shape)} to {bname}")

    # 1) 单块：ref vs ref'(噪声基线) vs fused
    functional.reset_net(model)
    with torch.no_grad():
        o1 = ref(x)
    functional.reset_net(model)
    with torch.no_grad():
        o2 = ref(x)
    base_noise = (o1 - o2).abs().max().item()
    fused = FusedSpikeAttention.from_reference(ref, fold_bn=False).to(dev).eval()
    rep = compare_models(ref, fused, x)
    print(f"[P6] single-block: |ref-ref'|(noise)={base_noise:.3e}  |ref-fused|={rep['max_abs_diff']:.3e}  "
          f"bit_exact={rep['bit_exact']}")

    # 2) 全模型替换 → 端到端
    xb = torch.randn(4, 3, 224, 224, device=dev)
    functional.reset_net(model)
    with torch.no_grad():
        out_ref = model(xb)
    out_ref = out_ref[0] if isinstance(out_ref, (tuple, list)) else out_ref
    functional.reset_net(model)
    with torch.no_grad():
        out_ref2 = model(xb)
    out_ref2 = out_ref2[0] if isinstance(out_ref2, (tuple, list)) else out_ref2
    full_noise = (out_ref - out_ref2).abs().max().item()
    n = fuse_spiking_attention(model, fold_bn=False)
    functional.reset_net(model)
    with torch.no_grad():
        out_fused = model(xb)
    out_fused = out_fused[0] if isinstance(out_fused, (tuple, list)) else out_fused
    dmax = (out_ref - out_fused).abs().max().item()
    agree = (out_ref.argmax(-1) == out_fused.argmax(-1)).float().mean().item()
    print(f"[P6] FULL MODEL fused {n} blocks: |ref-ref'|(noise)={full_noise:.3e}  "
          f"|ref-fused|={dmax:.3e}  top1-agree={agree*100:.2f}%")

    # 3) 测速单块
    functional.reset_net(model)
    r_eager = BU.bench(lambda: (functional.reset_net(ref), ref(x))[1], warmup=15, iters=60)
    r_fused = BU.bench(lambda: fused(x), warmup=25, iters=100)
    print(f"\n[P6 speed single MS_Attention block]")
    print(f"   eager : {BU.fmt(r_eager)}")
    print(f"   fused : {BU.fmt(r_fused)}  => {r_eager['median_ms']/r_fused['median_ms']:.2f}x")
    BU.gpu_guard(tag="P6sdt-end")
    print("\nP6sdt_DONE")


if __name__ == "__main__":
    main()
