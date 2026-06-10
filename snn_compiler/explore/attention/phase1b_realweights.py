"""P1b 刻画（真实权重）：加载 Spikingformer-8-768 / SDT-V2-55M 的训练权重，跑一个 batch，
hook 每个注意力块，测真实的 Q/K/V/attn 发放率与各块维度；并把某一代表块的真实
(input, q, k, v 脉冲) 存盘，供后续 kernel 开发/对拍用真实脉冲分布。

跑法（A100, triton-src）：
    cd ~/charlley/snn_compiler_attn && SJ_NEURON_BACKEND=torch \
        python snn_compiler/explore/attention/phase1b_realweights.py
"""
import os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser("~/charlley/snn_infer_triton"))
import timm_compat   # noqa
import sj_compat     # noqa
import _bench_util as BU

dev = "cuda"
CK = {
    "sf":   "/home/liushifeng/lsf/checkpoints/spikingformer_8_768/checkpoint-284.pth.tar",
    "sdtv2":"/home/liushifeng/lsf/checkpoints/spike_driven_v2_metaspikeformer/55M_kd_T4.pth",
}


def build_sf():
    sys.path.insert(0, os.path.expanduser("~/charlley/snn_infer/repos/Spikingformer/imagenet"))
    import model as SF
    m = SF.vit_snn(img_size_h=224, img_size_w=224, patch_size=16, in_channels=3,
                   num_classes=1000, embed_dims=768, num_heads=8, mlp_ratios=4,
                   qkv_bias=False, depths=8, sr_ratios=1, T=4)
    sd = torch.load(CK["sf"], map_location="cpu", weights_only=False)["state_dict"]
    miss, unexp = m.load_state_dict(sd, strict=False)
    print(f"[sf] load missing={len(miss)} unexpected={len(unexp)}")
    return m.to(dev).eval(), "SpikingSelfAttention"


def build_sdtv2():
    sys.path.insert(0, os.path.expanduser("~/charlley/snn_infer/repos/Spike-Driven-Transformer-V2/classification"))
    import models as M
    m = M.metaspikformer_8_512(kd=True)
    m.T = 4
    sd = torch.load(CK["sdtv2"], map_location="cpu", weights_only=False)["model"]
    miss, unexp = m.load_state_dict(sd, strict=False)
    print(f"[sdtv2] load missing={len(miss)} unexpected={len(unexp)}")
    return m.to(dev).eval(), "MS_Attention_RepConv_qkv_id"


def characterize(model, attn_clsname, tag, B=8, save_block=None):
    # 找到所有注意力块
    attn_mods = [(n, m) for n, m in model.named_modules()
                 if type(m).__name__ == attn_clsname]
    print(f"[{tag}] #attention blocks = {len(attn_mods)}")
    rec = {}           # block_name -> {lif: rate, dims:...}
    captured = {}
    handles = []
    for bname, blk in attn_mods:
        for lif in ("q_lif", "k_lif", "v_lif", "attn_lif"):
            sub = getattr(blk, lif, None)
            if sub is None:
                continue
            def mk(bn, ln):
                def hook(m, i, o):
                    t = o if isinstance(o, torch.Tensor) else o[0]
                    rec.setdefault(bn, {})[ln] = float(t.float().mean().item())
                return hook
            handles.append(sub.register_forward_hook(mk(bname, lif)))
        # 抓某一块的输入与 q/k/v 脉冲
        if save_block is not None and bname == save_block:
            def cap(m, i, o):
                captured["x"] = i[0].detach().to("cpu")
            handles.append(blk.register_forward_hook(cap))

    x = torch.randn(B, 3, 224, 224, device=dev)
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()

    # 汇总发放率（跨块均值）
    import statistics as st
    agg = {}
    for ln in ("q_lif", "k_lif", "v_lif", "attn_lif"):
        vals = [d[ln] for d in rec.values() if ln in d]
        if vals:
            agg[ln] = (st.mean(vals), min(vals), max(vals))
    print(f"[{tag}] firing rate (mean[min,max] across blocks):")
    for ln, (mu, lo, hi) in agg.items():
        print(f"     {ln:9s}= {mu:.3f}  [{lo:.3f}, {hi:.3f}]")
    return rec, agg, captured


def main():
    BU.gpu_guard(tag="P1b-start")
    print("=" * 70)
    sf, sfc = build_sf()
    # block name 形如 'block.0.attn' ；抓第一块
    sf_blocks = [n for n, m in sf.named_modules() if type(m).__name__ == sfc]
    characterize(sf, sfc, "Spikingformer", B=8, save_block=(sf_blocks[3] if len(sf_blocks) > 3 else sf_blocks[0]))
    del sf; torch.cuda.empty_cache()

    print("=" * 70)
    sd, sdc = build_sdtv2()
    sd_blocks = [n for n, m in sd.named_modules() if type(m).__name__ == sdc]
    characterize(sd, sdc, "SDT-V2", B=8, save_block=(sd_blocks[0] if sd_blocks else None))
    del sd; torch.cuda.empty_cache()

    BU.gpu_guard(tag="P1b-end")
    print("\nP1b_DONE")


if __name__ == "__main__":
    main()
