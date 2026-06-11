"""Capture REAL compilation artifacts for one model on the Triton path.
Run with TRITON_CACHE_DIR / TORCHINDUCTOR_CACHE_DIR / TORCH_COMPILE_DEBUG[_DIR] set
in the environment so all IR lands in known dirs.
"""
import os, sys, argparse, torch
BASE = "/home/liushifeng/charlley/snn_infer"
TB = "/home/liushifeng/charlley/snn_infer_triton"
sys.path.insert(0, TB)
import timm_compat            # noqa: F401
import sj_compat              # noqa: F401  -> neurons forced to step_mode='m', backend='triton'
import snn_eval_lib_triton as L

ap = argparse.ArgumentParser()
ap.add_argument("model", choices=["sew", "sf", "sdtv2"])
ap.add_argument("--bs", type=int, default=8)
a = ap.parse_args()

torch.backends.cudnn.enabled = False
import torch._inductor.config as ic
ic.max_autotune = True
ic.conv_1x1_as_mm = True
ic.compile_threads = 1
ic.max_autotune_gemm_backends = "TRITON"
ic.max_autotune_conv_backends = "ATEN,TRITON"

CK = {
    "sew": "/home/liushifeng/lsf/checkpoints/sew_resnet34_imagenet/sew34_checkpoint_319.pth",
    "sf": "/home/liushifeng/lsf/checkpoints/spikingformer_8_768/checkpoint-284.pth.tar",
    "sdtv2": "/home/liushifeng/lsf/checkpoints/spike_driven_v2_metaspikeformer/55M_kd_T4.pth",
}


def build(name):
    if name == "sew":
        sys.path.insert(0, os.path.join(BASE, "repos/Spike-Element-Wise-ResNet/imagenet"))
        import sew_resnet
        sd = torch.load(CK[name], map_location="cpu", weights_only=False)["model"]
        m = sew_resnet.sew_resnet34(T=4, connect_f="ADD", num_classes=1000, zero_init_residual=False)
        if not any(".module." in k for k in m.state_dict()):
            sd = {k.replace(".module.", "."): v for k, v in sd.items()}
        m.load_state_dict(sd, strict=False)
        return m.cuda().eval(), "r256_bilinear"
    if name == "sf":
        sys.path.insert(0, os.path.join(BASE, "repos/Spikingformer/imagenet"))
        import model as SF
        sd = torch.load(CK[name], map_location="cpu", weights_only=False)["state_dict"]
        m = SF.vit_snn(img_size_h=224, img_size_w=224, patch_size=16, in_channels=3, num_classes=1000,
                       embed_dims=768, num_heads=8, mlp_ratios=4, qkv_bias=False, depths=8, sr_ratios=1, T=4)
        m.load_state_dict(sd, strict=False)
        return m.cuda().eval(), "r224_bicubic"
    if name == "sdtv2":
        sys.path.insert(0, os.path.join(BASE, "repos/Spike-Driven-Transformer-V2/classification"))
        import models as M
        sd = torch.load(CK[name], map_location="cpu", weights_only=False)["model"]
        m = M.metaspikformer_8_512(kd=True)
        m.T = 4
        m.load_state_dict(sd, strict=False)
        return m.cuda().eval(), "r256_bicubic"


m, tf = build(a.model)
m = torch.compile(m, mode="max-autotune-no-cudagraphs")
ds = L.ValSet(a.bs, L.make_transform(tf))
x = torch.stack([ds[i][0] for i in range(len(ds))]).cuda()
with torch.no_grad(), L._amp_ctx("bf16"):
    y = m(x)
    torch.cuda.synchronize()
print("CAPTURE_DONE", a.model, "out", tuple((y[0] if isinstance(y, (tuple, list)) else y).shape))
