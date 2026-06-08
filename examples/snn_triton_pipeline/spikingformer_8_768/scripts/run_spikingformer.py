import os, sys, argparse, torch
BASE = os.path.expanduser("~/charlley/snn_infer")
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "repos/Spikingformer/imagenet"))
import snn_eval_lib as L
import model as SF

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--bs", type=int, default=50)
ap.add_argument("--tf", default="r224_bicubic")  # ckpt args: crop_pct=1.0, bicubic
a = ap.parse_args()

CKPT = "/home/liushifeng/lsf/checkpoints/spikingformer_8_768/checkpoint-284.pth.tar"
sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
# Spikingformer-8-768: dim=768, depths=8, heads=8, patch16, T=4 (matches ckpt args)
m = SF.vit_snn(img_size_h=224, img_size_w=224, patch_size=16, in_channels=3, num_classes=1000,
               embed_dims=768, num_heads=8, mlp_ratios=4, qkv_bias=False,
               depths=8, sr_ratios=1, T=4)
miss, unexp = m.load_state_dict(sd, strict=False)
L.report_load("Spikingformer-8-768", miss, unexp)
L.evaluate(m, "Spikingformer-8-768", n=a.n, batch_size=a.bs, transform_kind=a.tf)
