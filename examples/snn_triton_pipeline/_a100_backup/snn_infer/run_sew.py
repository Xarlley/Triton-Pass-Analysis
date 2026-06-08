import os, sys, argparse, torch
BASE = os.path.expanduser("~/charlley/snn_infer")
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "repos/Spike-Element-Wise-ResNet/imagenet"))
import snn_eval_lib as L
import sew_resnet

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--bs", type=int, default=100)
a = ap.parse_args()

CKPT = "/home/liushifeng/lsf/checkpoints/sew_resnet34_imagenet/sew34_checkpoint_319.pth"
sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]
# Older-spikingjelly SeqToANNContainer stored wrapped conv/bn under '.module.';
# this spikingjelly's SeqToANNContainer is an nn.Sequential subclass ('.0','.1'). Same math, rename keys.
sd = {k.replace(".module.", "."): v for k, v in sd.items()}
m = sew_resnet.sew_resnet34(T=4, connect_f="ADD", num_classes=1000, zero_init_residual=False)
miss, unexp = m.load_state_dict(sd, strict=False)
L.report_load("SEW-ResNet-34 (connect_f=ADD)", miss, unexp)
L.evaluate(m, "SEW-ResNet-34", n=a.n, batch_size=a.bs, transform_kind="r256_bilinear")
