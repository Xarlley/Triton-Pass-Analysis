import os, sys, argparse, torch
BASE = os.path.expanduser("~/charlley/snn_infer")
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "repos/Spike-Driven-Transformer-V2/classification"))
import snn_eval_lib as L
import models as M

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--bs", type=int, default=32)
a = ap.parse_args()

CKPT = "/home/liushifeng/lsf/checkpoints/spike_driven_v2_metaspikeformer/55M_kd_T4.pth"
sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]
# 55M = metaspikformer_8_512 (embed [128,256,512,640]); KD head present; T set after build.
m = M.metaspikformer_8_512(kd=True)
m.T = 4
miss, unexp = m.load_state_dict(sd, strict=False)
L.report_load("Meta-SpikeFormer 55M (SDT-V2)", miss, unexp)
L.evaluate(m, "Meta-SpikeFormer-55M", n=a.n, batch_size=a.bs, transform_kind="r256_bicubic")
