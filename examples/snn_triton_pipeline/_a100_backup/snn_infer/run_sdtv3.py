import os, sys, argparse, torch
BASE = os.path.expanduser("~/charlley/snn_infer")
MLD = os.path.join(BASE, "repos/Spike-Driven-Transformer-V3/SDT_V3/Classification/Model_Large")
sys.path.insert(0, BASE)
sys.path.insert(0, MLD)
import snn_eval_lib as L
import spikformer as S

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--mode", default="drop", choices=["drop", "repconv"])
a = ap.parse_args()

CKPT = "/home/liushifeng/lsf/checkpoints/spike_driven_v3_espikeformer/83M_1x4.pth"
sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]
# 83M "1x4": spikformer12_512 (choice=base, embed [128,256,512], depths=12, T=1, multispike lens=4)
if a.mode == "repconv":
    S.RepConv2 = S.RepConv  # make block rep_conv a 512->768->512 RepConv
m = S.spikformer12_512()
msd = m.state_dict()
# drop checkpoint keys whose shape disagrees with the model (the block rep_conv.conv1.1 BN-768 anomaly)
dropped = [k for k in sd if k in msd and tuple(sd[k].shape) != tuple(msd[k].shape)]
sd2 = {k: v for k, v in sd.items() if k not in dropped}
miss, unexp = m.load_state_dict(sd2, strict=False)
print("[SDT-V3] dropped %d shape-mismatched ckpt keys, e.g. %s" % (len(dropped), dropped[:2]))
L.report_load("E-SpikeFormer 83M (SDT-V3)", miss, unexp)
L.evaluate(m, "E-SpikeFormer-83M", n=a.n, batch_size=a.bs, transform_kind="r256_bicubic")
