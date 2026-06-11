import os, sys, argparse, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import snn2_lib as L2
import snn_eval_lib_triton as L          # ImageNet val eval + profiler (shared)
import ms_resnet_triton as M

CKPT = "/home/liushifeng/lsf/checkpoints/ms_resnet104_imagenet/resnet104.pth"

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--bs", type=int, default=25)
ap.add_argument("--compile", action="store_true")
ap.add_argument("--profile", action="store_true")
ap.add_argument("--amp", default="none", choices=["none", "bf16", "fp16"])
ap.add_argument("--triton-conv", dest="triton_conv", action="store_true")
a = ap.parse_args()

if a.compile:
    L2.setup_inductor_triton(triton_conv=a.triton_conv)

m = M.build_msresnet104(CKPT).cuda()
print("[MS-ResNet-104] backend=%s built; T=6" % M.BACKEND)
tag = "MS-ResNet-104 [%s%s%s]" % (M.BACKEND, "+compile" if a.compile else "", "+" + a.amp if a.amp != "none" else "")
if a.compile:
    m = torch.compile(m, mode="max-autotune-no-cudagraphs")
if a.profile:
    L.profile_kernels(m, transform_kind="r256_bilinear", batch_size=a.bs, amp=a.amp)
L.evaluate(m, tag, n=a.n, batch_size=a.bs, transform_kind="r256_bilinear", amp=a.amp)
