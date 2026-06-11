import os, sys, argparse, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import snn2_lib as L
import ottt_vgg_triton as M

CKPT = "/home/liushifeng/lsf/checkpoints/ottt_vgg11_cifar10/cifar10_ottta.pth"
DATA = os.environ.get("CIFAR_DIR", "/home/liushifeng/charlley/snn_infer/data/cifar10")

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=10000)
ap.add_argument("--bs", type=int, default=100)
ap.add_argument("--compile", action="store_true")
ap.add_argument("--profile", action="store_true")
ap.add_argument("--amp", default="none", choices=["none", "bf16", "fp16"])
ap.add_argument("--triton-conv", dest="triton_conv", action="store_true")
a = ap.parse_args()

if a.compile:
    L.setup_inductor_triton(triton_conv=a.triton_conv)

m = M.build_ottt(CKPT).cuda()
print("[OTTT VGG-11-WS] backend=%s built; T=%d" % (M.BACKEND, m.T))
tag = "OTTT-VGG11-WS [%s%s%s]" % (M.BACKEND, "+compile" if a.compile else "", "+" + a.amp if a.amp != "none" else "")
if a.compile:
    m = torch.compile(m, mode="max-autotune-no-cudagraphs")
if a.profile:
    ds = L.cifar10_testset(DATA)
    x = torch.stack([ds[i][0] for i in range(a.bs)]).cuda()
    L.profile_kernels_x(m, x, amp=a.amp)
L.evaluate_cifar(m, tag, DATA, n=a.n, batch_size=a.bs, amp=a.amp)
