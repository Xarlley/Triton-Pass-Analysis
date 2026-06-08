import os, sys, argparse, torch
BASE = "/home/liushifeng/charlley/snn_infer"
TBASE = "/home/liushifeng/charlley/snn_infer_triton"
sys.path.insert(0, TBASE)
import timm_compat          # noqa: F401  (not needed for SEW, harmless)
import sj_compat            # noqa: F401  installs shim -> triton neurons (BEFORE repo import)
sys.path.insert(0, os.path.join(BASE, "repos/Spike-Element-Wise-ResNet/imagenet"))
import snn_eval_lib_triton as L
import sew_resnet

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--bs", type=int, default=100)
ap.add_argument("--compile", action="store_true")
ap.add_argument("--profile", action="store_true")
ap.add_argument("--amp", default="none", choices=["none","bf16","fp16"])
ap.add_argument("--triton-conv", dest="triton_conv", action="store_true")
a = ap.parse_args()

if a.compile:
    torch.backends.cudnn.enabled = False
    import torch._inductor.config as ic
    ic.max_autotune = True
    ic.conv_1x1_as_mm = True
    ic.compile_threads = 1                          # compile in-process: source-built triton driver fails in subproc workers
    ic.max_autotune_gemm_backends = "TRITON"        # matmul/linear/1x1-conv(as mm) -> triton only (no cublas)
    ic.max_autotune_conv_backends = "ATEN,TRITON" if a.triton_conv else "ATEN"  # bf16 halves smem -> triton conv can fit
    print("compile_threads=1 gemm=TRITON conv=ATEN conv_1x1_as_mm=True cudnn=", __import__("torch").backends.cudnn.enabled)

CK = "/home/liushifeng/lsf/checkpoints/sew_resnet34_imagenet/sew34_checkpoint_319.pth"
sd = torch.load(CK, map_location="cpu", weights_only=False)["model"]
m = sew_resnet.sew_resnet34(T=4, connect_f="ADD", num_classes=1000, zero_init_residual=False)
# Align SeqToANNContainer key style: ckpt has '.module.'; strip it iff THIS spikingjelly
# stores wrapped conv/bn directly (no '.module.' in the model's own keys).
mk = list(m.state_dict().keys())
if not any(".module." in k for k in mk):
    sd = {k.replace(".module.", "."): v for k, v in sd.items()}
miss, unexp = m.load_state_dict(sd, strict=False)
L.report_load("SEW-ResNet-34 [triton]", miss, unexp)
m = m.cuda().eval()
tag = "SEW-ResNet-34 [triton%s]" % ("+compile" if a.compile else "")
if a.compile:
    m = torch.compile(m, mode="max-autotune-no-cudagraphs")
if a.profile:
    L.profile_kernels(m, transform_kind="r256_bilinear", batch_size=a.bs, amp=a.amp)
tag = tag.replace("]", "+%s]" % a.amp) if a.amp!="none" else tag
L.evaluate(m, tag, n=a.n, batch_size=a.bs, transform_kind="r256_bilinear", amp=a.amp)
