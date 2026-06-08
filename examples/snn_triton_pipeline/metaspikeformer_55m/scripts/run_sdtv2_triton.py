import os, sys, argparse, torch
BASE = "/home/liushifeng/charlley/snn_infer"
TBASE = "/home/liushifeng/charlley/snn_infer_triton"
sys.path.insert(0, TBASE)
import timm_compat          # noqa: F401
import sj_compat            # noqa: F401
sys.path.insert(0, os.path.join(BASE, "repos/Spike-Driven-Transformer-V2/classification"))
import snn_eval_lib_triton as L
import models as M

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--bs", type=int, default=32)
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

CK = "/home/liushifeng/lsf/checkpoints/spike_driven_v2_metaspikeformer/55M_kd_T4.pth"
sd = torch.load(CK, map_location="cpu", weights_only=False)["model"]
m = M.metaspikformer_8_512(kd=True)
m.T = 4
miss, unexp = m.load_state_dict(sd, strict=False)
L.report_load("Meta-SpikeFormer-55M [triton]", miss, unexp)
m = m.cuda().eval()
tag = "Meta-SpikeFormer-55M [triton%s]" % ("+compile" if a.compile else "")
if a.compile:
    m = torch.compile(m, mode="max-autotune-no-cudagraphs")
if a.profile:
    L.profile_kernels(m, transform_kind="r256_bicubic", batch_size=a.bs, amp=a.amp)
tag = tag.replace("]", "+%s]" % a.amp) if a.amp!="none" else tag
L.evaluate(m, tag, n=a.n, batch_size=a.bs, transform_kind="r256_bicubic", amp=a.amp)
