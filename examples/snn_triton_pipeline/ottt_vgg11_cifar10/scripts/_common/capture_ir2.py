"""Capture REAL compilation artifacts for ottt / msresnet (Triton path, bf16+compile).
Run with TRITON_CACHE_DIR / TORCHINDUCTOR_CACHE_DIR / TORCH_COMPILE_DEBUG[_DIR] set.
"""
import os, sys, argparse, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import snn2_lib as L

ap = argparse.ArgumentParser()
ap.add_argument("model", choices=["ottt", "msresnet"])
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--triton-conv", dest="triton_conv", action="store_true", default=True)
a = ap.parse_args()

L.setup_inductor_triton(triton_conv=a.triton_conv)

if a.model == "ottt":
    import ottt_vgg_triton as M
    m = M.build_ottt("/home/liushifeng/lsf/checkpoints/ottt_vgg11_cifar10/cifar10_ottta.pth").cuda()
    ds = L.cifar10_testset(os.environ.get("CIFAR_DIR", "/home/liushifeng/charlley/snn_infer/data/cifar10"))
    x = torch.stack([ds[i][0] for i in range(a.bs)]).cuda()
else:
    import ms_resnet_triton as M
    m = M.build_msresnet104("/home/liushifeng/lsf/checkpoints/ms_resnet104_imagenet/resnet104.pth").cuda()
    x = torch.randn(a.bs, 3, 224, 224, device="cuda")   # IR is shape/arch-determined; random input ok for capture

m = torch.compile(m, mode="max-autotune-no-cudagraphs")
with torch.no_grad(), L._amp_ctx("bf16"):
    y = m(x)
    torch.cuda.synchronize()
print("CAPTURE_DONE", a.model, "out", tuple(y.shape))
