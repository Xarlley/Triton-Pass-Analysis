"""Shared helpers for the two new models (OTTT VGG / MS-ResNet).
CIFAR10 eval (OTTT) + model-agnostic CUDA-kernel profiler + inductor-triton config.
ImageNet eval for MS-ResNet reuses snn_eval_lib_triton.py.
"""
import os, time, contextlib, torch
import torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional as AF

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def _amp_ctx(amp):
    if amp == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if amp == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _reset(model):
    AF.reset_net(getattr(model, "_orig_mod", model))


def _classify(nm):
    s = nm.lower()
    if "triton" in s or any(k in s for k in ["multistep_lif", "multistep_if", "multistep_plif"]):
        return "triton"
    if "cudnn" in s:
        return "cudnn"
    if any(k in s for k in ["cutlass", "sgemm", "hgemm", "gemm", "cublas", "ampere_", "cgemm", "dgemm", "gemv"]):
        return "cublas/gemm"
    return "other"


def setup_inductor_triton(triton_conv=False):
    """Whole-net Triton: no cudnn, triton-only GEMM, triton conv where it fits, in-process compile."""
    torch.backends.cudnn.enabled = False
    import torch._inductor.config as ic
    ic.max_autotune = True
    ic.conv_1x1_as_mm = True
    ic.compile_threads = 1
    ic.max_autotune_gemm_backends = "TRITON"
    ic.max_autotune_conv_backends = "ATEN,TRITON" if triton_conv else "ATEN"
    print("[inductor] gemm=TRITON conv=%s conv_1x1_as_mm=True compile_threads=1 cudnn=%s"
          % (ic.max_autotune_conv_backends, torch.backends.cudnn.enabled))


class _PtCifar(torch.utils.data.Dataset):
    """Lossless CIFAR10 test from a decoded .pt (uint8 NHWC + labels, from HF parquet).
    Equivalent to torchvision: uint8 -> /255 -> Normalize(OTTT stats)."""
    def __init__(self, pt):
        d = torch.load(pt, weights_only=False)
        self.x, self.y = d["images"], d["labels"]
        self.mean = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
        self.std = torch.tensor(CIFAR_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = self.x[i].permute(2, 0, 1).float().div_(255.0)
        return (img - self.mean) / self.std, int(self.y[i])


def cifar10_testset(data_dir):
    """CIFAR10 test set, lossless. Prefer decoded .pt; then torchvision batches; then ImageFolder."""
    pt = os.path.join(data_dir, "cifar10_test.pt")
    if os.path.isfile(pt):
        return _PtCifar(pt)
    tf = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
    if os.path.isdir(os.path.join(data_dir, "cifar-10-batches-py")):
        return torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=tf)
    imgdir = os.path.join(data_dir, "imgs", "test")
    if os.path.isdir(imgdir):
        return torchvision.datasets.ImageFolder(imgdir, transform=tf)
    raise FileNotFoundError("no CIFAR10 test data under %s" % data_dir)


@torch.no_grad()
def profile_kernels_x(model, x, amp="none", reset=True):
    from torch.profiler import profile, ProfilerActivity
    model.eval()
    with _amp_ctx(amp):
        model(x)
    if reset: _reset(model)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        with _amp_ctx(amp):
            model(x)
        torch.cuda.synchronize()
    if reset: _reset(model)
    agg = {}
    for e in prof.key_averages():
        us = getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)
        if us <= 0: continue
        c = _classify(e.key); d = agg.setdefault(c, {"us": 0.0, "names": {}})
        d["us"] += us; d["names"][e.key] = d["names"].get(e.key, 0) + us
    total = sum(d["us"] for d in agg.values()) or 1.0
    print("  -- CUDA kernel time by class (one batch) --")
    for c in ["triton", "cublas/gemm", "cudnn", "other"]:
        if c in agg:
            print("     %-12s %9.1f us  (%.1f%%)" % (c, agg[c]["us"], 100 * agg[c]["us"] / total))
            for nm, us in sorted(agg[c]["names"].items(), key=lambda kv: -kv[1])[:3]:
                print("         %9.1f us  %s" % (us, nm[:88]))
    return agg


@torch.no_grad()
def evaluate_cifar(model, name, data_dir, n=10000, batch_size=100, amp="none", reset=True, workers=4, device="cuda"):
    ds = cifar10_testset(data_dir)
    if n < len(ds):
        ds = torch.utils.data.Subset(ds, range(n))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    model.eval().to(device)
    t1 = tot = 0
    st = time.time()
    for x, y in dl:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with _amp_ctx(amp):
            out = model(x)
        t1 += (out.argmax(1) == y).sum().item()
        tot += y.numel()
        if reset: _reset(model)
    dt = time.time() - st
    print("[%s] RESULT  N=%d  top1=%.2f%%  (%.1fs, %.1f img/s)" % (name, tot, t1 / tot * 100, dt, tot / dt))
    return t1 / tot * 100, tot
