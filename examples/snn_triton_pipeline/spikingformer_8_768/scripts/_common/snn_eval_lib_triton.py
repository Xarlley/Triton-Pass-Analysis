"""ImageNet-val eval helpers for the Triton-backend SNN port.
Reuses the dataset/labels prepared under ~/charlley/snn_infer/data.
Adds: spikingjelly activation_based reset, and a CUDA-kernel profiler that
classifies kernels as triton / cublas / cudnn / other (to audit Option B).
"""
import os, time, contextlib, torch, PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from spikingjelly.activation_based import functional as AF


def _amp_ctx(amp):
    if amp == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if amp == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _reset(model):
    # torch.compile wraps the module; reset the original so neuron memories are found
    AF.reset_net(getattr(model, "_orig_mod", model))

DATA = "/home/liushifeng/charlley/snn_infer/data"
FLAT = os.path.join(DATA, "val_flat")
LABELS = os.path.join(DATA, "val_synset_labels.txt")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_labels():
    wnids = [l.strip() for l in open(LABELS) if l.strip()]
    sorted_wnids = sorted(set(wnids))
    assert len(sorted_wnids) == 1000, len(sorted_wnids)
    return wnids, {w: i for i, w in enumerate(sorted_wnids)}


def make_transform(kind):
    if kind == "r256_bilinear":
        return T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])
    if kind == "r256_bicubic":
        return T.Compose([T.Resize(256, interpolation=PIL.Image.BICUBIC), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])
    if kind == "r224_bicubic":
        return T.Compose([T.Resize(224, interpolation=PIL.Image.BICUBIC), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])
    raise ValueError(kind)


class ValSet(Dataset):
    def __init__(self, n, transform):
        wnids, w2i = build_labels()
        self.tf = transform
        self.items = []
        for i in range(1, n + 1):
            fn = "ILSVRC2012_val_%08d.JPEG" % i
            p = os.path.join(FLAT, fn)
            if os.path.exists(p):
                self.items.append((p, w2i[wnids[i - 1]]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, y = self.items[idx]
        return self.tf(Image.open(p).convert("RGB")), y


def report_load(name, miss, unexp):
    print("[%s] load_state_dict: missing=%d unexpected=%d" % (name, len(miss), len(unexp)))
    if miss:
        print("   missing[:6]:", list(miss)[:6])
    if unexp:
        print("   unexpected[:6]:", list(unexp)[:6])


def _classify(nm):
    s = nm.lower()
    # inductor triton kernels (triton_poi/red/mm/per...) + spikingjelly hand-written triton neuron kernels
    if "triton" in s or any(k in s for k in ["multistep_lif", "multistep_if", "multistep_plif"]):
        return "triton"
    if "cudnn" in s:
        return "cudnn"
    if any(k in s for k in ["cutlass", "sgemm", "hgemm", "gemm", "cublas", "ampere_", "cgemm", "dgemm", "gemv"]):
        return "cublas/gemm"
    return "other"


@torch.no_grad()
def profile_kernels(model, device="cuda", transform_kind="r256_bicubic", batch_size=16, reset=True, amp="none"):
    """Run ONE batch under the profiler and tabulate CUDA kernels by class."""
    from torch.profiler import profile, ProfilerActivity
    ds = ValSet(batch_size, make_transform(transform_kind))
    x = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
    model.eval().to(device)
    with torch.no_grad(), _amp_ctx(amp):
        model(x)  # warmup / compile
    if reset:
        _reset(model)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        with _amp_ctx(amp):
            model(x)
        torch.cuda.synchronize()
    if reset:
        _reset(model)
    agg = {}
    for e in prof.key_averages():
        cuda_us = getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)
        if cuda_us <= 0:
            continue
        cls = _classify(e.key)
        d = agg.setdefault(cls, {"us": 0.0, "names": {}})
        d["us"] += cuda_us
        d["names"][e.key] = d["names"].get(e.key, 0) + cuda_us
    total = sum(d["us"] for d in agg.values()) or 1.0
    print("  -- CUDA kernel time by class (one batch, bs=%d) --" % batch_size)
    for cls in ["triton", "cublas/gemm", "cudnn", "other"]:
        if cls in agg:
            print("     %-12s %8.1f us  (%.1f%%)" % (cls, agg[cls]["us"], 100 * agg[cls]["us"] / total))
            top = sorted(agg[cls]["names"].items(), key=lambda kv: -kv[1])[:3]
            for nm, us in top:
                print("         %8.1f us  %s" % (us, nm[:90]))
    return agg


@torch.no_grad()
def evaluate(model, name, n=2000, batch_size=50, transform_kind="r256_bicubic",
             reset=True, workers=8, device="cuda", amp="none"):
    ds = ValSet(n, make_transform(transform_kind))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    model.eval().to(device)
    t1 = t5 = tot = 0
    st = time.time()
    for x, y in dl:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with _amp_ctx(amp):
            out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        _, pred = out.topk(5, 1, True, True)
        pred = pred.t()
        corr = pred.eq(y.view(1, -1))
        t1 += corr[:1].reshape(-1).float().sum().item()
        t5 += corr[:5].reshape(-1).float().sum().item()
        tot += y.numel()
        if reset:
            _reset(model)
    dt = time.time() - st
    print("[%s] RESULT  N=%d  top1=%.2f%%  top5=%.2f%%  (%.1fs, %.1f img/s)" % (
        name, tot, t1 / tot * 100, t5 / tot * 100, dt, tot / dt))
    return t1 / tot * 100, t5 / tot * 100, tot
