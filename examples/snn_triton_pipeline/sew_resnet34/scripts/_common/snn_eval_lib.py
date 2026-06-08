"""Shared ImageNet-val inference/eval helpers for the 4 SNN checkpoints.
Labels: TF synset-labels file (line i = WNID of ILSVRC2012_val_{i:08d}.JPEG),
sorted-WNID -> class index == standard torchvision/timm convention.
"""
import os, time, torch, PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

BASE   = os.path.expanduser("~/charlley/snn_infer")
DATA   = os.path.join(BASE, "data")
FLAT   = os.path.join(DATA, "val_flat")
LABELS = os.path.join(DATA, "val_synset_labels.txt")
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)


def build_labels():
    wnids = [l.strip() for l in open(LABELS) if l.strip()]
    sorted_wnids = sorted(set(wnids))
    assert len(sorted_wnids) == 1000, len(sorted_wnids)
    w2i = {w: i for i, w in enumerate(sorted_wnids)}
    return wnids, w2i


def make_transform(kind):
    if kind == "r256_bilinear":   # torchvision default (SEW-ResNet)
        return T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])
    if kind == "r256_bicubic":    # DeiT/MAE eval (SDT-V2 / V3), crop_pct=224/256
        return T.Compose([T.Resize(256, interpolation=PIL.Image.BICUBIC), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])
    if kind == "r224_bicubic":    # crop_pct=1.0 (Spikingformer ckpt args)
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


@torch.no_grad()
def evaluate(model, name, n=2000, batch_size=50, transform_kind="r256_bicubic", reset=True, workers=8, device="cuda"):
    try:
        from spikingjelly.clock_driven import functional
    except Exception:
        functional = None
    ds = ValSet(n, make_transform(transform_kind))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    model.eval().to(device)
    t1 = t5 = tot = 0
    st = time.time()
    for x, y in dl:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        _, pred = out.topk(5, 1, True, True)
        pred = pred.t()
        corr = pred.eq(y.view(1, -1))
        t1 += corr[:1].reshape(-1).float().sum().item()
        t5 += corr[:5].reshape(-1).float().sum().item()
        tot += y.numel()
        if reset and functional is not None:
            functional.reset_net(model)
    dt = time.time() - st
    print("[%s] RESULT  N=%d  top1=%.2f%%  top5=%.2f%%  (%.1fs, %.1f img/s, T=%s)" % (
        name, tot, t1 / tot * 100, t5 / tot * 100, dt, tot / dt, getattr(model, "T", "?")))
    return t1 / tot * 100, t5 / tot * 100, tot
