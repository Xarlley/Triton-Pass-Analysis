"""SEW-ResNet-34: accelerate with snn_compiler. Correctness + speed.

Reference  = original Spike-Element-Wise-ResNet model, pretrained ckpt 319,
             SpikingJelly eager torch neuron backend (ground truth + naive baseline).

Two accelerated variants, both keeping SEW topology EXACTLY
(residual ADD AFTER both neurons; downsample has its own IF neuron;
 stem conv-bn applied before the T-replication):

  EXACT : conv+BN kept as separate eager ops, only the IF neuron replaced by
          snn_compiler's fused Triton IF kernel  ->  bit-exact to reference.
  FOLD  : snn_compiler FusedConvBNNeuron (BN folded into conv, conv+bias+IF) ->
          faster, but BN-fold perturbs pre-activations ~1e-3 -> a few
          threshold-borderline spikes flip; real-data accuracy measured.
"""
import os, sys, time, argparse, contextlib
import torch, torch.nn as nn, torch.nn.functional as F

os.environ.setdefault("SJ_NEURON_BACKEND", "torch")
BASE  = "/home/liushifeng/charlley/snn_infer"
TBASE = "/home/liushifeng/charlley/snn_infer_triton"
SNNC  = "/home/liushifeng/charlley/snn_compiler_test"
sys.path.insert(0, SNNC); sys.path.insert(0, TBASE)
import sj_compat                                              # noqa
sys.path.insert(0, os.path.join(BASE, "repos/Spike-Element-Wise-ResNet/imagenet"))
import sew_resnet
from spikingjelly.activation_based import functional as AF
from snn_compiler.nn.modules import FusedConvBNNeuron, IFNode
from snn_compiler.kernels.neurons import if_lif
from snn_compiler.kernels.fused import fold_conv_bn

CK = "/home/liushifeng/lsf/checkpoints/sew_resnet34_imagenet/sew34_checkpoint_319.pth"
IFK = dict(neuron="if", soft_reset=False, v_threshold=1.0, v_reset=0.0)


def load_reference():
    sd = torch.load(CK, map_location="cpu", weights_only=False)["model"]
    m = sew_resnet.sew_resnet34(T=4, connect_f="ADD", num_classes=1000,
                                zero_init_residual=False)
    if not any(".module." in k for k in m.state_dict().keys()):
        sd = {k.replace(".module.", "."): v for k, v in sd.items()}
    miss, unexp = m.load_state_dict(sd, strict=False)
    assert not miss and not unexp
    return m.eval()


def _find_conv_bn(mod):
    conv = bn = None
    for s in mod.modules():
        if isinstance(s, nn.Conv2d) and conv is None: conv = s
        elif isinstance(s, nn.BatchNorm2d) and bn is None: bn = s
    return conv, bn


def _ifk(x):                                                  # spike via Triton IF kernel
    return if_lif(x.contiguous() if (x.is_contiguous()) else x, layout="NCHW", **IFK)


# ===================== EXACT variant (bit-exact) =====================
class ExactBlock(nn.Module):
    def __init__(self, rb):
        super().__init__()
        self.conv1 = rb.conv1                                 # SeqToANN(conv,bn) reused as-is
        self.conv2 = rb.conv2
        self.ds = rb.downsample[0] if rb.downsample is not None else None  # SeqToANN(conv,bn)

    def forward(self, x):
        out = _ifk(self.conv1(x))
        out = _ifk(self.conv2(out))
        identity = x if self.ds is None else _ifk(self.ds(x))
        return out + identity


class SEWExact(nn.Module):
    def __init__(self, ref, T=4):
        super().__init__()
        self.T = T
        self.conv1, self.bn1 = ref.conv1, ref.bn1
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.stages = nn.ModuleList(
            nn.ModuleList(ExactBlock(b) for b in getattr(ref, ln))
            for ln in ("layer1", "layer2", "layer3", "layer4"))
        self.fc = ref.fc

    def forward(self, x):
        y = self.bn1(self.conv1(x))
        x = _ifk(y.unsqueeze(0).repeat(self.T, 1, 1, 1, 1))
        T, B = x.shape[0], x.shape[1]
        x = self.maxpool(x.reshape(T * B, *x.shape[2:])).view(T, B, 64, 56, 56)
        for st in self.stages:
            for blk in st: x = blk(x)
        x = F.adaptive_avg_pool2d(x.reshape(T * B, *x.shape[2:]), 1).flatten(1).view(T, B, -1)
        return self.fc(x.mean(0))


# ===================== FOLD variant (snn_compiler fused) =====================
class FoldBlock(nn.Module):
    def __init__(self, rb, layout):
        super().__init__()
        c1, n1 = _find_conv_bn(rb.conv1); c2, n2 = _find_conv_bn(rb.conv2)
        self.b1 = FusedConvBNNeuron(c1.eval(), n1.eval(), layout=layout, **IFK)
        self.b2 = FusedConvBNNeuron(c2.eval(), n2.eval(), layout=layout, **IFK)
        if rb.downsample is not None:
            dc, dn = _find_conv_bn(rb.downsample)
            self.ds = FusedConvBNNeuron(dc.eval(), dn.eval(), layout=layout, **IFK)
        else:
            self.ds = None

    def forward(self, x):
        out = self.b2(self.b1(x))
        identity = x if self.ds is None else self.ds(x)
        return out + identity


class StemFold(nn.Module):
    def __init__(self, conv1, bn1, T, layout):
        super().__init__()
        w, b = fold_conv_bn(conv1.weight.detach(), None, bn1.weight.detach(),
                            bn1.bias.detach(), bn1.running_mean.detach(),
                            bn1.running_var.detach(), bn1.eps)
        self.weight = nn.Parameter(w); self.bias = nn.Parameter(b)
        self.stride = conv1.stride; self.padding = conv1.padding
        self.T, self.layout = T, layout
        self.neuron = IFNode(layout=layout, soft_reset=False, v_threshold=1.0, v_reset=0.0)

    def forward(self, x):
        if self.layout == "NHWC":
            x = x.contiguous(memory_format=torch.channels_last)
            w = self.weight.to(memory_format=torch.channels_last)
        else:
            w = self.weight
        y = F.conv2d(x, w, bias=self.bias, stride=self.stride, padding=self.padding)
        return self.neuron(y.unsqueeze(0).repeat(self.T, 1, 1, 1, 1))


class SEWFold(nn.Module):
    def __init__(self, ref, T=4, layout="NCHW"):
        super().__init__()
        self.T, self.layout = T, layout
        self.stem = StemFold(ref.conv1, ref.bn1, T, layout)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.stages = nn.ModuleList(
            nn.ModuleList(FoldBlock(b, layout) for b in getattr(ref, ln))
            for ln in ("layer1", "layer2", "layer3", "layer4"))
        self.fc = nn.Linear(ref.fc.in_features, ref.fc.out_features)
        self.fc.load_state_dict(ref.fc.state_dict())

    def forward(self, x):
        x = self.stem(x)
        T, B = x.shape[0], x.shape[1]
        x4 = x.reshape(T * B, *x.shape[2:])
        if self.layout == "NHWC": x4 = x4.contiguous(memory_format=torch.channels_last)
        x = self.maxpool(x4).view(T, B, 64, 56, 56)
        for st in self.stages:
            for blk in st: x = blk(x)
        x = F.adaptive_avg_pool2d(x.reshape(T * B, *x.shape[2:]), 1).flatten(1).view(T, B, -1)
        return self.fc(x.mean(0).to(self.fc.weight.dtype))


def amp_ctx(amp):
    if amp == "bf16": return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def cmp(yref, y, tag):
    md = (yref - y).abs().max().item()
    agree = (yref.argmax(1) == y.argmax(1)).float().mean().item() * 100
    print(f"  [{tag:22}] max|Δ|={md:.3e}  argmax-agree={agree:6.2f}%")


def bench(fn, reset_fn=None, warmup=10, iters=40):
    for _ in range(warmup):
        if reset_fn: reset_fn()
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        if reset_fn: reset_fn()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2], sum(ts) / len(ts), ts[0]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="correctness", choices=["correctness", "speed"])
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--iters", type=int, default=40)
    a = ap.parse_args()
    torch.manual_seed(0)
    ref = load_reference().cuda()

    if a.mode == "correctness":
        x = torch.randn(8, 3, 224, 224, device="cuda")
        AF.reset_net(ref); yref = ref(x).float()
        print("Correctness vs reference (eager fp32), random bs=8:")
        cmp(yref, SEWExact(ref).cuda().eval()(x).float(), "EXACT fp32")
        cmp(yref, SEWFold(ref, layout="NCHW").cuda().eval()(x).float(), "FOLD fp32 NCHW")
        fb = SEWFold(ref, layout="NHWC").cuda().eval().to(torch.bfloat16)
        xb = x.to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
        cmp(yref, fb(xb).float(), "FOLD bf16 NHWC")

    if a.mode == "speed":
        B = a.bs
        x = torch.randn(B, 3, 224, 224, device="cuda")
        print(f"\n=== SPEED  bs={B} T=4  ms/img (median|mean|best over {a.iters}it), img/s, speedup ===")
        rst = lambda: AF.reset_net(ref)
        med0, mn0, bst0 = bench(lambda: ref(x), reset_fn=rst, iters=a.iters)
        print(f"  ref eager fp32       : {med0/B*1e3:6.3f}|{mn0/B*1e3:6.3f}|{bst0/B*1e3:6.3f}  {B/med0:7.1f} img/s   1.00x")
        medb, mnb, bstb = bench(lambda: amp_ctx('bf16').__enter__() or ref(x), reset_fn=rst, iters=a.iters)
        # proper bf16 ctx:
        def ref_bf16():
            with amp_ctx('bf16'): ref(x)
        medb, mnb, bstb = bench(ref_bf16, reset_fn=rst, iters=a.iters)
        print(f"  ref eager bf16(amp)  : {medb/B*1e3:6.3f}|{mnb/B*1e3:6.3f}|{bstb/B*1e3:6.3f}  {B/medb:7.1f} img/s   {med0/medb:.2f}x")

        def run(model, xin):
            return lambda: model(xin)
        ex = SEWExact(ref).cuda().eval()
        m, mn, b = bench(run(ex, x), iters=a.iters)
        print(f"  EXACT fp32 (Triton IF): {m/B*1e3:6.3f}|{mn/B*1e3:6.3f}|{b/B*1e3:6.3f}  {B/m:7.1f} img/s   {med0/m:.2f}x")
        del ex; torch.cuda.empty_cache()

        f32 = SEWFold(ref, layout="NCHW").cuda().eval()
        m, mn, b = bench(run(f32, x), iters=a.iters)
        print(f"  FOLD fp32 NCHW       : {m/B*1e3:6.3f}|{mn/B*1e3:6.3f}|{b/B*1e3:6.3f}  {B/m:7.1f} img/s   {med0/m:.2f}x")
        del f32; torch.cuda.empty_cache()

        fb = SEWFold(ref, layout="NHWC").cuda().eval().to(torch.bfloat16)
        xb = x.to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
        m, mn, b = bench(run(fb, xb), iters=a.iters)
        print(f"  FOLD bf16 NHWC       : {m/B*1e3:6.3f}|{mn/B*1e3:6.3f}|{b/B*1e3:6.3f}  {B/m:7.1f} img/s   {med0/m:.2f}x (vs fp32)  {medb/m:.2f}x (vs bf16-ref)")


if __name__ == "__main__":
    main()
