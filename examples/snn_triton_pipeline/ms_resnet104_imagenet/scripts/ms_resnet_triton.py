"""MS-ResNet-104 (ImageNet) rewritten with spikingjelly + Triton neuron backend.

Original: Ariande1/MS-ResNet (TNNLS'24, membrane-shortcut ResNet).
- mem_update neuron: H[t]=decay*H[t-1]*(1-S[t-1])+X[t], decay=0.25, thresh=0.5, hard-reset-to-0, T=6.
  This is EXACTLY spikingjelly LIFNode(tau=4/3, decay_input=False, v_threshold=0.5, v_reset=0.0):
  1-1/tau = 1-0.75 = 0.25 = decay; hard reset to 0; input added at scale 1.
- TDBN (batch_norm_2d) normalizes over (T*B,H,W) per channel == BatchNorm over merged T,B -> kept as-is.
- Snn_Conv2d loops conv over T -> replaced by a single batched conv on [T*B,...] (compile-friendly), same math.
We monkeypatch the original module so structure + checkpoint keys are untouched.
"""
import os, sys, torch, torch.nn as nn, torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate, functional

BACKEND = os.environ.get("SJ_NEURON_BACKEND", "triton")
REPO = "/home/liushifeng/charlley/snn_infer/repos/MS-ResNet"


def _import_msresnet():
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    import models.MS_ResNet as M
    return M


class _MemUpdateTriton(nn.Module):
    """Drop-in for MS-ResNet mem_update: membrane-shortcut LIF via spikingjelly triton multi-step.
    LIFNode(tau=4/3, decay_input=False, v_threshold=0.5, v_reset=0.0) == original dynamics."""
    def __init__(self):
        super().__init__()
        self.lif = neuron.LIFNode(tau=4.0 / 3.0, decay_input=False, v_threshold=0.5,
                                  v_reset=0.0, surrogate_function=surrogate.Sigmoid(),
                                  detach_reset=True, step_mode="m", backend=BACKEND)

    def forward(self, x):                    # x: [T, B, C, H, W]
        functional.reset_net(self.lif)       # original mem starts at 0 each call
        return self.lif(x)


def _snn_conv_forward(self, input):          # batched per-timestep conv (replaces python T-loop)
    T, B = input.shape[0], input.shape[1]
    y = F.conv2d(input.flatten(0, 1), self.weight, self.bias, self.stride,
                 self.padding, self.dilation, self.groups)
    return y.view(T, B, *y.shape[1:])


def build_msresnet104(ckpt_path):
    M = _import_msresnet()
    M.mem_update = _MemUpdateTriton          # rebind module global -> used at construction
    M.Snn_Conv2d.forward = _snn_conv_forward # batched conv (keeps nn.Conv2d weights/keys)
    net = M.resnet104()
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    miss, unexp = net.load_state_dict(sd, strict=False)
    print("[MS-ResNet-104] load: missing=%d unexpected=%d" % (len(miss), len(unexp)))
    if miss:
        print("  missing[:6]:", list(miss)[:6])
    if unexp:
        print("  unexpected[:6]:", list(unexp)[:6])
    return net.eval()
