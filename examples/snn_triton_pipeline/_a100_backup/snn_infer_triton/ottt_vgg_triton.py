"""OTTT VGG-11-WS (CIFAR10) rewritten in multi-step spikingjelly + Triton neuron backend.

Original: pkuxmq/OTTT-SNN  (NeurIPS'22, "Online Training Through Time").
- Online single-step OnlineLIFNode looped T times externally; charge v=v*(1-1/tau)+x,
  v_threshold=1, v_reset=None (SOFT reset), tau=2, T=6, Scale(2.74) after each neuron,
  Weight-Standardized convs (ScaledWSConv2d), AvgPool, head Linear(512,10) (fc_hw=1).
Here: equivalent MULTI-STEP feed-forward SNN. Forward is identical (no cross-layer temporal
feedback => time-major vs layer-major give the same values; online detach only affects grads).
Neuron -> spikingjelly activation_based LIFNode(step_mode='m', backend=$SJ_NEURON_BACKEND).
"""
import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate, layer, functional

BACKEND = os.environ.get("SJ_NEURON_BACKEND", "triton")
T_STEPS = 6


class ScaledWSConv2d(nn.Conv2d):
    """Weight-standardized conv (verbatim from OTTT repo: var unbiased, eps=1e-4, learnable gain)."""
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, gain=True, eps=1e-4):
        super().__init__(in_c, out_c, kernel_size, stride, padding, dilation, groups, bias)
        self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1)) if gain else None
        self.eps = eps

    def get_weight(self):
        fan_in = np.prod(self.weight.shape[1:])
        mean = torch.mean(self.weight, axis=[1, 2, 3], keepdims=True)
        var = torch.var(self.weight, axis=[1, 2, 3], keepdims=True)
        weight = (self.weight - mean) / ((var * fan_in + self.eps) ** 0.5)
        if self.gain is not None:
            weight = weight * self.gain
        return weight

    def forward(self, x):
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)


class Scale(nn.Module):
    def __init__(self, scale): super().__init__(); self.scale = scale
    def forward(self, x): return x * self.scale


def _lif():
    # OTTT OnlineLIFNode(tau=2, decay_input=False, v_threshold=1, v_reset=None[soft]) -> multi-step triton LIF
    return neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=None,
                          surrogate_function=surrogate.Sigmoid(), detach_reset=True,
                          step_mode="m", backend=BACKEND)


CFG_A = [64, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512]


class MSVGG11WS(nn.Module):
    """Multi-step Weight-Standardized VGG-11 (spikingjelly idiom: SeqToANNContainer wraps stateless ops)."""
    def __init__(self, num_classes=10, T=T_STEPS):
        super().__init__()
        self.T = T
        feats, self.convs = [], []
        in_c = 3
        for v in CFG_A:
            if v == "M":
                feats.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))
            else:
                conv = ScaledWSConv2d(in_c, v, kernel_size=3, padding=1, stride=1)
                self.convs.append(conv)
                feats += [layer.SeqToANNContainer(conv), _lif(), Scale(2.74)]
                in_c = v
        self.features = nn.Sequential(*feats)
        self.avgpool = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.head = nn.Linear(512, num_classes)        # fc_hw=1 -> 512 in features

    def forward(self, x):
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)   # [B,C,H,W] -> [T,B,C,H,W] (direct encoding)
        x = self.features(x)                            # spikes [T,B,512,4,4]
        x = self.avgpool(x).flatten(2)                  # [T,B,512]
        Tt, B, _ = x.shape
        x = self.head(x.flatten(0, 1)).view(Tt, B, -1)  # per-step logits [T,B,10]
        return x.sum(0)                                 # == OTTT total_fr (sum over T)


def load_ottt_checkpoint(model, ckpt_path):
    """Map OTTT checkpoint (features.{0|X.op}.{weight,bias,gain}, classifier.0.op.{weight,bias})
    onto the rebuilt model positionally."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["net"]
    # ordered conv triples from ckpt: features.0 then features.{N}.op ...
    conv_idx = []
    for k in sd:
        if k.startswith("features.") and k.endswith(".weight"):
            conv_idx.append(k[:-len(".weight")])           # prefix e.g. 'features.3.op' or 'features.0'
    # keep declaration order (numeric by feature index)
    conv_idx = sorted(set(conv_idx), key=lambda p: int(p.split(".")[1]))
    assert len(conv_idx) == len(model.convs), (len(conv_idx), len(model.convs))
    new = {}
    for pref, conv in zip(conv_idx, model.convs):
        cn = dict(conv.named_parameters(prefix="", recurse=False))  # weight,bias,gain
        # find the module path of this conv in the model
        # we'll just copy tensors directly
        conv.weight.data.copy_(sd[pref + ".weight"])
        if conv.bias is not None and (pref + ".bias") in sd:
            conv.bias.data.copy_(sd[pref + ".bias"])
        if conv.gain is not None and (pref + ".gain") in sd:
            conv.gain.data.copy_(sd[pref + ".gain"])
    # head: classifier.0.op.{weight,bias} (or classifier.0.{weight,bias})
    hk = "classifier.0.op" if ("classifier.0.op.weight" in sd) else "classifier.0"
    model.head.weight.data.copy_(sd[hk + ".weight"])
    model.head.bias.data.copy_(sd[hk + ".bias"])
    return model


def build_ottt(ckpt_path, num_classes=10):
    m = MSVGG11WS(num_classes=num_classes)
    load_ottt_checkpoint(m, ckpt_path)
    return m.eval()
