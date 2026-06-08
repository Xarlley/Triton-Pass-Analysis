"""Minimal shim for the handful of timm/einops symbols the repo model files
import, so they load under triton-src (torch 2.12) without installing timm.
Only inference-relevant pieces matter:
- trunc_normal_ / _cfg: only touch random init (overwritten by load_state_dict)
- DropPath: identity in eval
- register_model / create_model: we instantiate model classes directly
Import BEFORE importing repo model code.
"""
import sys, types, collections.abc as cabc
import torch
import torch.nn as nn


def to_2tuple(x):
    if isinstance(x, cabc.Iterable) and not isinstance(x, str):
        return tuple(x)
    return (x, x)


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep) * mask.floor_()


def register_model(fn):
    return fn


def _cfg(url="", **kwargs):
    d = {"url": url, "num_classes": 1000, "input_size": (3, 224, 224), "pool_size": None,
         "crop_pct": 0.9, "interpolation": "bicubic", "mean": (0.485, 0.456, 0.406),
         "std": (0.229, 0.224, 0.225), "first_conv": "", "classifier": ""}
    d.update(kwargs)
    return d


def create_model(*a, **k):
    raise NotImplementedError("timm_compat.create_model: instantiate the model class directly")


def _reg(fq, **attrs):
    m = types.ModuleType(fq)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[fq] = m
    return m


_layers = _reg("timm.models.layers", to_2tuple=to_2tuple, trunc_normal_=trunc_normal_, DropPath=DropPath)
_registry = _reg("timm.models.registry", register_model=register_model)
_vit = _reg("timm.models.vision_transformer", _cfg=_cfg)
_models = _reg("timm.models", layers=_layers, registry=_registry, vision_transformer=_vit, create_model=create_model)
_reg("timm", models=_models, __version__="0.6.12-shim")
# NOTE: einops is really installed in triton-src (spikingjelly needs it) -> do NOT shim it.
