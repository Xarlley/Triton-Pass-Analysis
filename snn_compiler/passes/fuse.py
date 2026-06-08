"""模型级图重写 pass：识别 Conv→(BN)→Neuron 模式并替换为融合 module。

策略
====
本 pass 不依赖 torch.fx 的 symbolic trace（多步 SNN 包含 T 循环，trace 不稳），
而是直接遍历模型的 named_modules 结构，按层间相邻关系替换。这与 PyTorch
官方 torch.ao.quantization 的 fuse_modules 思路一致——逐层模式匹配。

支持的模式
==========
1. (Conv2d, IFNode/LIFNode)              → FusedConvNeuron
2. (Conv2d, BatchNorm2d, IFNode/LIFNode) → FusedConvBNNeuron
3. (Linear, IFNode/LIFNode)              → FusedLinearNeuron

支持的 neuron module
====================
- 本框架的 IFNode / LIFNode（直接识别）
- SpikingJelly 的 IFNode / LIFNode（duck-typing：检查 v_threshold、tau 属性）
- 任何具有 'v_threshold' 属性、可在 forward(x_seq) 输出 spike_seq 的 module

使用方式
========
    from snn_compiler.passes import fuse_snn_model
    fused = fuse_snn_model(model, layout="NCHW")  # 替换原 module 为融合版

只在 model.eval() 状态使用（推理）；训练态请保留原结构。
"""
from __future__ import annotations

from typing import Iterable, List, Tuple, Type
import torch
import torch.nn as nn

from ..nn.modules import (
    IFNode, LIFNode, CubaLIFNode, EIFNode,
    FusedConvNeuron, FusedConvBNNeuron, FusedLinearNeuron,
    FusedConvBNAddNeuron, FusedAddNeuron,
)


# ============================================================
#   neuron 识别：本框架 + SpikingJelly 都接受
# ============================================================
def _is_if_node(m: nn.Module) -> bool:
    if isinstance(m, IFNode):
        return True
    cls_name = type(m).__name__
    if cls_name == "IFNode" and hasattr(m, "v_threshold"):
        return True
    return False


def _is_lif_node(m: nn.Module) -> bool:
    if isinstance(m, LIFNode):
        return True
    cls_name = type(m).__name__
    if cls_name == "LIFNode" and hasattr(m, "v_threshold") and hasattr(m, "tau"):
        return True
    return False


def _is_neuron(m: nn.Module) -> bool:
    return _is_if_node(m) or _is_lif_node(m)


def _neuron_kwargs(m: nn.Module) -> dict:
    """从一个 neuron module 抽取构造参数。

    优先识别本框架；其次按 SJ LIFNode/IFNode 的常见属性。
    """
    if isinstance(m, IFNode):
        return dict(
            neuron="if", decay=m.decay,
            soft_reset=m.soft_reset, v_threshold=m.v_threshold,
            v_reset=m.v_reset, layout=m.layout,
        )
    if isinstance(m, LIFNode):
        return dict(
            neuron="lif", tau=m.tau, decay=m.decay, decay_input=m.decay_input,
            soft_reset=m.soft_reset, v_threshold=m.v_threshold,
            v_reset=m.v_reset, layout=m.layout,
        )
    # SpikingJelly duck-typing
    v_th = getattr(m, "v_threshold", 1.0)
    v_reset = getattr(m, "v_reset", 0.0)
    if v_reset is None:                       # SJ 用 None 表示 soft reset
        soft = True
        v_reset = 0.0
    else:
        soft = False
    if _is_lif_node(m):
        tau = getattr(m, "tau", 2.0)
        decay_input = getattr(m, "decay_input", True)
        return dict(
            neuron="lif", tau=tau, decay_input=decay_input,
            soft_reset=soft, v_threshold=v_th, v_reset=v_reset, layout="NCHW",
        )
    return dict(
        neuron="if", decay=1.0,
        soft_reset=soft, v_threshold=v_th, v_reset=v_reset, layout="NCHW",
    )


# ============================================================
#   核心 fuser：递归扫描 nn.Sequential / nn.ModuleList
# ============================================================
def _fuse_seq(seq: nn.Sequential, *, layout: str, fold_bn: bool = True) -> nn.Sequential:
    new_layers: List[nn.Module] = []
    i = 0
    layers = list(seq.children())
    n = len(layers)
    fused_count = 0

    while i < n:
        a = layers[i]

        # Pattern 1: Conv → BN → Neuron
        if (isinstance(a, nn.Conv2d) and i + 2 < n
                and isinstance(layers[i + 1], nn.BatchNorm2d)
                and _is_neuron(layers[i + 2])):
            kw = _neuron_kwargs(layers[i + 2])
            kw["layout"] = layout
            mod = FusedConvBNNeuron(a.eval(), layers[i + 1].eval(), fold_bn=fold_bn, **kw)
            mod = mod.to(device=a.weight.device, dtype=a.weight.dtype)
            new_layers.append(mod)
            i += 3
            fused_count += 1
            continue

        # Pattern 2: Conv → Neuron (no BN)
        if (isinstance(a, nn.Conv2d) and i + 1 < n
                and _is_neuron(layers[i + 1])):
            kw = _neuron_kwargs(layers[i + 1])
            kw["layout"] = layout
            mod = FusedConvNeuron(
                a.in_channels, a.out_channels, a.kernel_size,
                stride=a.stride, padding=a.padding, dilation=a.dilation,
                groups=a.groups, bias=(a.bias is not None),
                **kw,
            )
            mod = mod.to(device=a.weight.device, dtype=a.weight.dtype)
            with torch.no_grad():
                mod.weight.copy_(a.weight)
                if a.bias is not None and mod.bias is not None:
                    mod.bias.copy_(a.bias)
            new_layers.append(mod)
            i += 2
            fused_count += 1
            continue

        # Pattern 3: Linear → Neuron
        if (isinstance(a, nn.Linear) and i + 1 < n
                and _is_neuron(layers[i + 1])):
            kw = _neuron_kwargs(layers[i + 1])
            kw.pop("layout", None)
            mod = FusedLinearNeuron(
                a.in_features, a.out_features, bias=(a.bias is not None),
                **kw,
            )
            mod = mod.to(device=a.weight.device, dtype=a.weight.dtype)
            with torch.no_grad():
                mod.weight.copy_(a.weight)
                if a.bias is not None and mod.bias is not None:
                    mod.bias.copy_(a.bias)
            new_layers.append(mod)
            i += 2
            fused_count += 1
            continue

        # 递归进入子 Sequential
        if isinstance(a, nn.Sequential):
            sub_seq, sub_n = _fuse_seq(a, layout=layout, fold_bn=fold_bn)
            new_layers.append(sub_seq)
            fused_count += sub_n
        else:
            new_layers.append(a)
        i += 1

    return nn.Sequential(*new_layers), fused_count


def fuse_snn_model(model: nn.Module, *, layout: str = "NCHW",
                   fold_bn: bool = True) -> Tuple[nn.Module, int]:
    """递归扫描模型，把所有 Conv/Linear→(BN)→Neuron 模式替换为融合 module。

    Args:
        model: 处于 eval() 状态的模型。
        layout: 'NCHW' or 'NHWC'。
        fold_bn: True（默认）折叠 BN 进 conv（最快，但非逐位一致）；
            False 保留 BN 为独立算子、只融合 neuron（逐位一致，见
            ``FusedConvBNNeuron``）。

    Returns:
        (fused_model, n_fused): 替换后的新模型，及替换计数。

    限制：
    - 仅在 nn.Sequential 内识别相邻的线性模式。残差等非顺序连接走
      ``fuse_modules_path`` 或直接构造 FusedConvBNAddNeuron / FusedAddNeuron。
    - **请在信任结果前用 ``snn_compiler.verify.assert_equivalent`` 校验**。
    """
    if not isinstance(model, nn.Sequential):
        total = 0
        for name, child in list(model.named_children()):
            new_child, n = fuse_snn_model(child, layout=layout, fold_bn=fold_bn)
            setattr(model, name, new_child)
            total += n
        return model, total
    new_seq, n = _fuse_seq(model, layout=layout, fold_bn=fold_bn)
    return new_seq, n


# ============================================================
#   路径式融合 —— 用于非 Sequential 容器（ResNet block / custom）
# ============================================================
def _get_submodule(model: nn.Module, path: str) -> nn.Module:
    """按 'a.b.c' 取出子 module。"""
    cur = model
    for part in path.split("."):
        cur = getattr(cur, part) if not part.isdigit() else cur[int(part)]
    return cur


def _set_submodule(model: nn.Module, path: str, mod: nn.Module) -> None:
    """按 'a.b.c' 设置子 module。"""
    parts = path.split(".")
    cur = model
    for part in parts[:-1]:
        cur = getattr(cur, part) if not part.isdigit() else cur[int(part)]
    last = parts[-1]
    if last.isdigit():
        cur[int(last)] = mod
    else:
        setattr(cur, last, mod)


def fuse_modules_path(model: nn.Module, fusion_groups, *, layout: str = "NCHW",
                      fold_bn: bool = True) -> int:
    """按显式路径融合。

    Args:
        model: 处于 eval() 状态的模型。
        fusion_groups: ``[(conv_path, [bn_path], neuron_path, target_path), ...]``
            每个组里：第一个是 Conv2d 或 Linear 的路径；可选 BN 路径；neuron 路径；
            最后一项是融合后 module 放置的路径（通常等于 neuron_path）。
            如果只有 3 个元素，第 3 个被同时当作 neuron_path 与 target_path。
        layout: 'NCHW' / 'NHWC'。

    Returns:
        实际成功融合的组数。

    使用例（ResNet basic block，假设 `block` 是 BasicBlock 实例）::

        from snn_compiler.passes import fuse_modules_path
        # block.conv1 + block.bn1 + block.neuron1 → block.neuron1 (FusedConvBNNeuron)
        n = fuse_modules_path(block, [
            ("conv1", "bn1", "neuron1"),
        ])
        # 之后 block.forward 里把 conv1/bn1 的调用换成 block.neuron1(x)。
        # 残差线 conv2-bn2 + identity + neuron2 用 FusedConvBNAddNeuron 直接构造，
        # 见 snn_compiler.zoo.resnet。
    """
    n_done = 0
    for grp in fusion_groups:
        if len(grp) == 3:
            conv_p, bn_p, neuron_p = grp
            target_p = neuron_p
        elif len(grp) == 4:
            conv_p, bn_p, neuron_p, target_p = grp
        else:
            raise ValueError(f"fusion group must have 3 or 4 elements, got {grp}")

        conv = _get_submodule(model, conv_p)
        neuron = _get_submodule(model, neuron_p)
        kw = _neuron_kwargs(neuron)
        kw["layout"] = layout

        if bn_p is None or bn_p == "":
            # Conv → Neuron / Linear → Neuron
            if isinstance(conv, nn.Conv2d):
                mod = FusedConvNeuron(
                    conv.in_channels, conv.out_channels, conv.kernel_size,
                    stride=conv.stride, padding=conv.padding,
                    dilation=conv.dilation, groups=conv.groups,
                    bias=(conv.bias is not None), **kw,
                ).to(device=conv.weight.device, dtype=conv.weight.dtype)
                with torch.no_grad():
                    mod.weight.copy_(conv.weight)
                    if conv.bias is not None:
                        mod.bias.copy_(conv.bias)
            elif isinstance(conv, nn.Linear):
                kw_lin = dict(kw); kw_lin.pop("layout", None)
                mod = FusedLinearNeuron(
                    conv.in_features, conv.out_features,
                    bias=(conv.bias is not None), **kw_lin,
                ).to(device=conv.weight.device, dtype=conv.weight.dtype)
                with torch.no_grad():
                    mod.weight.copy_(conv.weight)
                    if conv.bias is not None:
                        mod.bias.copy_(conv.bias)
            else:
                raise TypeError(f"{conv_p} is not Conv2d or Linear: {type(conv)}")
        else:
            # Conv → BN → Neuron
            bn = _get_submodule(model, bn_p)
            if not isinstance(conv, nn.Conv2d) or not isinstance(bn, nn.BatchNorm2d):
                raise TypeError(f"{conv_p}/{bn_p}: expected Conv2d/BatchNorm2d")
            mod = FusedConvBNNeuron(conv.eval(), bn.eval(), fold_bn=fold_bn, **kw).to(
                device=conv.weight.device, dtype=conv.weight.dtype
            )

        _set_submodule(model, target_p, mod)
        # 把已被融合掉的 conv 和 bn 替换为 nn.Identity（避免 forward 重复调用）
        if conv_p != target_p:
            _set_submodule(model, conv_p, nn.Identity())
        if bn_p and bn_p != target_p:
            _set_submodule(model, bn_p, nn.Identity())
        n_done += 1
    return n_done


def fuse_conv_bn_add_neuron_path(model: nn.Module,
                                  conv_p: str, bn_p: str, neuron_p: str,
                                  *, target_p: str | None = None,
                                  layout: str = "NCHW",
                                  fold_bn: bool = True) -> nn.Module:
    """把 (conv, bn, neuron) 替换为 FusedConvBNAddNeuron，返回融合后 module。

    与 fuse_modules_path 不同：这里**不会改 model 自己的 forward**，只是把
    Conv/BN/Neuron 三个子 module 抽出来组装成一个 FusedConvBNAddNeuron，把它
    放到 target_p（默认 neuron_p），并把 conv/bn 设为 Identity。

    剩余的"残差怎么传"由 model.forward 自行处理 —— 在 ResNet 中通常是把
    `self.neuron2(out + identity)` 改成 `self.neuron2(out, identity)`，因为
    FusedConvBNAddNeuron 的 forward 签名是 (x, residual)。
    """
    target_p = target_p or neuron_p
    conv = _get_submodule(model, conv_p)
    bn = _get_submodule(model, bn_p)
    neuron = _get_submodule(model, neuron_p)
    kw = _neuron_kwargs(neuron)
    kw["layout"] = layout
    mod = FusedConvBNAddNeuron(conv.eval(), bn.eval(), fold_bn=fold_bn, **kw).to(
        device=conv.weight.device, dtype=conv.weight.dtype
    )
    _set_submodule(model, target_p, mod)
    if conv_p != target_p:
        _set_submodule(model, conv_p, nn.Identity())
    if bn_p != target_p:
        _set_submodule(model, bn_p, nn.Identity())
    return mod
