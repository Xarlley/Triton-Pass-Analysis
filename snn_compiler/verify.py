"""推理等价性校验 —— 防止"静默用错"。

snn_compiler 的加速来自两类改写，二者都**可能在用户不知情时改变推理结果**：

1. **拓扑改写**：把 Conv-BN-Neuron / 残差块换成融合 module。若把网络的连接关系
   接错（最典型：SEW-ResNet 的残差加在神经元**之后**、downsample 自带神经元，而
   标准 ResNet 的残差加在神经元**之前**、downsample 无神经元 —— 用错了会算成
   另一个网络且**不会报错**），输出就静默错了。
2. **BN 折叠**（``FusedConvBNNeuron(fold_bn=True)``）：数学等价但有 ~1e-3 数值扰动，
   在脉冲硬阈值下会翻转个别边界脉冲并级联。

本模块给出一个**一行校验**：把你的原网络与加速网络喂同一组输入，逐元素比对输出，
一旦超出容差就**显式报错**并给出可能原因与修复建议。**强烈建议**在信任加速网络
之前先跑一次 :func:`assert_equivalent`。

用法::

    from snn_compiler.verify import assert_equivalent
    x = torch.randn(T, B, 3, H, W, device="cuda")   # 或你网络真实的输入
    assert_equivalent(reference_model, fast_model, x)        # 默认要求逐位一致
    # 允许 BN 折叠的小数值差，但要求 top-1 不变：
    assert_equivalent(reference_model, fast_model, x, atol=5e-3, require_bitexact=False)
"""
from __future__ import annotations

from typing import Any
import torch


def _reset(model) -> None:
    """尽力重置 SpikingJelly / 本框架的神经元状态（若有）。"""
    try:
        from spikingjelly.activation_based import functional as AF
        AF.reset_net(getattr(model, "_orig_mod", model))
        return
    except Exception:
        pass
    try:
        from spikingjelly.clock_driven import functional as CF
        CF.reset_net(getattr(model, "_orig_mod", model))
    except Exception:
        pass


def _as_tensor(out) -> torch.Tensor:
    if isinstance(out, (tuple, list)):
        out = out[0]
    if not isinstance(out, torch.Tensor):
        raise TypeError(f"model output is not a tensor (got {type(out)})")
    return out


@torch.no_grad()
def _run(model, x) -> torch.Tensor:
    _reset(model)
    out = _as_tensor(model(x))
    _reset(model)
    return out.float()


@torch.no_grad()
def compare_models(reference, fast, example_input, *,
                   reduce_logits: bool = True) -> dict[str, Any]:
    """跑两个模型、比对输出，返回度量字典（不抛异常）。

    Args:
        reference: 原始（未加速）模型 —— 当作 ground truth。
        fast: 加速后的模型。
        example_input: 喂给两个模型的同一个输入张量。
        reduce_logits: 若输出最后一维像是 logits（>=2 类），统计 top-1 一致率。

    Returns 包含 key：``shape_match`` / ``max_abs_diff`` / ``mean_abs_diff`` /
    ``rel_max`` / ``argmax_agree``（None 表示不适用）/ ``spike_mismatch_frac``
    （两边都像脉冲 0/1 时给出，否则 None）/ ``bit_exact``。
    """
    yref = _run(reference, example_input)
    yfast = _run(fast, example_input)

    rep: dict[str, Any] = {
        "ref_shape": tuple(yref.shape),
        "fast_shape": tuple(yfast.shape),
        "shape_match": tuple(yref.shape) == tuple(yfast.shape),
    }
    if not rep["shape_match"]:
        rep.update(max_abs_diff=float("inf"), mean_abs_diff=float("inf"),
                   rel_max=float("inf"), argmax_agree=None,
                   spike_mismatch_frac=None, bit_exact=False)
        return rep

    diff = (yref - yfast).abs()
    max_abs = diff.max().item()
    rep["max_abs_diff"] = max_abs
    rep["mean_abs_diff"] = diff.mean().item()
    rep["rel_max"] = max_abs / (yref.abs().max().item() + 1e-12)
    rep["bit_exact"] = (max_abs == 0.0)

    rep["argmax_agree"] = None
    if reduce_logits and yref.ndim >= 2 and yref.shape[-1] >= 2:
        agree = (yref.argmax(-1) == yfast.argmax(-1)).float().mean().item()
        rep["argmax_agree"] = agree

    def _looks_spike(t):
        u = torch.unique(t)
        return u.numel() <= 3 and float(u.abs().max()) <= 1.0 + 1e-6
    rep["spike_mismatch_frac"] = (
        (yref != yfast).float().mean().item()
        if (_looks_spike(yref) and _looks_spike(yfast)) else None
    )
    return rep


def _format(rep: dict[str, Any]) -> str:
    parts = [f"shape_match={rep['shape_match']}",
             f"max|Δ|={rep['max_abs_diff']:.3e}",
             f"rel={rep['rel_max']:.3e}"]
    if rep.get("argmax_agree") is not None:
        parts.append(f"top1-agree={rep['argmax_agree']*100:.2f}%")
    if rep.get("spike_mismatch_frac") is not None:
        parts.append(f"spike-mismatch={rep['spike_mismatch_frac']*100:.2f}%")
    return "  ".join(parts)


@torch.no_grad()
def assert_equivalent(reference, fast, example_input, *,
                      atol: float = 0.0, require_bitexact: bool = True,
                      min_argmax_agree: float = 1.0, verbose: bool = True):
    """断言加速模型与原模型在给定输入上等价；不满足则抛 AssertionError。

    默认（``require_bitexact=True, atol=0``）要求**逐位一致** —— 用于
    ``fold_bn=False`` 的加速路径或纯 neuron 替换，能 100% 保证推理结果不变。

    若用了 ``fold_bn=True`` 等有损融合，设 ``require_bitexact=False`` 并给一个
    合理的 ``atol``（同时仍校验 ``min_argmax_agree`` 默认要求 top-1 完全不变）。

    报错信息会指出最可能的原因（拓扑接错 / BN 折叠）与修复建议。

    Returns: 校验通过时返回度量字典（同 :func:`compare_models`）。
    """
    rep = compare_models(reference, fast, example_input)
    if verbose:
        print("[snn_compiler.verify] " + _format(rep))

    if not rep["shape_match"]:
        raise AssertionError(
            f"输出形状不一致：reference={rep['ref_shape']} vs fast={rep['fast_shape']}。"
            "几乎一定是拓扑接错（例如残差/downsample/最后的时间维归约方式不同）。"
        )

    ok = rep["bit_exact"] if require_bitexact else (rep["max_abs_diff"] <= atol)
    if rep["argmax_agree"] is not None and rep["argmax_agree"] < min_argmax_agree:
        ok = False

    if not ok:
        agree = rep["argmax_agree"]
        # 诊断：相对差很小 → 多半是数值（BN 折叠）；相对差大 → 多半是拓扑接错。
        small_numeric = (0 < rep["max_abs_diff"] and rep["rel_max"] <= 0.05
                         and (agree is None or agree >= 0.9))
        if small_numeric:
            hint = (
                "差异很小且 top-1 几乎不变 —— 最可能是 **BN 折叠的数值扰动**"
                "（fold_bn=True 在脉冲硬阈值下翻转了个别边界脉冲）。"
                "若需逐位一致，请对所有 FusedConvBN(Add)Neuron 用 fold_bn=False；"
                "若可接受该误差，请用 require_bitexact=False 并设合适的 atol。"
            )
        else:
            hint = (
                "差异较大 —— 最可能是 **拓扑接错**：检查残差是加在神经元之前还是之后"
                "（SEW-ResNet 加在之后、downsample 自带神经元），多分支合流是否对齐，"
                "以及最后时间维的归约（mean/sum/rate）是否与原网络一致。"
                "可用 snn_compiler.verify.compare_models 配合逐层 forward hook 定位。"
            )
        raise AssertionError(
            "加速模型与原模型不等价：" + _format(rep) + "\n  → " + hint
        )
    return rep
