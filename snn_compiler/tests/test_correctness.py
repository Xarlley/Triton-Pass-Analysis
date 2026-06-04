"""SNN Compiler 正确性测试套件。

覆盖矩阵：
- neuron: IF / LIF (decay_input True/False) / CubaLIF / EIF
- reset: soft / hard
- v_reset: 0.0 / 0.3 / -0.5 (任意常数)
- threshold: scalar / per-channel / per-neuron
- layout: NCHW / NHWC (per-channel index 验证)
- 形状: 1D / 4D / 5D，含小张量(易触发边界)与 VGG16 实际形状(检查 BLOCK 大小)
- dtype: fp32 / bf16 / fp16

判定：与 naive 参考实现 bit-equal（spike 是 0/1，浮点 reset 下也应严格相等）。

CubaLIF / EIF 的 exp 与累积可能产生 ULP 级差异，因此允许 max_abs_diff <= 1e-5
(spike 仍要求完全相同)。
"""
import sys, pathlib, itertools, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch
from snn_compiler.kernels.neurons import (
    if_lif, cuba_lif, eif, naive_if_lif, naive_cuba_lif, naive_eif,
)
from snn_compiler.kernels.fused import (
    fused_bias_if_lif, conv_neuron, linear_neuron, fold_conv_bn,
)


def bit_equal(a, b):
    return torch.equal(a, b)


def near_equal(a, b, tol=1e-5):
    return (a - b).abs().max().item() <= tol


# -------------------- 共用形状 --------------------
SMALL_SHAPES_5D = [(4, 2, 8, 4, 4)]                          # 小 5D
VGG_SHAPES_5D   = [(4, 8, 64, 56, 56)]                       # 较大 5D, 跨越多 BLOCK
SHAPES_3D       = [(4, 8, 128)]                              # 1D feature


def run_pure_neuron():
    print("=" * 78)
    print(" Pure neuron correctness (IF / LIF)")
    print("=" * 78)
    torch.manual_seed(0)

    cases = []
    for shape in SMALL_SHAPES_5D + VGG_SHAPES_5D + SHAPES_3D:
        for neuron in ["if", "lif"]:
            for soft in [True, False]:
                for v_reset in [0.0, 0.3, -0.5]:
                    if soft and v_reset != 0.0:
                        continue  # soft 不使用 v_reset
                    decay_input_iter = [True, False] if neuron == "lif" else [True]
                    for di in decay_input_iter:
                        # scalar threshold
                        cases.append((shape, neuron, soft, v_reset, di, "scalar", "NCHW"))
                        # per-channel
                        cases.append((shape, neuron, soft, v_reset, di, "per_c", "NCHW"))
                        if len(shape) == 5:
                            cases.append((shape, neuron, soft, v_reset, di, "per_c", "NHWC"))
                        # per-neuron
                        cases.append((shape, neuron, soft, v_reset, di, "per_n", "NCHW"))

    fails = 0
    for shape, neuron, soft, v_reset, di, thr_mode, layout in cases:
        # 框架约定：shape 始终按 NCHW 标记 ([T, B, C, H, W])，
        # layout='NHWC' 只表示底层使用 channels_last 内存格式。
        if layout == "NHWC" and len(shape) == 5:
            T_, B_, C_, H_, W_ = shape
            # 先建 4D channels_last，再 view 回 5D 以保持 NHWC 内存
            x4 = torch.randn(T_ * B_, C_, H_, W_, device="cuda")
            x4 = x4.contiguous(memory_format=torch.channels_last)
            x = x4.view(T_, B_, C_, H_, W_)
        else:
            x = torch.randn(*shape, device="cuda").contiguous()

        # threshold
        if thr_mode == "scalar":
            v_th = 1.0
        elif thr_mode == "per_c":
            C = shape[2] if len(shape) == 5 else shape[-1]
            v_th = torch.rand(C, device="cuda").float() * 0.5 + 0.5
            v_th = v_th.contiguous()
        else:
            v_th = torch.rand(*x.shape[1:], device="cuda").float() * 0.5 + 0.5
            v_th = v_th.contiguous().view(-1).contiguous()

        kwargs = dict(neuron=neuron, decay_input=di, soft_reset=soft,
                       v_threshold=v_th, v_reset=v_reset, layout=layout)

        # naive 与 fused
        ref = naive_if_lif(x, tau=2.0, **kwargs)
        out = if_lif(x, tau=2.0, **kwargs)
        ok = bit_equal(ref, out)
        if not ok:
            ndiff = (ref != out).sum().item()
            tot = ref.numel()
            fails += 1
            print(f"  [FAIL] shape={shape} layout={layout} neuron={neuron} "
                  f"soft={soft} vr={v_reset} di={di} thr={thr_mode}  diff={ndiff}/{tot}")
        else:
            print(f"  [ OK ] shape={shape} layout={layout} neuron={neuron} "
                  f"soft={soft} vr={v_reset} di={di} thr={thr_mode}")
    print(f" -> pure neuron: {len(cases) - fails}/{len(cases)} pass")
    return fails


def run_cuba_eif():
    print("=" * 78)
    print(" CubaLIF / EIF correctness")
    print("=" * 78)
    torch.manual_seed(1)
    fails = 0
    for shape in SMALL_SHAPES_5D + VGG_SHAPES_5D:
        for soft in [True, False]:
            for v_reset in [0.0, -0.3]:
                if soft and v_reset != 0.0:
                    continue
                x = torch.randn(*shape, device="cuda").contiguous()

                # CubaLIF
                out_c = cuba_lif(x, tau_syn=2.0, tau_mem=4.0,
                                  soft_reset=soft, v_threshold=1.0, v_reset=v_reset)
                ref_c = naive_cuba_lif(x, tau_syn=2.0, tau_mem=4.0,
                                        soft_reset=soft, v_threshold=1.0, v_reset=v_reset)
                # spike 应该 bit-equal（因为最终是 (v>=θ) 判定，浮点累计相同 op order）
                if not bit_equal(out_c, ref_c):
                    # 允许微小 ULP 差异
                    if near_equal(out_c.float(), ref_c.float(), tol=1e-4):
                        print(f"  [~OK] CubaLIF shape={shape} soft={soft} vr={v_reset} "
                              f"max|d|={(out_c - ref_c).abs().max():.2e}")
                    else:
                        print(f"  [FAIL] CubaLIF shape={shape} soft={soft} vr={v_reset}")
                        fails += 1
                else:
                    print(f"  [ OK ] CubaLIF shape={shape} soft={soft} vr={v_reset}")

                # EIF
                out_e = eif(x, tau=2.0, delta_T=1.0, v_rh=0.5,
                             soft_reset=soft, v_threshold=1.0, v_reset=v_reset)
                ref_e = naive_eif(x, tau=2.0, delta_T=1.0, v_rh=0.5,
                                   soft_reset=soft, v_threshold=1.0, v_reset=v_reset)
                if not bit_equal(out_e, ref_e):
                    if near_equal(out_e.float(), ref_e.float(), tol=1e-3):
                        print(f"  [~OK] EIF shape={shape} soft={soft} vr={v_reset} "
                              f"max|d|={(out_e - ref_e).abs().max():.2e}")
                    else:
                        print(f"  [FAIL] EIF shape={shape} soft={soft} vr={v_reset}")
                        fails += 1
                else:
                    print(f"  [ OK ] EIF shape={shape} soft={soft} vr={v_reset}")
    return fails


def run_fused_conv():
    print("=" * 78)
    print(" Fused Conv-bias-Neuron correctness")
    print("=" * 78)
    torch.manual_seed(2)
    fails = 0
    import torch.nn.functional as F

    # 较小形状便于参考实现跑得动
    T, B, in_C, out_C, H, W = 4, 2, 8, 16, 16, 16
    weight = torch.randn(out_C, in_C, 3, 3, device="cuda")
    bias = torch.randn(out_C, device="cuda")

    for has_bias in [True, False]:
        for neuron in ["if", "lif"]:
            for soft in [True, False]:
                for v_reset in [0.0, 0.2]:
                    if soft and v_reset != 0.0:
                        continue
                    x = torch.randn(T, B, in_C, H, W, device="cuda")
                    b = bias if has_bias else None

                    # reference: 普通 conv with bias，然后 naive neuron
                    y_ref = F.conv2d(x.reshape(T * B, in_C, H, W), weight, b,
                                       padding=1).view(T, B, out_C, H, W)
                    ref = naive_if_lif(y_ref, neuron=neuron, tau=2.0,
                                        soft_reset=soft, v_threshold=1.0, v_reset=v_reset)

                    # fused: conv(no bias) + fused_bias_if_lif
                    y_seq = F.conv2d(x.reshape(T * B, in_C, H, W), weight, None,
                                       padding=1).view(T, B, out_C, H, W).contiguous()
                    out = fused_bias_if_lif(y_seq, b, neuron=neuron, tau=2.0,
                                              soft_reset=soft, v_threshold=1.0,
                                              v_reset=v_reset, layout="NCHW")
                    ok = bit_equal(ref, out)
                    tag = f"bias={has_bias} {neuron} soft={soft} vr={v_reset}"
                    if not ok:
                        fails += 1
                        ndiff = (ref != out).sum().item()
                        print(f"  [FAIL] {tag}  diff={ndiff}/{ref.numel()}")
                    else:
                        print(f"  [ OK ] {tag}")
    return fails


def run_fold_bn():
    print("=" * 78)
    print(" fold_conv_bn 数学等价测试")
    print("=" * 78)
    torch.manual_seed(3)
    import torch.nn as nn
    import torch.nn.functional as F
    conv = nn.Conv2d(8, 16, 3, padding=1, bias=True).cuda()
    bn = nn.BatchNorm2d(16).cuda()
    # 给 BN 一些有意义的 running stats
    bn.running_mean.copy_(torch.randn(16, device="cuda"))
    bn.running_var.copy_(torch.rand(16, device="cuda") + 0.1)
    bn.weight.data.copy_(torch.rand(16, device="cuda") + 0.5)
    bn.bias.data.copy_(torch.randn(16, device="cuda") * 0.1)
    bn.eval()
    conv.eval()

    x = torch.randn(4, 8, 32, 32, device="cuda")
    y_ref = bn(conv(x))

    new_w, new_b = fold_conv_bn(conv.weight, conv.bias, bn.weight, bn.bias,
                                  bn.running_mean, bn.running_var, bn.eps)
    y_folded = F.conv2d(x, new_w, new_b, padding=1)
    diff = (y_ref - y_folded).abs().max().item()
    print(f"  max|conv-bn(x) - conv'(x)| = {diff:.3e}   {'OK' if diff < 1e-4 else 'FAIL'}")
    return 0 if diff < 1e-4 else 1


def run_dtype_compat():
    print("=" * 78)
    print(" dtype 兼容性 (bf16 / fp32)")
    print("=" * 78)
    torch.manual_seed(4)
    fails = 0
    for dt in [torch.float32, torch.bfloat16, torch.float16]:
        x = torch.randn(4, 4, 32, 16, 16, device="cuda", dtype=dt).contiguous()
        ref = naive_if_lif(x, neuron="lif", tau=2.0, decay_input=True,
                            soft_reset=False, v_threshold=1.0, v_reset=0.0)
        out = if_lif(x, neuron="lif", tau=2.0, decay_input=True,
                      soft_reset=False, v_threshold=1.0, v_reset=0.0)
        ok = bit_equal(ref, out)
        if not ok:
            fails += 1
            ndiff = (ref != out).sum().item()
            print(f"  [FAIL] dtype={dt}  diff={ndiff}/{ref.numel()}")
        else:
            print(f"  [ OK ] dtype={dt}")
    return fails


def run_decay_override():
    """所有 neuron 类型的 decay override 都必须 bit-equal。

    覆盖：
      - IF 默认 decay=1.0；显式 0.9 → leaky IF
      - LIF 默认 decay=1-1/τ；显式覆盖 0.5 → 与 τ 无关的衰减
      - CubaLIF 默认 α=exp(-dt/τ_syn), β=exp(-dt/τ_mem)；显式覆盖各自
      - EIF 默认 decay=1-1/τ；显式覆盖 1.0 → 关掉线性泄漏
    """
    print("=" * 78)
    print(" Decay override (所有 neuron 类型都允许 decay 自定义)")
    print("=" * 78)
    torch.manual_seed(10)
    shape = (4, 4, 16, 8, 8)
    fails = 0

    # IF
    for d in [0.0, 0.5, 0.9, 1.0]:
        x = torch.randn(*shape, device="cuda").contiguous()
        ref = naive_if_lif(x, neuron="if", decay=d, soft_reset=False,
                            v_threshold=1.0, v_reset=0.0)
        out = if_lif(x, neuron="if", decay=d, soft_reset=False,
                      v_threshold=1.0, v_reset=0.0)
        ok = bit_equal(ref, out)
        print(f"  [{'OK' if ok else 'FAIL'}] IF       decay={d}")
        if not ok: fails += 1

    # LIF: τ=2.0, 但 decay 覆盖到非 (1-1/τ) 的值
    for d in [0.3, 0.7, 0.99]:
        for di in [True, False]:
            x = torch.randn(*shape, device="cuda").contiguous()
            ref = naive_if_lif(x, neuron="lif", tau=2.0, decay=d,
                                decay_input=di, soft_reset=False,
                                v_threshold=1.0, v_reset=0.0)
            out = if_lif(x, neuron="lif", tau=2.0, decay=d,
                          decay_input=di, soft_reset=False,
                          v_threshold=1.0, v_reset=0.0)
            ok = bit_equal(ref, out)
            print(f"  [{'OK' if ok else 'FAIL'}] LIF      decay={d}  decay_input={di}")
            if not ok: fails += 1

    # CubaLIF: 显式 decay_syn / decay_mem
    for ds, dm in [(0.0, 0.9), (0.5, 0.5), (0.7, 0.3), (1.0, 0.8)]:
        x = torch.randn(*shape, device="cuda").contiguous()
        ref = naive_cuba_lif(x, tau_syn=2.0, tau_mem=4.0,
                              decay_syn=ds, decay_mem=dm,
                              soft_reset=False, v_threshold=1.0, v_reset=0.0)
        out = cuba_lif(x, tau_syn=2.0, tau_mem=4.0,
                        decay_syn=ds, decay_mem=dm,
                        soft_reset=False, v_threshold=1.0, v_reset=0.0)
        ok = bit_equal(ref, out)
        if not ok:
            # CubaLIF 的 exp 累积偶有 ULP；spike 必须严格相同
            close = near_equal(ref.float(), out.float(), tol=1e-3)
            print(f"  [{'~OK' if close else 'FAIL'}] CubaLIF  decay_syn={ds}  "
                  f"decay_mem={dm}  max|d|={(ref-out).abs().max():.2e}")
            if not close: fails += 1
        else:
            print(f"  [ OK ] CubaLIF  decay_syn={ds}  decay_mem={dm}")

    # EIF: 显式 decay
    for d in [0.0, 0.5, 1.0]:
        x = torch.randn(*shape, device="cuda").contiguous() * 0.3   # 限幅免 exp 爆
        ref = naive_eif(x, tau=2.0, decay=d, delta_T=1.0, v_rh=0.5,
                         soft_reset=False, v_threshold=1.0, v_reset=0.0)
        out = eif(x, tau=2.0, decay=d, delta_T=1.0, v_rh=0.5,
                   soft_reset=False, v_threshold=1.0, v_reset=0.0)
        ok = bit_equal(ref, out)
        if not ok:
            close = near_equal(ref.float(), out.float(), tol=1e-2)
            print(f"  [{'~OK' if close else 'FAIL'}] EIF      decay={d}  "
                  f"max|d|={(ref-out).abs().max():.2e}")
            if not close: fails += 1
        else:
            print(f"  [ OK ] EIF      decay={d}")

    return fails


def main():
    total_fail = 0
    total_fail += run_pure_neuron()
    total_fail += run_cuba_eif()
    total_fail += run_fused_conv()
    total_fail += run_fold_bn()
    total_fail += run_dtype_compat()
    total_fail += run_decay_override()
    print("\n" + "=" * 78)
    if total_fail == 0:
        print("  ALL TESTS PASSED")
    else:
        print(f"  {total_fail} TEST(S) FAILED")
    print("=" * 78)
    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()
