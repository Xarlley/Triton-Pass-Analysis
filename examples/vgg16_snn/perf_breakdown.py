"""perf_breakdown.py — 定位 eager (cuDNN+cuBLAS) vs 全 Triton 的性能差距具体出现在哪几类 kernel。

测过的事实：
  BATCH=50, T=4, RTX 5070 Ti，path B (VGG16SNN, BN+MaxPool):
    eager + cuDNN/cuBLAS   : 约 7.41 ms / 张  (来自 SpikingJelly-Triton-Patch.md §7.1)
    torch.compile + Triton : 约 9.39 ms / 张
    →  Triton 慢约 1.98 ms / 张 (约 +27%)，差距 ≈ 110 ms / 一次 BATCH=50 forward

  本脚本不重复测墙钟，而是用 torch.profiler 抓 per-kernel self CUDA time 并按算子
  类别聚合，对每一类 kernel 给出 (eager_us, compile_us, diff_us)，定位差距来源。

  目的：回答「这 ~2 ms / 张的差距究竟落在哪几类 kernel」。

输出（落到 Document/IR-Trace/perf_breakdown/）:
  - eager_kernels.txt       eager 模式每个 CUDA kernel 的 self/total 时间 + 调用次数
  - compile_kernels.txt     全 Triton 模式同上
  - breakdown.txt           按算子类别聚合的对照表（核心产物）

用法:
    BATCH=50 python examples/vgg16_snn/perf_breakdown.py
    # （可选）BATCH=56 也能跑，等价路径 B 显存上限
"""
import os
import sys
import pathlib
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch
import torch.nn as nn
import torch._dynamo
import torch._inductor.config as inductor_cfg
from torch.profiler import profile, ProfilerActivity, record_function

from spikingjelly.activation_based import functional, neuron, layer
from vgg16_test import (
    VGG16SNN, NUM_CLASSES,
    patch_spikingjelly_for_full_graph,
)

BATCH = int(os.environ.get("BATCH", 50))
T = 4
SEED = 42

OUT = pathlib.Path("/home/charlley/Code/Triton-Pass-Analysis/Document/IR-Trace/perf_breakdown")
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda")


# ---------------- kernel 分类 ----------------
def categorize(kernel_name: str) -> str:
    """把一个 CUDA kernel 名归入 conv / pool / gemm / BN / LIF / elementwise / memcpy / other。
    eager 路径里 cuDNN/cuBLAS 内核与 ATen native CUDA 内核名带特征字串；
    compile 路径里 Inductor 生成 Triton kernel 名同样有规律。"""
    k = kernel_name

    # ---- SJ 手写 LIF (eager 与 compile 共用，结果应近似一致) ----
    if "_multistep_lif" in k or "multistep_lif_forward" in k:
        return "LIF (SJ Triton)"

    # ---- compile 模式: Inductor 生成的 Triton kernel ----
    if "triton_tem_fused_convolution" in k:
        return "conv (Inductor Triton tem)"
    if "triton_poi_fused__native_batch_norm" in k:
        return "BN+conv-epilogue (Inductor Triton poi)"
    if "triton_poi_fused_convolution_max_pool" in k:
        return "MaxPool+conv-epilogue (Inductor Triton poi)"
    if "triton_poi_fused_max_pool" in k:
        return "MaxPool (Inductor Triton poi)"
    if "triton_poi_fused_convolution" in k:
        return "conv-epilogue only (Inductor Triton poi)"
    if "triton_tem_fused_addmm" in k:
        return "gemm (Inductor Triton tem)"
    if "triton_poi_fused" in k:
        return "elementwise (Inductor Triton poi)"

    # ---- eager 模式: cuDNN / cuBLAS / ATen native ----
    if "cutlass__5x_cudnn" in k or "cutlass_tensorop" in k:
        return "conv (cuDNN cutlass)"
    if "sm80_xmma_fprop" in k or "implicit_convolve_sgemm" in k:
        return "conv (cuDNN xmma/sgemm)"
    if "nchwToNhwc" in k or "nhwcToNchw" in k:
        return "conv (cuDNN layout xform)"
    if "gemmSN_" in k or "ampere_sgemm" in k or "cublas" in k.lower():
        return "gemm (cuBLAS)"
    if "max_pool2d_with_indices_out_frame" in k or ("max_pool" in k and "at::native" in k):
        return "MaxPool (ATen native CUDA)"
    if "avg_pool2d_out" in k or ("avg_pool" in k and "at::native" in k):
        return "AvgPool (ATen native CUDA)"
    # cuDNN 的 BN 推理 kernel 实际叫 `cudnn::bn_fw_inf_*_kernel_NCHW`，名字里不含 batch_norm
    if "batch_norm" in k.lower() or "bn_fw_inf" in k or "bn_bw_" in k:
        return "BN (cuDNN/ATen)"
    if "elementwise_kernel" in k or "vectorized_elementwise" in k or "FillFunctor" in k:
        return "elementwise (ATen native CUDA)"
    if "reduce_kernel" in k:
        return "reduce (ATen native CUDA)"
    # cutlass_80_simt_sgemm_*_align1 是 cuBLASLt 在 sm_80 走的 sgemm cutlass 模板，归 gemm
    if "cutlass_80_simt_sgemm" in k or "cutlass_80_wmma_sgemm" in k:
        return "gemm (cuBLAS cutlass)"
    if "winograd" in k:
        return "conv (cuDNN winograd)"
    if "vectorized_layer_norm" in k or "layer_norm" in k.lower():
        return "LayerNorm (ATen)"

    if "Memcpy" in k or "Memset" in k:
        return "memcpy/memset"
    return "other"


def build_model():
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    m = VGG16SNN(NUM_CLASSES)
    functional.set_step_mode(m, "m")
    m.eval().cuda()
    return m


def fresh_input():
    g = torch.Generator(device="cuda").manual_seed(SEED)
    return torch.randn(T, BATCH, 3, 224, 224, generator=g, device=DEVICE)


def profile_one(mode_label: str, callable_fn, x, model_for_reset,
                warmup=3, n_iters=3):
    """跑 warmup 次预热 + n_iters 次 profile 累计。

    n_iters > 1 用于让 Triton autotune 在 warmup 内彻底走稳：
    第 1-2 个 warmup 触发 SJ multistep_lif 的 4 个 autotune cfg trial 编译，
    第 3 个 warmup 时 cache 已命中只调最优 cfg。
    profile 阶段跑 n_iters 次后求平均能进一步抹平噪声。"""
    for _ in range(warmup):
        functional.reset_net(model_for_reset)
        with torch.no_grad():
            out = callable_fn(x)
            if isinstance(out, tuple): out = out[0]
        torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function(f"forward_{mode_label}"), torch.no_grad():
            for _ in range(n_iters):
                functional.reset_net(model_for_reset)
                out = callable_fn(x)
                if isinstance(out, tuple): out = out[0]
        torch.cuda.synchronize()

    return prof, n_iters


def is_gpu_kernel_name(name: str) -> bool:
    """判断是否是真正的 GPU kernel 名（而非 ATen op wrapper、record_function 标记等）。
    特征:
      - CUDA kernel 函数名以 'void ' 开头（C++ ABI demangle 后）
      - Inductor 生成 Triton kernel 名以 'triton_' 开头
      - SJ 手写 Triton kernel 是 '_multistep_lif_forward_kernel' (+ 可选数字后缀)
      - 设备内存操作: 'Memcpy *' / 'Memset *'
    其他都是 CPU 端的 ATen op / dispatcher / Python wrapper，应过滤掉避免双重计入。"""
    return (
        name.startswith("void ")
        or name.startswith("triton_")
        or name.startswith("_multistep_")
        or name.startswith("Memcpy")
        or name.startswith("Memset")
    )


def to_kernel_rows(prof):
    """从 profiler events 抽**真正的 GPU kernel 行**: [(name, self_us, calls, total_us)]。
    torch 2.11 把属性名从 self_cuda_time_total 改为 self_device_time_total。

    严格过滤掉 ATen op wrapper / record_function 标记 / Python dispatcher 行，
    只保留真正在 GPU 上跑的 kernel；同名行（profiler 偶尔返回重复 key）合并保留唯一一条。"""
    seen = {}
    for ev in prof.key_averages():
        name = ev.key
        if not is_gpu_kernel_name(name):
            continue
        self_us = getattr(ev, "self_device_time_total", getattr(ev, "self_cuda_time_total", 0))
        total_us = getattr(ev, "device_time_total", getattr(ev, "cuda_time_total", 0))
        if self_us <= 0 and total_us <= 0:
            continue
        # 同名取首次记录（profiler 偶尔返回同一 kernel 的两份汇总，self_us 一致）
        if name in seen:
            continue
        seen[name] = (name, self_us, ev.count, total_us)
    rows = list(seen.values())
    rows.sort(key=lambda r: -r[1])
    return rows


def write_kernel_table(path, rows, title):
    with open(path, "w") as f:
        f.write(f"# {title}\n")
        f.write(f"# 列: rank | self_cuda_us | calls | total_cuda_us | name\n\n")
        f.write(f"{'rank':>4}  {'self_us':>10}  {'calls':>5}  {'total_us':>10}  name\n")
        f.write("-" * 100 + "\n")
        for i, (name, self_us, n, total_us) in enumerate(rows):
            f.write(f"{i+1:>4}  {self_us:>10.1f}  {n:>5}  {total_us:>10.1f}  {name}\n")


def aggregate_by_category(rows):
    by_cat = {}
    for name, self_us, n, total_us in rows:
        cat = categorize(name)
        if cat not in by_cat:
            by_cat[cat] = {"self_us": 0.0, "calls": 0, "n_kernels": 0}
        by_cat[cat]["self_us"] += self_us
        by_cat[cat]["calls"] += n
        by_cat[cat]["n_kernels"] += 1
    return by_cat


def main():
    print(f"BATCH = {BATCH}    T = {T}    device = {DEVICE}")
    print(f"输出目录: {OUT}")
    x = fresh_input()

    # -------------------- 1) eager 模式 --------------------
    print("\n========== Mode A: EAGER (cuDNN + cuBLAS + ATen native + SJ LIF Triton) ==========")
    m_eager = build_model()
    prof_eager, iters_eager = profile_one("eager", m_eager, x, m_eager)
    rows_eager = to_kernel_rows(prof_eager)
    # 把 self_us 平摊到单次 forward (除以 n_iters)
    rows_eager = [(n, s / iters_eager, max(1, c // iters_eager), t / iters_eager)
                  for (n, s, c, t) in rows_eager]
    write_kernel_table(OUT / "eager_kernels.txt", rows_eager,
                       f"EAGER kernel 表 (BATCH={BATCH}, T={T})")
    by_eager = aggregate_by_category(rows_eager)
    eager_total = sum(c["self_us"] for c in by_eager.values())
    print(f"  捕获 kernel 行数: {len(rows_eager)}, 总 self_cuda = {eager_total/1e3:.2f} ms")
    del m_eager
    import gc; gc.collect(); torch.cuda.empty_cache()

    # -------------------- 2) 全 Triton 编译 --------------------
    print("\n========== Mode B: torch.compile + 全 Triton (Inductor 接管 ATen 算子) ==========")
    torch._dynamo.config.recompile_limit = 256
    torch._dynamo.config.cache_size_limit = 256
    inductor_cfg.max_autotune = True
    inductor_cfg.max_autotune_gemm_backends = "TRITON"
    inductor_cfg.max_autotune_conv_backends = "TRITON"
    inductor_cfg.force_disable_caches = True
    patch_spikingjelly_for_full_graph()

    m_compile = build_model()
    compiled = torch.compile(m_compile)

    # 先冷启动一次完成 compile + autotune（这一段不计入 profile）
    print(f"  冷启动 compile + autotune (~50-120s)...")
    t_cold0 = time.perf_counter()
    functional.reset_net(m_compile)
    with torch.no_grad():
        compiled(x)
    torch.cuda.synchronize()
    print(f"  冷启动耗时: {time.perf_counter()-t_cold0:.1f}s")

    prof_compile, iters_compile = profile_one("compile", compiled, x, m_compile)
    rows_compile = to_kernel_rows(prof_compile)
    rows_compile = [(n, s / iters_compile, max(1, c // iters_compile), t / iters_compile)
                    for (n, s, c, t) in rows_compile]
    write_kernel_table(OUT / "compile_kernels.txt", rows_compile,
                       f"COMPILE 全 Triton kernel 表 (BATCH={BATCH}, T={T})")
    by_compile = aggregate_by_category(rows_compile)
    compile_total = sum(c["self_us"] for c in by_compile.values())
    print(f"  捕获 kernel 行数: {len(rows_compile)}, 总 self_cuda = {compile_total/1e3:.2f} ms")

    # -------------------- 3) 类别对照 --------------------
    print("\n\n========== 按算子类别聚合对照 ==========")
    all_cats = sorted(set(by_eager) | set(by_compile))
    rows_cmp = []
    for cat in all_cats:
        e = by_eager.get(cat, {"self_us": 0.0, "calls": 0, "n_kernels": 0})
        c = by_compile.get(cat, {"self_us": 0.0, "calls": 0, "n_kernels": 0})
        diff = c["self_us"] - e["self_us"]
        rows_cmp.append((cat, e["self_us"], e["calls"], c["self_us"], c["calls"], diff))

    # 按 diff 绝对值降序排
    rows_cmp.sort(key=lambda r: -abs(r[-1]))

    header = f"{'algorithm category':<46}  {'eager_us':>10}  {'eager_n':>7}  {'compile_us':>10}  {'compile_n':>9}  {'Δ (us)':>11}  {'Δ (%)':>8}"
    sep = "-" * len(header)

    print(header); print(sep)
    lines_for_file = [header, sep]
    for cat, e_us, e_n, c_us, c_n, diff in rows_cmp:
        pct = (diff / e_us * 100) if e_us > 0 else float("inf") if diff > 0 else 0
        pct_str = f"{pct:+.1f}%" if e_us > 0 else ("(new)" if diff > 0 else "")
        line = (f"{cat:<46}  {e_us:>10.1f}  {e_n:>7}  {c_us:>10.1f}  {c_n:>9}  "
                f"{diff:>+11.1f}  {pct_str:>8}")
        print(line); lines_for_file.append(line)

    print(sep)
    summary = (
        f"{'TOTAL':<46}  {eager_total:>10.1f}  {sum(c['calls'] for c in by_eager.values()):>7}  "
        f"{compile_total:>10.1f}  {sum(c['calls'] for c in by_compile.values()):>9}  "
        f"{compile_total - eager_total:>+11.1f}  "
        f"{(compile_total - eager_total)/eager_total*100:+.1f}%"
    )
    print(summary); lines_for_file.append(sep); lines_for_file.append(summary)
    print()
    print(f"折算单张延迟差异 = {(compile_total - eager_total) / BATCH:+.2f} us/张 "
          f"(总样本 = {BATCH}*1 BATCH per forward)")

    with open(OUT / "breakdown.txt", "w") as f:
        f.write(f"# perf_breakdown.py 输出（BATCH={BATCH}, T={T}）\n")
        f.write(f"# eager_total_cuda_us = {eager_total:.1f}\n")
        f.write(f"# compile_total_cuda_us = {compile_total:.1f}\n")
        f.write(f"# diff = {compile_total - eager_total:+.1f} us "
                f"({(compile_total - eager_total)/eager_total*100:+.1f}%)\n")
        f.write(f"# 折算单张 = {(compile_total - eager_total) / BATCH:+.2f} us / 张\n\n")
        for line in lines_for_file:
            f.write(line + "\n")

    # 也导出 chrome trace 以便后续可视化
    prof_eager.export_chrome_trace(str(OUT / "eager_trace.json"))
    prof_compile.export_chrome_trace(str(OUT / "compile_trace.json"))

    print(f"\n产物落在 {OUT}/")


if __name__ == "__main__":
    main()
