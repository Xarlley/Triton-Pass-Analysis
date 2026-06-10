"""分块系统实验（横向对比）：固定网络，测 (时间步 T × 分块大小 chunk) 下的
【峰值显存】【稳态推理时间】【冷启动/编译时间】，导出显存比/速度比/甜点。

关键背景：snn_compiler 的融合 LIF kernel 用 `tl.static_range(0, T)`（编译期展开），其中传给 kernel 的
"T" = 全 T 路径的 T 或分块路径的 chunk 长度。**展开长度越大，triton 编译（冷启动）越慢、且超线性**
（实测 full-T 在 T=128 时编译 >170s，T=64 则秒级）。故本实验把"展开长度"控制在 ≤64（chunk≤64、full 仅在 T=64），
并**显式测量冷启动/编译时间**作为第三个维度。

两个干扰按要求处理：
- **残留内存**：每个 (mode,T,chunk) 单元在【全新子进程/全新 CUDA 上下文】里跑，进程退出由 OS 回收 → 彻底隔离。
- **冷启动**：单独记录"首次调用时间（含编译/autotune/分配器）"；稳态时间取 warmup 之后的中位数（剔除冷启动）。

复用 oom_fallback_demo 的 ConvSNNStack（同一网络、snn_compiler 公开 API），不修改 snn_compiler。
跑法：python experiments/large-T-oom-fallback/chunk_sweep.py
单元：... chunk_sweep.py --worker {full|chunked} <T> <chunk> <B> <H>
"""
import sys, os, time, subprocess
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from oom_fallback_demo import ConvSNNStack

dev = "cuda"
GiB = 1024 ** 3
TOTAL_GiB = torch.cuda.get_device_properties(0).total_memory / GiB


def worker(mode, T, chunk, B, H, warmup_extra=2, iters=3):
    torch.manual_seed(0)
    model = ConvSNNStack(in_ch=3, num_classes=100).cuda().eval().to(torch.bfloat16)
    try:
        if mode == "full":
            x = torch.randn(T, B, 3, H, H, device=dev, dtype=torch.bfloat16)
            fn = lambda: model.full_T(x)
        else:
            gen = lambda i, c: torch.randn(c, B, 3, H, H, device=dev, dtype=torch.bfloat16)
            fn = lambda: model.chunked(gen, T, B, chunk)
        # 冷启动：首次调用（含 triton 编译 + cudnn autotune + 分配器首触发）
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(); torch.cuda.synchronize()
        cold = (time.perf_counter() - t0) * 1e3
        # 再多预热几次，进入稳态
        for _ in range(warmup_extra):
            fn()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(iters):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fn(); torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        peak = torch.cuda.max_memory_allocated() / GiB
        ts.sort(); steady = ts[len(ts) // 2]
        print(f"PEAK={peak:.3f} STEADY={steady:.3f} COLD={cold:.1f} STATUS=ok", flush=True)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        st = "oom" if "out of memory" in str(e).lower() else f"err:{type(e).__name__}"
        print(f"PEAK=nan STEADY=nan COLD=nan STATUS={st}", flush=True)


def run_cell(mode, T, chunk, B, H, timeout=240):
    try:
        r = subprocess.run([sys.executable, __file__, "--worker", mode, str(T), str(chunk),
                            str(B), str(H)], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return dict(status="timeout")          # 编译/运行超时（如展开过大编译过慢）
    for line in r.stdout.splitlines():
        if line.startswith("PEAK="):
            kv = dict(p.split("=") for p in line.split())
            return dict(status=kv["STATUS"],
                        peak=(None if kv["PEAK"] == "nan" else float(kv["PEAK"])),
                        steady=(None if kv["STEADY"] == "nan" else float(kv["STEADY"])),
                        cold=(None if kv["COLD"] == "nan" else float(kv["COLD"])))
    err = (r.stderr or "")[-200:]
    return dict(status="oom" if "out of memory" in err.lower() else f"crash(rc={r.returncode})")


def fmt(d, key, w=9):
    if not d:
        return f"{'-':>{w}}"
    if d.get("status") == "ok":
        v = d.get(key)
        if v is None:
            return f"{'-':>{w}}"
        return f"{v:>{w}.2f}" if key == "peak" else f"{v:>{w}.1f}"
    label = {"oom": "OOM", "timeout": "TMO"}.get(d["status"], "ERR")
    return f"{label:>{w}}"


def main():
    B, H = 8, 112
    print(f"GPU: {torch.cuda.get_device_name(0)}  total={TOTAL_GiB:.1f} GiB  torch={torch.__version__}")
    print(f"网络: ConvSNNStack（5 层 conv-bn-LIF + 头）B={B} H=W={H} bf16；子进程隔离 + 冷启动单列记录\n")

    res = {}
    # 组 1：固定 T=64（full 跑得通、展开≤64 编译可控），扫 chunk —— 核心权衡曲线
    T0 = 64
    g1_chunks = [1, 2, 4, 8, 16, 32, 64]
    res[("full", T0)] = run_cell("full", T0, T0, B, H)
    for ck in g1_chunks:
        res[("chunked", T0, ck)] = run_cell("chunked", T0, ck, B, H)
    # 组 2：大 T + 小 chunk（展开小、编译快、显存随 chunk 不随 T）；含 full 以示其 OOM/编译爆炸
    g2 = {128: [8, 32], 256: [8, 32], 512: [8, 32]}
    for T, cks in g2.items():
        for ck in cks:
            res[("chunked", T, ck)] = run_cell("chunked", T, ck, B, H)
    res[("full", 128)] = run_cell("full", 128, 128, B, H, timeout=260)   # 慢编译演示（~170s+）

    # ---- 组 1 表：chunk 权衡 @ T=64 ----
    print("=== 组1：固定 T=64，扫 chunk —— 峰值显存 / 稳态时间 / 冷启动(编译)时间 ===")
    print(f"{'配置':>12} | {'峰值GiB':>9} | {'稳态ms':>9} | {'冷启动ms':>10} | {'显存比':>7} | {'速度比':>7}")
    full = res[("full", T0)]
    pbase = full["peak"] if full["status"] == "ok" else None
    tbase = full["steady"] if full["status"] == "ok" else None
    rows = [("full(全T)", full)] + [(f"chunk={ck}", res[("chunked", T0, ck)]) for ck in g1_chunks]
    for name, d in rows:
        mr = f"{d['peak']/pbase:.2f}" if (d.get("status") == "ok" and pbase) else "-"
        sr = f"{d['steady']/tbase:.2f}" if (d.get("status") == "ok" and tbase) else "-"
        print(f"{name:>12} | {fmt(d,'peak')} | {fmt(d,'steady')} | {fmt(d,'cold',10)} | {mr:>7} | {sr:>7}")

    # ---- 组 2 表：大 T，显存是否随 T 不变 ----
    print("\n=== 组2：大 T + 小 chunk —— 峰值显存(GiB) / 稳态时间(ms)；对照 full ===")
    print(f"{'T':>5} | {'full':>14} | {'chunk=8':>16} | {'chunk=32':>16}")
    for T in [128, 256, 512]:
        f8 = res.get(("chunked", T, 8)); f32 = res.get(("chunked", T, 32))
        fl = res.get(("full", T))
        def c(d):
            if not d:
                return f"{'-':>14}"
            if d["status"] != "ok":
                label = {"oom": "OOM", "timeout": "TMO(编译)"}.get(d["status"], "ERR")
                return f"{label:>14}"
            return f"{d['peak']:.2f}GiB/{d['steady']:.0f}ms".rjust(14)
        print(f"{T:>5} | {c(fl):>14} | {c(f8):>16} | {c(f32):>16}")

    # ---- 甜点 ----
    print("\n=== 解读 ===")
    print("- 显存：分块峰值 ∝ chunk（不随 T），全 T 峰值 ∝ T → 大 T 必须分块。")
    print("- 稳态速度：chunk 越小、kernel 启动越多 → 越慢；存在一个'够用就好'的最大 chunk。")
    print("- 冷启动：展开长度(=chunk 或 T)越大，triton 编译越慢且超线性 → 大 chunk/全 T 还要付高编译税。")
    print("\nCHUNK_SWEEP_DONE")


if __name__ == "__main__":
    if len(sys.argv) >= 7 and sys.argv[1] == "--worker":
        worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
    else:
        main()
