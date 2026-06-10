"""共享基准工具：GPU 占用门控 + 冷启动感知计时。

本机是共享 A100，且 Triton/cudnn/分配器都有冷启动，所以：
- `gpu_guard()`：测速前检查**他人进程**占用与利用率，污染时显式告警。
- `bench()`：先 warmup（吃掉 autotune/JIT/分配器冷启动），再取多次迭代**中位数** + p10/p90。
"""
import os, time, subprocess, statistics
import torch


def _smi(query, extra=""):
    cmd = ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"]
    return subprocess.check_output(cmd).decode().strip()


def gpu_other_usage():
    """返回 (他人进程占用显存MiB, [(pid,mem)...])，排除本进程。"""
    mypid = os.getpid()
    other, procs = 0, []
    try:
        out = _smi("compute-apps=pid,used_memory")
    except Exception:
        return 0, []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        pid, mem = [x.strip() for x in line.split(",")[:2]]
        if int(pid) != mypid:
            other += int(mem)
            procs.append((pid, mem))
    return other, procs


def gpu_guard(max_other_mb=400, tag=""):
    """打印并返回 GPU 卫生状态；他人占用过多时告警（不强制退出，交调用方判断）。"""
    other, procs = gpu_other_usage()
    try:
        util = int(_smi("gpu=utilization.gpu").splitlines()[0])
    except Exception:
        util = -1
    clean = other <= max_other_mb
    flag = "CLEAN" if clean else "DIRTY"
    print(f"[gpu-guard{(' '+tag) if tag else ''}] {flag}: other-proc-mem={other}MiB util={util}% other-procs={procs}")
    return {"clean": clean, "other_mb": other, "util": util, "procs": procs}


def bench(fn, warmup=25, iters=100, sync=True):
    """冷启动感知计时：warmup 后取 iters 次的中位数（ms）。返回 dict。"""
    for _ in range(warmup):
        fn()
    if sync:
        torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return {
        "median_ms": ts[len(ts) // 2],
        "p10_ms": ts[max(0, len(ts) // 10)],
        "p90_ms": ts[min(len(ts) - 1, int(len(ts) * 0.9))],
        "min_ms": ts[0],
        "iters": iters, "warmup": warmup,
    }


def fmt(d):
    return f"median={d['median_ms']:.4f}ms (p10={d['p10_ms']:.4f} p90={d['p90_ms']:.4f} min={d['min_ms']:.4f})"
