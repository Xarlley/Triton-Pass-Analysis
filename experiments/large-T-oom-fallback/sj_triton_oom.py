"""验证：仅用「最新版 spikingjelly + triton」时，同一个网络在较大网络 + 较多时间步下，
**多步 triton 后端**（一次处理整段 [T,B,...]，享受 triton 高性能）会因激活显存 ∝ T 而 OOM，
而**单步 python 循环**（逐时间步、只持 [B,...]）却能跑通推理。

环境：conda env `sj_triton`（spikingjelly 0.0.0.0.15 + triton 3.7.0 + torch 2.12），本机 RTX 5070 Ti（16 GiB）。
不修改 spikingjelly 包文件：仅运行时 monkey-patch 其 triton 3.7 下的一处 bug（convert_and_store 多写一层 .element_ty）。

显存测量用**子进程隔离**：每个 (模式, T) 在全新进程/全新 CUDA 上下文中跑，进程退出由操作系统回收
全部显存——避免 spikingjelly triton custom-op / 失败前向的残留污染同进程内后续测量（实测同进程会污染）。

跑法：~/miniconda3/envs/sj_triton/bin/python experiments/large-T-oom-fallback/sj_triton_oom.py
"""
import sys, subprocess
import torch
import torch.nn as nn

# ---- 运行时 monkey-patch：修 spikingjelly triton kernel 在 triton 3.7 下的 convert_and_store bug ----
import triton
import triton.language as tl


@triton.jit
def _convert_and_store_fixed(pointer, value, boundary_check):
    value = value.to(pointer.dtype.element_ty)            # 去掉多余的 .element_ty
    tl.store(pointer, value, boundary_check=boundary_check)


def _patch_convert_and_store():
    import importlib
    targets = [
        "spikingjelly.activation_based.triton_kernel.triton_utils",
        "spikingjelly.activation_based.triton_kernel.neuron_kernel.integrate_and_fire",
        "spikingjelly.activation_based.triton_kernel.neuron_kernel.lif",
        "spikingjelly.activation_based.triton_kernel.neuron_kernel.plif",
    ]
    n = 0
    for name in targets:
        try:
            m = importlib.import_module(name)
        except Exception:
            continue
        if hasattr(m, "convert_and_store"):
            m.convert_and_store = _convert_and_store_fixed
            n += 1
    return n


_npatch = _patch_convert_and_store()
from spikingjelly.activation_based import neuron, layer, functional

dev = "cuda"
GiB = 1024 ** 3
TOTAL_GiB = torch.cuda.get_device_properties(0).total_memory / GiB


def build_net(num_classes=100):
    """多层 conv-bn-IF 脉冲 CNN（spikingjelly 层）。大空间分辨率 → 中间激活 [T,B,C,H,W] 主导显存。"""
    net = nn.Sequential(
        layer.Conv2d(3, 64, 3, padding=1, bias=False), layer.BatchNorm2d(64), neuron.IFNode(),
        layer.Conv2d(64, 64, 3, padding=1, bias=False), layer.BatchNorm2d(64), neuron.IFNode(),
        layer.MaxPool2d(2),
        layer.Conv2d(64, 128, 3, padding=1, bias=False), layer.BatchNorm2d(128), neuron.IFNode(),
        layer.MaxPool2d(2),
        layer.Conv2d(128, 128, 3, padding=1, bias=False), layer.BatchNorm2d(128), neuron.IFNode(),
        layer.AdaptiveAvgPool2d((1, 1)), layer.Flatten(), layer.Linear(128, num_classes),
    )
    return net.to(dev).eval()


def set_mode(net, *, multistep, backend):
    functional.set_step_mode(net, "m" if multistep else "s")
    for m in net.modules():
        if isinstance(m, neuron.BaseNode):
            try:
                m.backend = backend
            except Exception:
                pass


# ============================================================
#   worker：在全新进程里跑一个 (模式, T) 测量，打印 PEAK/STATUS 后退出
# ============================================================
def _is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def worker(mode, T, B, H):
    torch.manual_seed(0)
    net = build_net()
    # equiv 模式：同一输入跑多步 triton 与单步 python，打印逐元素最大差（证明回退路径算同一个东西）
    if mode == "equiv":
        with torch.no_grad():
            x = (torch.rand(T, B, 3, H, H, device=dev) < 0.5).float()
            set_mode(net, multistep=True, backend="triton")
            functional.reset_net(net); ym = net(x)
            set_mode(net, multistep=False, backend="torch")
            functional.reset_net(net); ys = torch.stack([net(x[t]) for t in range(T)], 0)
            torch.cuda.synchronize()
        print(f"MAXDIFF={(ym - ys).abs().max().item():.3e} STATUS=ok", flush=True)
        return
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.no_grad():
            if mode == "multistep":
                set_mode(net, multistep=True, backend="triton")
                functional.reset_net(net)
                x = torch.randn(T, B, 3, H, H, device=dev)     # 全 T 输入 [T,B,...]
                y = net(x)
            else:                                              # 单步 python 循环，逐步生成输入
                set_mode(net, multistep=False, backend="torch")
                functional.reset_net(net)
                outs = []
                for t in range(T):
                    outs.append(net(torch.randn(B, 3, H, H, device=dev)))
                y = torch.stack(outs, 0)
            torch.cuda.synchronize()
        print(f"PEAK={torch.cuda.max_memory_allocated()/GiB:.3f} STATUS=ok", flush=True)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        # 显式区分「真·显存 OOM」与「其它崩溃」——避免把非 OOM 失败误记为 OOM
        if _is_oom(e):
            print("PEAK=nan STATUS=oom", flush=True)
        else:
            raise


# ============================================================
#   parent：正确性探针（同进程小测）+ 子进程隔离的 OOM 扫描
# ============================================================
def run_one(mode, T, B, H, timeout=240):
    r = subprocess.run([sys.executable, __file__, "--worker", mode, str(T), str(B), str(H)],
                       capture_output=True, text=True, timeout=timeout)
    for line in r.stdout.splitlines():
        if line.startswith("PEAK="):
            peak = line.split()[0].split("=")[1]
            status = line.split()[1].split("=")[1]
            return (None if peak == "nan" else float(peak)), status
    # 没有 PEAK 行 → 子进程崩溃。**只在 stderr 确含显存 OOM 字样时才记 oom**，
    # 否则记 crash（暴露 timeout / triton 编译错 / 非法地址 等非 OOM 失败，防「假 OOM」）。
    err = (r.stderr or "")
    if "out of memory" in err.lower() or "OutOfMemoryError" in err:
        return None, "oom"
    return None, f"crash(rc={r.returncode})"


def probe():
    print("\n=== 正确性/后端探针（小网络小 T，同进程）===")
    torch.manual_seed(0)
    net = build_net()
    Ts, Bs, Hs = 8, 2, 32
    x = (torch.rand(Ts, Bs, 3, Hs, Hs, device=dev) < 0.5).float()
    with torch.no_grad():
        set_mode(net, multistep=True, backend="triton")
        functional.reset_net(net); ym = net(x)
        set_mode(net, multistep=False, backend="torch")
        functional.reset_net(net)
        ys = torch.stack([net(x[t]) for t in range(Ts)], 0)
    print(f"  多步 triton vs 单步 python：max|Δ|={(ym-ys).abs().max().item():.3e}  (≈0 即等价)")
    del net, x, ym, ys
    torch.cuda.empty_cache()


def main():
    print(f"[patch] convert_and_store 已就地替换于 {_npatch} 个模块（不改包文件）")
    print(f"GPU: {torch.cuda.get_device_name(0)}  total={TOTAL_GiB:.1f} GiB  "
          f"torch={torch.__version__}  triton={triton.__version__}")
    print(f"spikingjelly IFNode 支持后端: {neuron.IFNode(step_mode='m').supported_backends}")
    probe()

    # 大 T 等价性（同一输入、子进程隔离）：在多步仍跑得通的较大 T 上确认逐位一致
    for Te in (32,):
        r = subprocess.run([sys.executable, __file__, "--worker", "equiv", str(Te), "16", "112"],
                           capture_output=True, text=True, timeout=300)
        md = next((l.split()[0] for l in r.stdout.splitlines() if l.startswith("MAXDIFF=")),
                  f"MAXDIFF=?(crash rc={r.returncode})")
        print(f"  大 T 等价性 T={Te}（同一输入：多步triton vs 单步python）：{md}")

    print("\n=== OOM 扫描（BATCH=16, H=W=112；多步 triton vs 单步 python 循环；子进程隔离测显存）===")
    B, H = 16, 112
    print(f"{'T':>5} | {'多步triton峰值':>13} {'多步triton':>10} | {'单步python峰值':>14} {'单步python':>10} | 结论")
    for T in (4, 16, 32, 64, 128, 256, 512, 1024):
        ms_peak, ms_state = run_one("multistep", T, B, H)
        ss_peak, ss_state = run_one("singlestep", T, B, H)
        mp = f"{ms_peak:.2f}GiB" if ms_peak else ">16"
        sp = f"{ss_peak:.2f}GiB" if ss_peak else ">16"
        ms = "OK" if ms_state == "ok" else ("OOM" if ms_state == "oom" else ms_state)  # 暴露 crash
        ss = "OK" if ss_state == "ok" else ("OOM" if ss_state == "oom" else ss_state)
        concl = "" if ms == "OK" else ("★ triton(多步) OOM，python(单步) 存活"
                                       if ss == "OK" else "两者都 OOM")
        print(f"{T:>5} | {mp:>13} {ms:>10} | {sp:>14} {ss:>10} | {concl}")
    print("\nSJ_TRITON_OOM_DONE")


if __name__ == "__main__":
    if len(sys.argv) >= 6 and sys.argv[1] == "--worker":
        worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    else:
        main()
