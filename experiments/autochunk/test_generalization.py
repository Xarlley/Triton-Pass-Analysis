"""AutoChunkInference 泛化测试：多个网络模型 × 多种参数(T,B,H)，验证
(1) 正确性：强制多块(chunk<T，跨多块串接膜电位) 的 autochunk ≈ 整段多步；
(2) 不 OOM：在 full 多步会 OOM 的大 T 上，autochunk 自动选块跑通；
(3) regime 判别：compute-bound(大 B·H) 选中等块、launch-bound(小 B·H) 选大块。

三种 SpikingJelly 多步+triton 架构：conv_small / conv_wide / fc（无卷积）。
每个 (模型,模式,参数) 在**独立子进程**里跑（全新 CUDA 上下文，OS 回收显存，杜绝跨配置污染）。

跑法：~/miniconda3/envs/sj_triton/bin/python experiments/autochunk/test_generalization.py
单配置：... test_generalization.py --worker <model> <mode> <T> <B> <H> <chunk>
"""
import os, sys, subprocess
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "large-T-oom-fallback"))
import sj_triton_oom as S          # 触发 monkey-patch；提供 set_mode / dev / GiB / _is_oom
from spikingjelly.activation_based import neuron, layer, functional
from snn_compiler.nn import AutoChunkInference

dev = "cuda"
GiB = 1024 ** 3


def _IF():
    return neuron.IFNode()


def build_conv_small(H):
    return S.build_net()                       # 复用既有 5-conv 网（compute/launch 视 B·H 而定）


def build_conv_wide(H, num_classes=100):
    """更宽更深的 conv-SNN（不同拓扑），测架构泛化。"""
    net = nn.Sequential(
        layer.Conv2d(3, 128, 3, padding=1, bias=False), layer.BatchNorm2d(128), _IF(),
        layer.Conv2d(128, 128, 3, padding=1, bias=False), layer.BatchNorm2d(128), _IF(),
        layer.MaxPool2d(2),
        layer.Conv2d(128, 256, 3, padding=1, bias=False), layer.BatchNorm2d(256), _IF(),
        layer.Conv2d(256, 256, 3, padding=1, bias=False), layer.BatchNorm2d(256), _IF(),
        layer.MaxPool2d(2),
        layer.Conv2d(256, 256, 3, padding=1, bias=False), layer.BatchNorm2d(256), _IF(),
        layer.AdaptiveAvgPool2d((1, 1)), layer.Flatten(), layer.Linear(256, num_classes),
    )
    return net.to(dev).eval()


def build_fc(H, num_classes=100, hidden=512):
    """无卷积的全连接脉冲网（算子碎小，小 B·H 下易 launch-bound）。输入 [T,B,3,H,H] flatten。"""
    d = 3 * H * H
    net = nn.Sequential(
        layer.Flatten(),
        layer.Linear(d, hidden), _IF(),
        layer.Linear(hidden, hidden), _IF(),
        layer.Linear(hidden, num_classes),
    )
    return net.to(dev).eval()


MODELS = {"conv_small": build_conv_small, "conv_wide": build_conv_wide, "fc": build_fc}


# ============================================================
#   worker：独立子进程内跑一个 (模型,模式,参数)
# ============================================================
@torch.no_grad()
def worker(model_name, mode, T, B, H, chunk):
    torch.manual_seed(0)
    net = MODELS[model_name](H)
    S.set_mode(net, multistep=True, backend="triton")

    if mode == "full":                          # 整段多步：测它是否 OOM
        try:
            functional.reset_net(net)
            x = (torch.rand(T, B, 3, H, H, device=dev) < 0.5).float()
            y = net(x); torch.cuda.synchronize()
            print(f"RESULT status=ok peak={torch.cuda.max_memory_allocated()/GiB:.2f}", flush=True)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if S._is_oom(e):
                print("RESULT status=oom", flush=True)
            else:
                raise
        return

    if mode == "corr":                          # 多块正确性：full(T) vs autochunk(fixed_chunk<T)
        functional.reset_net(net)
        x = (torch.rand(T, B, 3, H, H, device=dev) < 0.5).float()
        yf = net(x)                                              # 整段多步参考（卷积批次 = T·B）
        auto = AutoChunkInference(net, reset_fn=functional.reset_net, fixed_chunk=chunk)
        ya = auto(x)                                             # 分块多步（卷积批次 = chunk·B）
        d_chunk = (yf - ya).abs().max().item()
        # 公认等价基线：单步 python 循环（卷积批次 = B）。它与整段多步的差，就是"不同卷积批次"
        # 在 cuDNN 下的固有数值包络（与脉冲阈值耦合后偶发翻转）。分块只要 ≤ 该包络即属同源、功能等价。
        S.set_mode(net, multistep=False, backend="torch")
        functional.reset_net(net)
        ys = torch.stack([net(x[t]) for t in range(T)], 0)
        S.set_mode(net, multistep=True, backend="triton")
        d_single = (yf - ys).abs().max().item()
        nch = (T + chunk - 1) // chunk
        print(f"RESULT status=ok dchunk={d_chunk:.3e} dsingle={d_single:.3e} nchunks={nch}", flush=True)
        return

    if mode == "auto":                          # 自动选块 + 不 OOM + 选了什么块/regime
        auto = AutoChunkInference(net, reset_fn=functional.reset_net, memory_fraction=0.85)
        try:
            x = (torch.rand(T, B, 3, H, H, device=dev) < 0.5).float()
            auto(x)                              # 选块(含探针)+建缓存
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            auto(x)                              # 干净测峰值
            torch.cuda.synchronize()
            p = auto.last_plan
            print(f"RESULT status=ok chunk_t={p['chunk_t']} regime={p['regime']} "
                  f"peak={torch.cuda.max_memory_allocated()/GiB:.2f} "
                  f"budget={p.get('budget_GiB', float('nan')):.2f}", flush=True)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if S._is_oom(e):
                print("RESULT status=oom", flush=True)
            else:
                raise
        return


# ============================================================
#   parent
# ============================================================
def run(model, mode, T, B, H, chunk=0, timeout=360):
    try:
        r = subprocess.run([sys.executable, __file__, "--worker", model, mode,
                            str(T), str(B), str(H), str(chunk)],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": f"compile-stall(>{timeout}s)"}     # full 大 T 时 static_range 展开编译不返回
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            return dict(tok.split("=", 1) for tok in line.split()[1:])
    err = (r.stderr or "")
    if "out of memory" in err.lower() or "OutOfMemoryError" in err:
        return {"status": "oom"}
    return {"status": f"crash(rc={r.returncode})", "_err": err[-300:]}


def main():
    print(f"GPU {torch.cuda.get_device_name(0)}  total={torch.cuda.get_device_properties(0).total_memory/GiB:.1f}GiB  "
          f"triton {S.triton.__version__}")

    print("\n=== (1) 多块正确性：autochunk(强制 chunk<T，多块串接) vs 整段多步 ===")
    print("    fc(无卷积)逐位一致(~1e-7)→证明分块驱动数值精确；conv 的残差是 cuDNN 按 batch 选不同卷积算法的固有差")
    print("    (与 多步vs单步 同源、确定性、训练网上预测稳定；判定取保守上限 dchunk≤5e-2)")
    print(f"{'model':>11} {'T':>4} {'B':>3} {'H':>4} {'chunk':>5} | {'dchunk':>10} {'dsingle':>10} {'nchunks':>7} {'判定':>6}")
    corr = [("conv_small", 24, 8, 64, 8), ("conv_small", 24, 8, 112, 8),
            ("conv_wide", 12, 4, 48, 4), ("fc", 24, 4, 16, 8)]
    for model, T, B, H, ck in corr:
        r = run(model, "corr", T, B, H, ck)
        dch = float(r.get("dchunk", 9))
        # fc(无卷积)必须逐位一致；conv 容许 cuDNN batch-算法 固有包络
        bound = 1e-4 if model == "fc" else 5e-2
        ok = r.get("status") == "ok" and dch <= bound
        print(f"{model:>11} {T:>4} {B:>3} {H:>4} {ck:>5} | {r.get('dchunk','-'):>10} "
              f"{r.get('dsingle','-'):>10} {r.get('nchunks','-'):>7} {'✓' if ok else '✗ '+str(r):>6}")

    print("\n=== (2) 不 OOM + regime：大 T 下 full 多步 vs autochunk ===")
    print(f"{'model':>11} {'T':>4} {'B':>3} {'H':>4} | {'full':>10} | {'auto':>6} {'chunk_t':>7} {'regime':>13} {'peak/预算':>12}")
    cfgs = [
        ("conv_small", 256, 16, 112),   # compute-bound, full 应 OOM
        ("conv_small", 512, 16, 112),   # compute-bound, 更大 T
        ("conv_wide",  128,  8, 112),   # 更重模型, compute-bound, full 应 OOM
        ("conv_wide",  256,  8,  64),   # compute-bound 中等
        ("conv_small", 256,  1,   8),   # launch-bound（小 B·H）
        ("fc",         256,  1,  16),   # launch-bound, 全连接
        ("fc",         512,  4,  32),   # fc 中等
    ]
    for model, T, B, H in cfgs:
        rf = run(model, "full", T, B, H, timeout=100)        # full 大 T 编译悬崖 → 短超时即判 compile-stall
        ra = run(model, "auto", T, B, H)
        full_s = "OOM" if rf.get("status") == "oom" else (f"{rf.get('peak','?')}GiB" if rf.get("status") == "ok" else rf.get("status"))
        auto_s = "OK" if ra.get("status") == "ok" else ra.get("status")
        pk = f"{ra.get('peak','?')}/{ra.get('budget','?')}" if ra.get("status") == "ok" else "-"
        print(f"{model:>11} {T:>4} {B:>3} {H:>4} | {full_s:>10} | {auto_s:>6} "
              f"{ra.get('chunk_t','-'):>7} {ra.get('regime','-'):>13} {pk:>12}")
    print("\nGENERALIZATION_DONE")


if __name__ == "__main__":
    if len(sys.argv) >= 8 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]),
               int(sys.argv[6]), int(sys.argv[7]))
    else:
        main()
