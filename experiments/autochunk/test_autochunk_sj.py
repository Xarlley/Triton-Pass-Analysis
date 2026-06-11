"""测试 snn_compiler.nn.AutoChunkInference 在 SpikingJelly-Triton 路线上的自动分块推理：
(1) 在 full 仍跑得通的 T 上，autochunk 输出与整段多步近似一致；
(2) 在 full 会 OOM 的大 T 上，autochunk 自动选块、跑通、不 OOM，且把显存用得较满。

环境 sj_triton；复用 large-T-oom-fallback/sj_triton_oom.py 的 patch+build_net+set_mode。
跑法：~/miniconda3/envs/sj_triton/bin/python experiments/autochunk/test_autochunk_sj.py
"""
import os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))                     # 仓库根 → import snn_compiler
sys.path.insert(0, os.path.join(HERE, "..", "large-T-oom-fallback"))
import sj_triton_oom as S
from spikingjelly.activation_based import functional
from snn_compiler.nn import AutoChunkInference

dev = "cuda"
GiB = 1024 ** 3


def main():
    print(f"GPU {torch.cuda.get_device_name(0)}  triton {S.triton.__version__}  "
          f"total={torch.cuda.get_device_properties(0).total_memory/GiB:.1f}GiB")
    torch.manual_seed(0)
    net = S.build_net().to(dev).eval()
    S.set_mode(net, multistep=True, backend="triton")
    auto = AutoChunkInference(net, reset_fn=functional.reset_net, memory_fraction=0.85, verbose=True)
    B, H = 16, 112

    # (1) 正确性：full 跑得通的 T=32
    T = 32
    x = (torch.rand(T, B, 3, H, H, device=dev) < 0.5).float()
    functional.reset_net(net)
    with torch.no_grad():
        y_full = net(x)
    y_auto = auto(x)
    d = (y_full - y_auto).abs().max().item()
    ag = (y_full.argmax(-1) == y_auto.argmax(-1)).float().mean().item()
    print(f"[正确性 T={T}] autochunk vs 整段多步：max|Δ|={d:.3e}  top1-agree={ag*100:.1f}%  "
          f"chunk_t={auto.last_plan['chunk_t']}\n")
    del x, y_full, y_auto; torch.cuda.empty_cache()

    # (2) 大 T：full 会 OOM（B=16,H=112 时 full 在 T=64 起 OOM），autochunk 应跑通
    for T in (128, 256, 512):
        torch.cuda.empty_cache()
        try:
            x = (torch.rand(T, B, 3, H, H, device=dev) < 0.5).float()
            y = auto(x)                                   # 第一次：选块(含探针)+真跑，建立缓存
            # 第二次：用缓存的 chunk_t 净跑一遍，干净测峰值（不含探针/失败重试）
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            y = auto(x)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / GiB
            p = auto.last_plan
            util = peak / p['budget_GiB'] * 100
            print(f"[大 T={T}] OK  chunk_t={p['chunk_t']}  regime={p['regime']}  "
                  f"净峰值={peak:.2f}GiB(预算≈{p['budget_GiB']:.1f}, 用满≈{util:.0f}%)  out={tuple(y.shape)}")
            del x, y
        except torch.cuda.OutOfMemoryError:
            print(f"[大 T={T}] autochunk 仍 OOM（不应发生）")
            torch.cuda.empty_cache()
    print("\nAUTOCHUNK_SJ_DONE")


if __name__ == "__main__":
    main()
