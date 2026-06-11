"""验证自动分块的前提：spikingjelly 多步(triton) 神经元是否在【不 reset】的情况下跨多次调用串接膜电位 v。
若是，则「reset 一次 → 按 chunk 喂 x[i:i+chunk] → 不 reset」的分块多步 == 整段多步（逐位/近似一致），
分块推理 spikingjelly 模型才成立（这是把 auto-chunk 接到 SpikingJelly-Triton 路线的基石）。

复用 large-T-oom-fallback/sj_triton_oom.py 的 monkey-patch + build_net + set_mode。
跑法：~/miniconda3/envs/sj_triton/bin/python experiments/autochunk/verify_premise.py
"""
import os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "large-T-oom-fallback"))
import sj_triton_oom as S          # 触发 monkey-patch；提供 build_net / set_mode
from spikingjelly.activation_based import functional

dev = "cuda"


@torch.no_grad()
def full_multistep(net, x):
    functional.reset_net(net)
    return net(x)


@torch.no_grad()
def chunked_multistep(net, x, chunk):
    """reset 一次，按 chunk 喂，块间不 reset（靠神经元自身的 self.v 串接）。"""
    T = x.shape[0]
    functional.reset_net(net)
    outs = []
    for i in range(0, T, chunk):
        outs.append(net(x[i:i + chunk].contiguous()))
    return torch.cat(outs, 0)


def main():
    print(f"GPU {torch.cuda.get_device_name(0)}  triton {S.triton.__version__}")
    torch.manual_seed(0)
    net = S.build_net().to(dev).eval()
    S.set_mode(net, multistep=True, backend="triton")
    T, B, H = 24, 4, 32
    x = (torch.rand(T, B, 3, H, H, device=dev) < 0.5).float()

    y_full = full_multistep(net, x)
    print(f"full multistep out {tuple(y_full.shape)}")
    for chunk in (1, 2, 3, 6, 8, 12):
        y_ck = chunked_multistep(net, x, chunk)
        d = (y_full - y_ck).abs().max().item()
        print(f"  chunk={chunk:>2}: max|Δ(full vs chunked-multistep)| = {d:.3e}  {'✓ 串接成立' if d < 1e-2 else '✗ 不串接(膜电位被 reset)'}")
    print("\nPREMISE_DONE")


if __name__ == "__main__":
    main()
