"""端到端 VGG-16 / ResNet-18 SNN 在 T = 4 / 16 / 64 / 128 下的延迟与显存。

目的：定位大 T 时是否 LIF-kernel-bound 还是 conv-bound 还是 memory-bound。
"""
import os, sys, time, statistics, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
import torch
import torch.nn as nn


def time_ms(fn, n_warm=3, n_iter=10):
    for _ in range(n_warm): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / n_iter


def run(arch_name, factory, B=4):
    print(f"\n=== {arch_name}  BATCH={B} ===")
    print(f"{'T':>4s}  {'lat_ms':>8s}  {'per_img_ms':>11s}  {'peak_GiB':>9s}  {'OK':>3s}")
    for T in [4, 16, 64, 128]:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(0)
            m = factory(num_classes=1000, neuron='lif', tau=2.0,
                         soft_reset=False, layout='NHWC',
                         fused=True, init_bn=True).cuda().eval().to(torch.bfloat16)
            for mod in m.modules():
                if isinstance(mod, nn.Conv2d):
                    mod.weight.data = mod.weight.data.to(memory_format=torch.channels_last)
            x = torch.randn(T, B, 3, 224, 224, device='cuda', dtype=torch.bfloat16)
            def call():
                with torch.no_grad(): m(x)
            ms = time_ms(call)
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"{T:>4d}  {ms:>8.2f}  {ms/B:>11.4f}  {peak:>9.2f}  {'OK':>3s}")
            del m, x
        except torch.AcceleratorError as e:
            print(f"{T:>4d}  {'OOM':>8s}  {'-':>11s}  {'-':>9s}  ERR  {str(e)[:60]}")
        except RuntimeError as e:
            print(f"{T:>4d}  {'OOM':>8s}  {'-':>11s}  {'-':>9s}  ERR  {str(e)[:60]}")


if __name__ == "__main__":
    from snn_compiler.zoo import vgg16_snn, resnet18_snn
    run('VGG-16 SNN', vgg16_snn, B=4)
    run('ResNet-18 SNN', resnet18_snn, B=4)
