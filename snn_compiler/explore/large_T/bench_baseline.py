"""测当前 `_bias_if_lif_kernel` 在 T 取 4/16/64/128 时的 effective bandwidth 与 GFLOPS。

RTX 5070 Ti 理论峰值：
- bf16 tensor: ~251 TFLOPS
- fp32 vec:    ~31 TFLOPS
- GDDR7 bandwidth：~672 GB/s（数据手册；此卡是 GDDR7，非 HBM）

判断当前 kernel 是 compute-bound 还是 bandwidth-bound：
  - Per-neuron 工作量：T × (load 4B + store 4B + ~7 fmadd) = 8T B 读写 + 7T flops
  - 在 RTX 5070 Ti 上，1 fma 周期 ≈ 1 B 传输周期，所以
    intensity = 7T / 8T = 0.875 flop/byte → 远低于 roofline turning point（~30 flop/byte）
  → SNN neuron kernel 在任何 T 下都该是 **bandwidth bound**。
  → 若实测 BW < 50% peak，说明 launch / IR 大小 / 同步开销在拖。
"""
import os, sys, time, statistics, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
import torch
import triton
from snn_compiler.kernels.fused import fused_bias_if_lif


def time_ms(fn, n_warm=10, n_iter=200):
    for _ in range(n_warm): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / n_iter


def bench(shape):
    T, B, C, H, W = shape
    NCL = B * C * H * W
    y = torch.randn(T, B, C, H, W, device='cuda').contiguous()
    bias = torch.randn(C, device='cuda')

    def call():
        return fused_bias_if_lif(y, bias, neuron='lif', tau=2.0,
                                  decay_input=True, soft_reset=False,
                                  v_threshold=1.0, v_reset=0.0, layout='NCHW')

    ms = time_ms(call)
    # 内存流量：每个 (t, idx) 读 4B + 写 4B = 8B。加 bias 1×C 读一次（常驻 cache）。
    bytes_rw = T * NCL * 8 + C * 4   # fp32
    bw_gibps = bytes_rw / (ms * 1e-3) / (1 << 30)
    # FLOPs：每步 7 fma + 1 cmp ≈ 8 flop/element
    flops = T * NCL * 8
    gflops = flops / (ms * 1e-3) / 1e9
    return ms, bw_gibps, gflops


def main():
    # 不同形状探查
    print(f"{'T':>4s} {'B':>3s} {'C':>4s} {'H':>3s} {'NCL':>10s} {'ms':>9s} "
          f"{'BW GiB/s':>10s} {'GFLOPS':>9s}")
    print("-" * 70)
    for T in [4, 16, 32, 64, 128]:
        for B, C, H, W in [(16, 64, 56, 56), (16, 128, 28, 28), (16, 512, 7, 7)]:
            ms, bw, gflops = bench((T, B, C, H, W))
            NCL = B * C * H * W
            print(f"{T:>4d} {B:>3d} {C:>4d} {H:>3d} {NCL:>10d} {ms:>9.4f} "
                  f"{bw:>10.1f} {gflops:>9.1f}")
        print()


if __name__ == "__main__":
    main()
