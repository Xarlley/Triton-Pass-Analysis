"""验证：
1. i64 byte-offset 修复在大 T 不再 crash
2. RateCodedLIFNode bit-equal sum-over-T 等于 naive LIFNode 输出 .sum(0)
3. StatefulLIFNode 串接多 chunk 与单一 forward bit-equal
4. RateCodedIFNode 在 IF 行为下同样 bit-equal
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch

from snn_compiler.kernels.fused import (
    fused_bias_if_lif, fused_bias_if_lif_rate, fused_bias_if_lif_stateful,
)
from snn_compiler.nn import (
    LIFNode, IFNode,
    RateCodedLIFNode, RateCodedIFNode, StatefulLIFNode,
)


def _naive_lif(x_seq, *, tau, decay_input, soft_reset, v_threshold, v_reset):
    """无 bias 的 LIF 朴素实现，shape 同入 x_seq。"""
    T = x_seq.shape[0]
    decay = 1.0 - 1.0 / tau
    scale = (1.0 / tau) if decay_input else 1.0
    v = torch.zeros_like(x_seq[0], dtype=torch.float32)
    spikes = []
    for t in range(T):
        v = decay * v + scale * x_seq[t].float()
        spk = (v >= v_threshold).float()
        spikes.append(spk)
        if soft_reset:
            v = v - spk * v_threshold
        else:
            v = torch.where(spk > 0, torch.full_like(v, v_reset), v)
    return torch.stack(spikes, dim=0)


def _naive_if(x_seq, *, decay, soft_reset, v_threshold, v_reset):
    T = x_seq.shape[0]
    v = torch.zeros_like(x_seq[0], dtype=torch.float32)
    spikes = []
    for t in range(T):
        v = decay * v + x_seq[t].float()
        spk = (v >= v_threshold).float()
        spikes.append(spk)
        if soft_reset:
            v = v - spk * v_threshold
        else:
            v = torch.where(spk > 0, torch.full_like(v, v_reset), v)
    return torch.stack(spikes, dim=0)


def test_i64_overflow_no_crash():
    """T*NCL*sizeof(elem) > 2^31 时 kernel 不再 illegal-memory-access。"""
    # T=128, B=4, C=64, H=W=224, NCL=12.8M, (T-1)*NCL*2B = 3.26 GB > 2^31
    T, B, C, H, W = 128, 4, 64, 224, 224
    cond = (T - 1) * B * C * H * W * 2  # bytes (bf16)
    assert cond > (2 << 30), f"test shape too small: {cond:,} bytes"
    y = torch.randn(T, B, C, H, W, device='cuda', dtype=torch.bfloat16).contiguous()
    bias = torch.randn(C, device='cuda')
    out = fused_bias_if_lif(y, bias, neuron='lif', tau=2.0,
                              decay_input=True, soft_reset=False,
                              v_threshold=1.0, v_reset=0.0, layout='NCHW')
    torch.cuda.synchronize()
    assert out.shape == y.shape
    assert not out.isnan().any().item()
    print(f"  [OK] i64 overflow handling: shape={tuple(out.shape)} no NaN")


def test_rate_coded_lif_bit_equal():
    print("== RateCodedLIFNode bit-equal vs naive sum-over-T ==")
    torch.manual_seed(0)
    for T in [4, 16, 64]:
        for soft in [True, False]:
            x = torch.randn(T, 2, 8, 16, 16, device='cuda').contiguous()
            ref = _naive_lif(x, tau=2.0, decay_input=True, soft_reset=soft,
                              v_threshold=1.0, v_reset=0.0).sum(0)
            mod = RateCodedLIFNode(tau=2.0, decay_input=True, soft_reset=soft,
                                     v_threshold=1.0, v_reset=0.0, layout='NCHW')
            out = mod(x)
            eq = torch.equal(ref, out)
            max_diff = (ref - out).abs().max().item()
            print(f"  T={T} {'soft' if soft else 'hard'}: bit-eq={eq} max|diff|={max_diff:.3e}")
            assert max_diff == 0.0, "rate-coded LIF must match sum-over-T naive"


def test_rate_coded_if_bit_equal():
    print("== RateCodedIFNode bit-equal vs naive sum-over-T ==")
    torch.manual_seed(0)
    for T in [4, 16, 64]:
        for decay in [1.0, 0.95]:
            for soft in [True, False]:
                x = torch.randn(T, 2, 8, 16, 16, device='cuda').contiguous()
                ref = _naive_if(x, decay=decay, soft_reset=soft,
                                  v_threshold=1.0, v_reset=0.0).sum(0)
                mod = RateCodedIFNode(decay=decay, soft_reset=soft,
                                        v_threshold=1.0, v_reset=0.0, layout='NCHW')
                out = mod(x)
                max_diff = (ref - out).abs().max().item()
                print(f"  T={T} decay={decay} {'soft' if soft else 'hard'}: "
                      f"max|diff|={max_diff:.3e}")
                assert max_diff == 0.0


def test_stateful_lif_chunked():
    print("== StatefulLIFNode chunked == single-call ==")
    torch.manual_seed(0)
    T = 32
    x = torch.randn(T, 2, 16, 14, 14, device='cuda').contiguous()
    for soft in [True, False]:
        # full forward
        ref = _naive_lif(x, tau=2.0, decay_input=True, soft_reset=soft,
                          v_threshold=1.0, v_reset=0.0)
        # chunked forward with StatefulLIFNode
        for chunk in [4, 8, 16, 32]:
            mod = StatefulLIFNode(tau=2.0, decay_input=True, soft_reset=soft,
                                    v_threshold=1.0, v_reset=0.0, layout='NCHW')
            v = None
            chunks = []
            for i in range(0, T, chunk):
                c = min(chunk, T - i)
                x_c = x[i:i + c].contiguous()
                if i + c < T:
                    spike_c, v = mod(x_c, v_init=v, return_v=True)
                else:
                    spike_c, v = mod(x_c, v_init=v, return_v=True)
                chunks.append(spike_c)
            out = torch.cat(chunks, dim=0)
            max_diff = (ref - out).abs().max().item()
            print(f"  T=32 chunk={chunk} {'soft' if soft else 'hard'}: max|diff|={max_diff:.3e}")
            assert max_diff == 0.0


def main():
    test_i64_overflow_no_crash()
    print()
    test_rate_coded_lif_bit_equal()
    print()
    test_rate_coded_if_bit_equal()
    print()
    test_stateful_lif_chunked()
    print("\n" + "=" * 60)
    print("  ALL LARGE-T + RATE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
