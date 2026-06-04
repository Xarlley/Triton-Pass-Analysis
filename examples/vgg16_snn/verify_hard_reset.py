"""Verify PrefixSumHardResetIFNode is mathematically equivalent to naive sequential hard-reset.

If hard-reset really "invalidated subsequent prefix-sum" as suggested, the two
implementations would diverge on the same input. We run them on random tensors
of various shapes / value ranges and compare the resulting spike trains bit-by-bit.
"""
import torch


def naive_hard_reset(x, threshold=1.0, v_reset=0.0):
    """Reference implementation: pure sequential, no prefix-sum.

    For each spatial position independently:
      v = 0
      for t in range(T):
          v = v + x[t]
          if v >= threshold:
              spike[t] = 1
              v = v_reset    # hard reset
          else:
              spike[t] = 0
    """
    T = x.shape[0]
    v = torch.zeros_like(x[0])
    spikes = []
    for t in range(T):
        v = v + x[t]
        spike = (v >= threshold).to(x.dtype)
        spikes.append(spike)
        # hard reset: where spike, set v to v_reset
        v = torch.where(spike > 0, torch.full_like(v, v_reset), v)
    return torch.stack(spikes, dim=0)


def prefix_sum_hard_reset(x, threshold=1.0):
    """Implementation under test: prefix-sum + per-element conditional baseline update.

    Assumes v_reset = 0.
    """
    cum = torch.cumsum(x, dim=0)
    last_cum_at_spike = torch.zeros_like(x[0])
    spikes = []
    for t in range(x.shape[0]):
        v_t = cum[t] - last_cum_at_spike
        spike = (v_t >= threshold).to(x.dtype)
        spikes.append(spike)
        last_cum_at_spike = torch.where(spike > 0, cum[t], last_cum_at_spike)
    return torch.stack(spikes, dim=0)


def test_case(name, x, threshold=1.0):
    s_naive = naive_hard_reset(x, threshold, v_reset=0.0)
    s_prefix = prefix_sum_hard_reset(x, threshold)
    bit_equal = torch.equal(s_naive, s_prefix)
    diff_count = (s_naive != s_prefix).sum().item()
    total = s_naive.numel()
    print(f"  {name:42s}  bit-equal={str(bit_equal):5s}  "
          f"diff={diff_count}/{total}  "
          f"spike rate naive={s_naive.mean().item():.4f}  "
          f"prefix={s_prefix.mean().item():.4f}")
    return bit_equal


torch.manual_seed(0)
print("=== 数值验证：prefix-sum hard-reset vs naive sequential hard-reset ===\n")

cases = [
    # (description, x, threshold)
    ("uniform positive [0,1)",
        torch.rand(4, 16, device="cuda"), 1.0),
    ("uniform [-1, 1)",
        2 * torch.rand(4, 16, device="cuda") - 1, 1.0),
    ("typical conv output range, x ~ N(0, 1)",
        torch.randn(4, 1024, device="cuda"), 1.0),
    ("large positive (causes consecutive spikes), x ~ U[1, 5)",
        1 + 4 * torch.rand(4, 256, device="cuda"), 1.0),
    ("large negative drops to 0 by reset (post-spike negative)",
        torch.tensor([[3.0], [0.0], [-5.0], [2.0]], device="cuda"), 1.0),
    ("hand-crafted: x=[0.5, 0.8, 0.7, 0.4], doc example",
        torch.tensor([[0.5], [0.8], [0.7], [0.4]], device="cuda"), 1.0),
    ("hand-crafted: x=[3, 3, 3, 3] all spike",
        torch.tensor([[3.0], [3.0], [3.0], [3.0]], device="cuda"), 1.0),
    ("hand-crafted: x=[3, 0, 3, 0] alternating",
        torch.tensor([[3.0], [0.0], [3.0], [0.0]], device="cuda"), 1.0),
    ("hand-crafted: x=[0.5, 0.8, -2.0, 1.5] negative after spike",
        torch.tensor([[0.5], [0.8], [-2.0], [1.5]], device="cuda"), 1.0),
    ("vgg16-style: T=4, B=4, C·H·W=64*224*224 (first conv output, small batch)",
        torch.randn(4, 4, 64, 224, 224, device="cuda"), 1.0),
    ("vgg16-style with multiple consecutive spikes (B=4)",
        2 + 0.5 * torch.randn(4, 4, 64, 224, 224, device="cuda"), 1.0),
    ("[fp64] same dense-spike case, double precision",
        (2 + 0.5 * torch.randn(4, 4, 64, 224, 224, device="cuda")).to(torch.float64), 1.0),
    ("threshold=0.5",
        torch.randn(4, 256, device="cuda"), 0.5),
    ("threshold=2.0 (sparse spiking)",
        torch.randn(4, 256, device="cuda"), 2.0),
    ("T=8 longer time window",
        torch.randn(8, 64, device="cuda"), 1.0),
    ("T=16 even longer",
        torch.randn(16, 64, device="cuda"), 1.0),
]

n_pass = 0
for name, x, thr in cases:
    if test_case(name, x, thr):
        n_pass += 1

print(f"\n=== {n_pass}/{len(cases)} cases passed bit-equal ===")
