"""P6c：Triton 打包 kernel + popcount KᵀV，看能否净胜 cutlass bmm。

P6b 发现：popcount kernel 比 bmm 快 4-5× 且逐位一致，但 torch 打包慢，吃掉红利。
这里用 Triton pack kernel（带宽受限）替代 torch 打包，测 full = pack + popcount 是否净胜 bmm。
跑法（A100, triton-src）：cd ~/charlley/snn_compiler_attn && python snn_compiler/explore/attention/phase6c_packkernel.py
"""
import os, sys
import torch
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
import _bench_util as BU
from phase6b_bitpack import bitpack_N, _ktv_popcount_kernel, _next_pow2

dev = "cuda"


@triton.jit
def _pack_N_kernel(k_ptr, out_ptr, N, D, W,
                   BLOCK_I: tl.constexpr):
    """k:[TBH,N,D] {0,1} -> out:[TBH,D,W] int32（沿 token 维打包，每 32 token -> 1 word）。"""
    pid_tbh = tl.program_id(0).to(tl.int64)
    pid_i = tl.program_id(1)
    i_off = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    i_mask = i_off < D
    N_i64 = N.to(tl.int64)
    D_i64 = D.to(tl.int64)
    W_i64 = W.to(tl.int64)
    base = pid_tbh * N_i64 * D_i64
    t_off = tl.arange(0, 32)
    shifts = (tl.full([32], 1, tl.int64) << t_off.to(tl.int64))     # [32]
    for w in range(0, W):
        n = w * 32 + t_off                                          # [32]
        n_mask = n < N
        addr = base + n.to(tl.int64)[None, :] * D_i64 + i_off.to(tl.int64)[:, None]  # [BI,32]
        m = i_mask[:, None] & n_mask[None, :]
        bit = tl.load(k_ptr + addr, mask=m, other=0).to(tl.int64)  # [BI,32]
        word = tl.sum(bit * shifts[None, :], axis=1)               # [BI] int64
        tl.store(out_ptr + pid_tbh * D_i64 * W_i64 + i_off.to(tl.int64) * W_i64 + w,
                 word.to(tl.int32), mask=i_mask)


def pack_N_triton(x):
    """x:[TBH,N,D] {0,1} -> [TBH,D,W] int32。"""
    TBH, N, D = x.shape
    W = (N + 31) // 32
    out = torch.empty((TBH, D, W), device=x.device, dtype=torch.int32)
    BLOCK_I = 64
    grid = (TBH, triton.cdiv(D, BLOCK_I))
    _pack_N_kernel[grid](x.contiguous(), out, N, D, W, BLOCK_I=BLOCK_I)
    return out


def popcount_ktv(kp, vp, T, B, H, D):
    TBH = T * B * H
    out = torch.empty((TBH, D, D), device=kp.device, dtype=torch.float32)
    BD = _next_pow2(D)
    W = kp.shape[-1]
    _ktv_popcount_kernel[(TBH,)](kp, vp, out, W, D=D, BLOCK_D=BD)
    return out.reshape(T, B, H, D, D)


def spikes(shape, rate):
    return (torch.rand(*shape, device=dev) < rate).float()


def run(tag, T, B, heads, N, d, rk, rv):
    k = spikes((T, B, heads, N, d), rk)
    v = spikes((T, B, heads, N, d), rv)
    TBH = T * B * heads
    kf = k.reshape(TBH, N, d); vf = v.reshape(TBH, N, d)
    kv_bmm = k.transpose(-2, -1) @ v

    # pack 正确性：triton vs torch
    kp_t = pack_N_triton(kf); kp_ref = bitpack_N(kf)
    pack_ok = torch.equal(kp_t, kp_ref)
    vp_t = pack_N_triton(vf)
    kv_pc = popcount_ktv(kp_t, vp_t, T, B, heads, d)
    corr = (kv_bmm - kv_pc).abs().max().item()

    def full_triton():
        return popcount_ktv(pack_N_triton(kf), pack_N_triton(vf), T, B, heads, d)

    r_bmm  = BU.bench(lambda: k.transpose(-2, -1) @ v, warmup=25, iters=100)
    r_pack = BU.bench(lambda: pack_N_triton(kf), warmup=25, iters=100)
    r_full = BU.bench(full_triton, warmup=25, iters=100)

    print(f"\n[{tag}] T={T} B={B} heads={heads} N={N} d={d} rk/rv={rk}/{rv}")
    print(f"   pack triton==torch: {pack_ok}   popcount-KᵀV vs bmm max|Δ|={corr:.3e}")
    print(f"   bmm(cutlass)          : {BU.fmt(r_bmm)}")
    print(f"   pack kernel (1 of 2)  : {BU.fmt(r_pack)}")
    print(f"   FULL (2×pack+popcount): {BU.fmt(r_full)}")
    print(f"   => full / bmm = {r_bmm['median_ms']/r_full['median_ms']:.2f}x "
          f"({'WIN' if r_full['median_ms']<r_bmm['median_ms'] else 'lose'})")


def main():
    BU.gpu_guard(tag="P6c-start")
    run("SF-like",     T=4, B=16, heads=8, N=196, d=96, rk=0.026, rv=0.047)
    run("SDT-like",    T=4, B=16, heads=8, N=196, d=64, rk=0.034, rv=0.078)
    run("SF-like B64", T=4, B=64, heads=8, N=196, d=96, rk=0.026, rv=0.047)
    BU.gpu_guard(tag="P6c-end")
    print("\nP6c_DONE")


if __name__ == "__main__":
    main()
