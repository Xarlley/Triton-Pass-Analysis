# 方法：T-Chunked Execution — 让大 T 在显存受限卡上跑得动

> 时间：2026-05-29
> 状态：已集成到 [snn_compiler/nn/chunked.py](../../snn_compiler/nn/chunked.py)
> 接口：`StatefulLIFNode`、`run_chunked(...)`、`ChunkedForward(model, chunk_t=...)`
> 适用：T 大（≥ 32）且 NCL × T × sizeof > GPU 单层激活预算时

## 1. 问题：T = 128 的内存爆炸

VGG-16 SNN 第一个 conv 输出形状是 `[T, B, 64, 224, 224]`：

| T | B | dtype | 单层激活 |
|---:|---:|---|---:|
| 4 | 16 | bf16 | 0.8 GiB |
| 4 | 64 | bf16 | 3.1 GiB |
| 64 | 16 | bf16 | **12.8 GiB** |
| 128 | 4 | bf16 | **6.4 GiB** |
| 128 | 16 | bf16 | **25.7 GiB**（超 16 GiB 卡） |

13 个 conv 层 + 13 个 LIF spike 都是同量级，T=128 + B=16 在 16 GiB 卡上**没办法**整网激活同时驻留。
即使 Inductor / 本框架的 fused kernel 复用 buffer，conv 输出仍要先写出再被 LIF 读。

## 2. 思路：把 T 切成 chunk，状态串接

```
全图 forward(x_seq: [T=128, B, ...])
  → for chunk_id in 0..T/chunk_t:
       chunk = x_seq[chunk_id*chunk_t : (chunk_id+1)*chunk_t]   # [chunk_t, B, ...]
       chunk_out = full_network_forward(chunk, lif_states)      # 各 LIF 拿上一 chunk 末态
       outputs.append(chunk_out)
  → cat(outputs)
```

每个 chunk 走完整网络（13 个 conv + 13 个 LIF），输出 spike 累积。瞬时显存
≈ `chunk_t / T` 倍原始峰值。状态：每 LIF 留 `[B, C, H, W]` fp32 v 张量
（≈ 单步激活，与 T 无关），13 层加起来 << 一个完整 T spike。

## 3. 数学等价：v 状态在 chunk 边界精确串接

LIF 数学：
```
v_t = decay * v_{t-1} + scale * (x_t + bias)
spike_t = (v_t ≥ v_th) ? 1 : 0
v_t = reset(v_t, spike_t)
```

只要把 chunk N+1 的 `v_0`（chunk N 的末态 `v_{chunk_t-1}` 经过 reset 后）作为 chunk N+1
的初始电位，整条 T 的 spike train 与单次 forward **bit-equal**。

验证：[snn_compiler/tests/test_largeT_and_rate.py](../../snn_compiler/tests/test_largeT_and_rate.py)::`test_stateful_lif_chunked` —
T=32 拆 chunk = 4 / 8 / 16 / 32，soft / hard 双 reset 模式共 8 种组合，**全部 max|diff|=0**。

## 4. 实现

### 4.1 Stateful LIF kernel

[snn_compiler/kernels/fused.py](../../snn_compiler/kernels/fused.py) 的
`_bias_if_lif_stateful_kernel`：

```python
@triton.jit
def _bias_if_lif_stateful_kernel(
    y_ptr, bias_ptr, spike_ptr, v_init_ptr, v_out_ptr,
    T: tl.constexpr, NCL: tl.constexpr, ...,
    HAS_VINIT: tl.constexpr, HAS_VOUT: tl.constexpr,
):
    ...
    if HAS_VINIT:
        v = tl.load(v_init_ptr + ncl_idx, mask=mask, other=0.0).to(tl.float32)
    else:
        v = tl.zeros([BLOCK_NCL], dtype=tl.float32)
    for t in tl.static_range(0, T, 1):
        ...
        tl.store(spike_ptr + t_off + ncl_idx, spike, mask=mask)
    if HAS_VOUT:
        tl.store(v_out_ptr + ncl_idx, v, mask=mask)
```

HAS_VINIT / HAS_VOUT 都是 constexpr，编译期 specialize。不传 v_init 时 v=0；
不要 v_out 时不写出（兼容现有"非 chunked"调用，零开销）。

### 4.2 nn.Module

```python
class StatefulLIFNode(nn.Module):
    def forward(self, x_seq, v_init=None, return_v=False):
        return fused_bias_if_lif_stateful(x_seq, None,
            v_init=v_init, return_v=return_v, ...)
```

### 4.3 Chunked driver

```python
from snn_compiler.nn import run_chunked

def forward_step(x_chunk, state):
    state = state or {}
    # Layer 1: stateful_lif1.forward(x_chunk, v_init=state.get('lif1'), return_v=True)
    spike1, v1 = stateful_lif1(x_chunk, v_init=state.get('lif1'), return_v=True)
    state['lif1'] = v1
    # ... more layers ...
    return out_chunk, state

y_seq = run_chunked(forward_step, x_seq, chunk_t=16)
```

或者用 [ChunkedForward](../../snn_compiler/nn/chunked.py) wrapper：

```python
from snn_compiler.nn import ChunkedForward

class MySNN(nn.Module):
    def forward_chunked(self, x_chunk, state):
        ...   # 同上
    def forward(self, x_seq):
        return self.forward_chunked(x_seq, None)[0]   # 非 chunked 路径

wrapped = ChunkedForward(MySNN(), chunk_t=16)
y_seq = wrapped(x_seq)
```

## 5. 性能 trade-off

T-chunked 是 **memory 与 latency 之间的权衡**：

- **memory**：chunk_t / T 倍降。T=128 chunk=16 → 8× 显存降；T=128 chunk=32 → 4× 降。
- **latency**：每 chunk 一次 kernel launch，T=128 chunk=16 → 8 次 launch，开销
  约 +50–100 µs × (chunks-1) ≈ 0.5 ms。对 VGG-16 几 ms 量级影响是 10–20% 慢。

**结论**：T-chunked 不应作为默认路径；它是**显存不够时**的 fallback。框架默认走 full-T
（最快），当 T 极大且显存不够时显式包 `ChunkedForward(model, chunk_t=16)`。

## 6. 与其它优化的关系

T-chunked 与 rate-coded / 残差融合等优化**完全正交**。链式叠加：

1. zoo `fused=True` → conv-BN-LIF 融合
2. 末层换 RateCodedLIFNode → 写带宽减
3. `ChunkedForward(model, chunk_t=16)` → 显存达标

T=128 大 batch 在 16 GiB 卡上的标准配方。
