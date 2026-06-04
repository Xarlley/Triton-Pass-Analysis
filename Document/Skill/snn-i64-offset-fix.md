# 方法：i64 字节偏移修复 — 让 SNN kernel 在大 T 下正确

> 时间：2026-05-29
> 状态：已集成到 snn_compiler/kernels/{fused,neurons}.py（4 个 kernel 同步修复）
> 影响：correctness bug，T × 单层激活 > 2 GiB 时未修复版会写入非法地址

## 1. 现象

VGG-16 SNN 在 T=128, BATCH=4, layout=NHWC, bf16 下，跑到 conv1 后的 LIF 立刻报：

```
CUDA error: an illegal memory access was encountered
```

排查过程：
- 把 BATCH=4 降到 1 → 不 crash。
- 拿 NCL = B × C × H × W 与 T 一起遍历，发现：**当 (T-1) × NCL × sizeof(elem) > 2³¹ 时**必 crash。
- (T=128, B=4, C=64, H=W=224) → NCL = 12.8M，(T-1) × NCL × bf16 = 3.26 GB > 2.15 GB → 触发。
- (T=128, B=1, C=64, H=W=224) → 0.82 GB < 2.15 GB → 正常。

## 2. 根因：Triton 默认用 i32 算字节偏移

抓 TTIR（[Document/IR-Trace/large_T/T128_block512_w8.ttir](../IR-Trace/large_T/T128_block512_w8.ttir)）看：

```mlir
%c101957632_i32 = arith.constant 101957632 : i32
%c101154816_i32 = arith.constant 101154816 : i32
...
```

每条 `arith.constant` 是 `t × NCL × sizeof(elem)` 的预算结果，类型是 **i32**。
i32 范围 ±2³¹ ≈ ±2.15 GB（字节）。当 (T-1) × NCL × 元素字节数 > 2³¹ 时：

- 编译期常量算出来 wrap 成负数
- `tt.load %y_ptr + (i32) wrapped_off + ncl_idx` 取到非法地址
- CUDA driver 抛 `cudaErrorIllegalAddress`

注意：**不能靠测试用 fp32 触发**（fp32 的话 NCL 更小才能撞临界，反而漏掉常见场景）。
bf16 + 大 H×W 是最常见的"大 T inference"配置，恰好首先撞上。

## 3. 修复：在 Python 层强制 i64

原代码（每个有 t-loop 的 kernel 都同款）：

```python
pid = tl.program_id(0)
ncl_idx = pid * BLOCK_NCL + tl.arange(0, BLOCK_NCL)
...
for t in tl.static_range(0, T, 1):
    y_t = tl.load(y_ptr + t * NCL + ncl_idx, ...)
    ...
    tl.store(spike_ptr + t * NCL + ncl_idx, spike, ...)
```

改：

```python
pid = tl.program_id(0)
ncl_idx = pid.to(tl.int64) * BLOCK_NCL + tl.arange(0, BLOCK_NCL).to(tl.int64)
mask = ncl_idx < NCL
NCL_i64 = tl.full([], NCL, dtype=tl.int64)
...
for t in tl.static_range(0, T, 1):
    t_off = tl.full([], t, dtype=tl.int64) * NCL_i64
    y_t = tl.load(y_ptr + t_off + ncl_idx, ...)
    ...
    tl.store(spike_ptr + t_off + ncl_idx, spike, ...)
```

要点：
1. `ncl_idx` 整条链显式上 i64（`pid.to(int64) * BLOCK + arange.to(int64)`）
2. `NCL` 借 `tl.full([], NCL, dtype=tl.int64)` 抬到 i64
3. `t * NCL` 重新由 i64 算，绝不被编译器抠回 i32 常量

代价：
- 每个地址多两条指令（一次 i32→i64，一次 i64 mul）
- 实测 bandwidth 不变（仍 ~705 GiB/s）
- PTX 行数：T=128 大配置从 6722 → 6750（+0.4%）

## 4. 修复覆盖的 kernel

[snn_compiler/kernels/fused.py](../../snn_compiler/kernels/fused.py)：
- `_bias_if_lif_kernel`（核心 Conv→BN→LIF 融合，本次问题源头）
- `_bias_if_lif_rate_kernel`（rate-coded 输出版，新增）
- `_bias_if_lif_stateful_kernel`（chunked driver 用，新增）

[snn_compiler/kernels/neurons.py](../../snn_compiler/kernels/neurons.py)：
- `_fused_spiking_neuron_kernel`（裸 IF/LIF，无 conv）
- `_cuba_lif_kernel`（CubaLIF 双状态）
- `_eif_kernel`（指数 IF）

## 5. 验证

[snn_compiler/tests/test_largeT_and_rate.py](../../snn_compiler/tests/test_largeT_and_rate.py) ::
`test_i64_overflow_no_crash`:

```python
T, B, C, H, W = 128, 4, 64, 224, 224
# (T-1) * NCL * 2 = 3.26 GiB > 2 GiB（i32 阈值）
y = torch.randn(T, B, C, H, W, device='cuda', dtype=torch.bfloat16).contiguous()
bias = torch.randn(C, device='cuda')
out = fused_bias_if_lif(y, bias, neuron='lif', tau=2.0, ...)
assert out.shape == y.shape and not out.isnan().any()
```

修复前：CUDA Illegal Memory Access。修复后：通过。

既有 188 个 bit-equal 测试全部继续通过。

## 6. 经验：在 Triton 上做 SNN，**永远用 i64 算地址**

SNN 推理常用配置：
- bf16
- BATCH * C * H * W 容易 > 10M
- T 4 起，128 不算特别极端（神经科学/事件 camera 流常 T = 1024+）

只要 `T * NCL * sizeof(elem) > 2³¹`，就该走 i64。本框架的所有 LIF kernel 现在都
按 i64 算。下游若要写自定义 kernel，照本文 §3 的 idiom 即可。
