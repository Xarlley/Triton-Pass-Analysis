# Prefix-Sum IF 神经元的 VGG16-SNN 实验：算法、实现与实测

> 本实验回答两个问题：
>
> 1. 把 LIF 神经元的「时间维顺序递推 + hard reset」替换为「prefix-sum 累积输入 + 顺序阈值比对 + soft-reset」之后，VGG16-SNN 在 RTX 5070 Ti 上的推理延迟会怎么变？
> 2. 这种替换在 eager 和 torch.compile + 全 Triton 两条路径下分别表现如何？
>
> 实验程序：[`examples/vgg16_snn/prefix_sum_snn.py`](../../examples/vgg16_snn/prefix_sum_snn.py)
> 实测产物：`/tmp/cold_start_results.jsonl` —— 与 path B / NIR-compile / SJ-direct 在同一脚本协议下记录的对照数据，文末复制了关键行。
>
> 验证环境：torch 2.11.0+cu130 (cuDNN 9, cuBLAS 13)、triton 3.7.0+gitef02d646、RTX 5070 Ti (sm_120)。

---

## 1. 算法设计

### 1.1 目标神经元语义

把每一层的 SJ `LIFNode`（带 decay、hard-reset）替换为 IF 神经元，**用 prefix-sum 重写时间维递推**：

```
forward(x: [T, B, C, H, W]):
  cum[t]      = sum_{k<=t} x[k]                       # ←─ prefix-sum，T 维方向
  v[t]        = cum[t] - threshold * spike_count[t-1] # ←─ soft-reset：每发一次 spike 减一份阈值
  spike[t]    = 1 if v[t] >= threshold else 0
  spike_count[t] = spike_count[t-1] + spike[t]
  return spike[0..T-1]
```

**算法等价性**：上式与标准 IF 的「每步 `v_t = v_{t-1} + x_t`、超阈即 spike + soft-reset `v_t -= threshold`」**逐元素恒等**。证明很短：

- 设 `c_t = sum_{k<=t} x_k`、`s_t = spike[t]`、`n_t = spike_count[t] = sum_{k<=t} s_k`。
- 标准 IF 满足：`v_t = v_{t-1} + x_t - threshold * s_{t-1}` ⇒ `v_t = c_t - threshold * sum_{k<t} s_k = c_t - threshold * n_{t-1}`。
- 阈值判定 `s_t = (v_t >= threshold)` ⇔ `s_t = (c_t >= threshold * (n_{t-1} + 1))`。

也就是说 prefix-sum 形式只是把同一组算式重排了：**把"线性累积"与"非线性阈值检查"在时间维上解耦** —— 线性部分（cumsum）形式上是可结合的，理论上可用 log-T 深度的 parallel scan；非线性部分（阈值 + spike_count 累加）严格因果，仍需 T 步顺序。

### 1.2 与现有 LIF 实现的差异

| | SpikingJelly LIFNode | 本实验 PrefixSumIFNode |
|---|---|---|
| 时间维计算结构 | 顺序递推 `v_t = (1-1/τ)·v_{t-1} + x_t` | prefix-sum + 顺序阈值检查 |
| Reset 语义 | hard reset (`v ← v_reset`, 通常 0) | soft reset (`v -= threshold`，excess 累加进下一步) |
| 是否含 decay | 是（`τ` 参数）| 否（IF，对应 τ→∞）|
| GPU 实现 | 一份 SJ 手写 `@triton.jit` fused-T-loop kernel（含 autotune） | 由 PyTorch 标准算子 `torch.cumsum` + `torch.where` 等组合而成；Inductor 可自由 fuse |

**注**：soft-reset 与 hard-reset 给出**不同的发放模式**（例如 `x = [3, 0, 0, 0]` 在 threshold=1 下，soft-reset 一发 3 次（excess 一直 ≥ 1），hard-reset 只发 1 次后 reset 到 0 就不再发）。本实验关注前向延迟而非数值结果，权重和输入都是随机的，所以**不做数值正确性比对** —— 各路径都跑同一份随机输入，对比的是同一算法骨架（13 Conv + 5 AvgPool + 3 FC + 15 神经元）的延迟。

### 1.3 网络结构（与 NIR 路径等价）

```
nn.Sequential(
    # features: 13 Conv + 5 AvgPool + 13 IF
    TimeBatchWrapper(nn.Conv2d(3,  64, 3, padding=1)),  PrefixSumIFNode(),
    TimeBatchWrapper(nn.Conv2d(64, 64, 3, padding=1)),  PrefixSumIFNode(),
    TimeBatchWrapper(nn.AvgPool2d(2, 2)),
    ... (5 个 block，每个 2-3 个 Conv-IF 后接一个 AvgPool) ...,
    TimeBatchWrapper(nn.AvgPool2d(2, 2)),
    # classifier: Flatten + 3 FC + 2 IF
    TimeBatchWrapper(nn.Flatten()),
    TimeBatchWrapper(nn.Linear(512*7*7, 4096)), PrefixSumIFNode(),
    TimeBatchWrapper(nn.Linear(4096, 4096)),    PrefixSumIFNode(),
    TimeBatchWrapper(nn.Linear(4096, 1000)),
)
```

`TimeBatchWrapper` 把 `[T, B, ...]` flatten 成 `[T·B, ...]` 喂给 `nn.Conv2d` / `nn.AvgPool2d` / `nn.Linear` / `nn.Flatten`，再 view 回 `[T, B, ...]` —— 与 SJ `layer.Conv2d.forward` step_mode='m' 分支等价（[`stateless_wrapper.py:176-190`](../../spikingjelly/spikingjelly/activation_based/layer/stateless_wrapper.py#L176-L190)），但不依赖 SJ 类继承，让 dynamo / Inductor trace 起来更直接。

---

## 2. 实现细节

### 2.1 PrefixSumIFNode 完整源码（17 行）

```python
class PrefixSumIFNode(nn.Module):
    def __init__(self, v_threshold: float = 1.0):
        super().__init__()
        self.v_threshold = v_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, ...]
        cum = torch.cumsum(x, dim=0)
        spike_count = torch.zeros_like(x[0])
        spikes = []
        for t in range(x.shape[0]):
            v_t = cum[t] - self.v_threshold * spike_count
            spike_t = (v_t >= self.v_threshold).to(x.dtype)
            spikes.append(spike_t)
            spike_count = spike_count + spike_t
        return torch.stack(spikes, dim=0)
```

- `torch.cumsum(x, dim=0)` —— PyTorch 内置 cumsum，CUDA 上走 `aten::cumsum` → `at::native::scan` 实现，多数情况是 log-T 深度的并行 scan，对 T=4 实际是 ~3 个 elementwise op；
- `for t in range(T)` —— Python 端循环；torch.compile 模式下 dynamo 看到 `T = x.shape[0]` 后会**完全展开**这 4 次迭代为 4 段内联代码；eager 模式下也是直接执行 4 次。

### 2.2 全 Triton 编译配置（compile 模式）

与 path B / NIR-compile / SJ-direct 完全一致的 Inductor 配置（保证横向对比公平）：

```python
torch._dynamo.config.recompile_limit = 256
torch._dynamo.config.cache_size_limit = 256
inductor_cfg.max_autotune = True
inductor_cfg.max_autotune_gemm_backends = "TRITON"
inductor_cfg.max_autotune_conv_backends = "TRITON"
inductor_cfg.force_disable_caches = True
```

- 不需要 `patch_spikingjelly_for_full_graph()` —— 我们没用 SJ `layer.BatchNorm2d`（也没有 BN），没有 `seq_to_ann_forward` 的 isinstance graph_break；
- 不需要 SJ `triton_kernel.multistep_lif` —— PrefixSumIFNode 全由原生 PyTorch 算子组成，Inductor 直接 codegen 全部逻辑。

### 2.3 实测协议

```
BATCH=56  T=4
WARMUP=5 次 forward
MEASURE_ITERS = ceil(10000 / BATCH) = 179 次 forward
总样本 = 179 × 56 = 10024 张

每次 forward 单独计时：torch.cuda.synchronize() + perf_counter()
冷启动：先 rm -rf ~/.triton/cache 保证 compile 路径真的从零编译
```

---

## 3. 实测结果

冷启动 10024 样本，BATCH=56，RTX 5070 Ti：

| 实现 | 单张延迟 (ms/张) | 吞吐 (张/s) | 峰值显存 (GiB) | 冷启动耗时 (s) |
|---|---:|---:|---:|---:|
| **PrefixSumIF-eager** | **7.963** | **125.6** | 12.71 | 0.6 |
| **PrefixSumIF-compile** | **8.241** | **121.3** | **6.00** | 89.7 |
| path B (VGG16SNN, compile) | 9.305 | 107.4 | 14.04 | 95.6 |
| NIR-compile (fold-BN + AvgPool, compile) | 9.297 | 107.5 | 14.04 | 96.1 |
| SJ-direct (LIFNode + AvgPool, compile) | 9.394 | 106.4 | 14.04 | 95.1 |

### 3.1 与其他路径的相对差距

- **PrefixSumIF-eager 比 NIR-compile 快 14.4%**（7.96 vs 9.30）；
- **PrefixSumIF-compile 比 NIR-compile 快 11.4%**（8.24 vs 9.30）；
- PrefixSumIF-eager 比 PrefixSumIF-compile 快 3.4% —— 与 [`eager-vs-triton-perf-gap.md`](eager-vs-triton-perf-gap.md) 的结论一致（eager + cuDNN conv > Inductor Triton conv），只是在 PrefixSumIF 这个简化神经元下差距比 path B（27%）小得多。

### 3.2 显存维度更显著的差异

| 实现 | 峰值显存 |
|---|---:|
| PrefixSumIF-compile | **6.00 GiB** |
| PrefixSumIF-eager | 12.71 GiB |
| path B / NIR / SJ-direct (compile) | 14.04 GiB |

**PrefixSumIF-compile 比其它 compile 路径峰值显存少 8 GiB（少 57%）**，原因见 §4.2。这意味着：
- 同一张 16 GiB GPU 可以把 BATCH 推到 ~120（实际未测，理论估算），其他路径只能到 56；
- 训练阶段反向通路对此更敏感（需要保留 forward 激活），节省显存可以放宽 batch / 模型规模约束。

---

## 4. 为什么 PrefixSumIF 比 LIFNode 快、且占显存少？

### 4.1 算子图层面

| 三类组成 | LIFNode 实现（path B / NIR / SJ-direct）| PrefixSumIFNode 实现（本实验）|
|---|---|---|
| 时间维递推 | **SJ 手写 `@triton.jit fused-T-loop` kernel**（[`_multistep_lif_forward_kernel`](../../spikingjelly/spikingjelly/activation_based/triton_kernel/neuron_kernel/lif.py#L34)）→ 通过 `torch.library.custom_op` 注册，dynamo 视作黑盒 launcher | `torch.cumsum` + 4 步 `cum[t] - threshold * spike_count` 的 elementwise op |
| 调用接口 | `triton_kernel.multistep_lif(x_seq, v, decay_input, tau, ...)`，**带 `@triton.autotune` 4 组 cfg + `restore_value=['grad_x_seq_ptr', 'grad_v_init_ptr']`** | 纯标准 PyTorch op chain |
| Inductor 能见度 | LIF 是 custom_op 黑盒：Inductor 只在 FX 图里看到一个 `higher_order.triton_kernel_wrapper_functional` 调用节点，**不能跨 LIF 上下文做融合** | 完全是普通 ATen op：Inductor 可以**自由把 cumsum 后的 4 步比对融进同一 Triton kernel**，甚至与下游 view/conv-epilogue 跨界融合 |

### 4.2 实测峰值显存的差距来源

SJ `multistep_lif_inference` kernel 是 `@triton.autotune(restore_value=['s_seq_ptr', 'v_seq_ptr'])` 设计的。`restore_value` 在 autotune benchmark phase 让 Triton **clone** 这两个输出 buffer，以确保每个 trial cfg 都能从相同状态开始 ——
对第 1 个 LIF 层这是 `[T, B, C, H, W] = [4, 56, 64, 224, 224] × 4 字节 = 2.68 GiB` 的额外 clone。前面 [`nir-call-stack-trace.md` §7.5](nir-call-stack-trace.md) 测过：NIR-eager 在 BATCH=48 就因为这条 clone 撑爆 16 GiB；compile 路径靠 Inductor buffer reuse 才能撑到 BATCH=56。

PrefixSumIFNode **没有 autotune restore_value 的 clone**，完全靠 Inductor 自己的 buffer 规划 ——
13 conv + 15 IF + 5 pool + 3 fc 的全部中间 buffer 经过 Inductor 的 reuse 后峰值只 6 GiB。

这是**算子设计选择对内存压力的直接影响** —— 把 LIF 从 custom_op 改成普通 ATen op chain，让编译器拿到了全程 buffer planning 的能力。

### 4.3 实测延迟的差距来源

参考 [`eager-vs-triton-perf-gap.md`](eager-vs-triton-perf-gap.md) 对 path B 的 kernel 类别分解：path B 的 LIF kernel 类共贡献 ~30 ms / forward (BATCH=32) 的 GPU 工作。PrefixSumIF 把这一段重写为：

- **1 次 cumsum**（per layer per forward）→ ~T 个 elementwise op 等价
- **4 次 `cum[t] - threshold * spike_count`、阈值比较、累加**（per layer per forward）→ 4 个简单 elementwise op
- **共计 ~8 个 elementwise op / per layer**，远少于 SJ kernel 内部 4 步内联代码的工作量

而且 Inductor 看到这些是普通 ATen op 后，**有充分自由把它们和上游 conv 的 epilogue 或下游 view 融在一起**，进一步减少 launch 数。

实测 compile 模式下整网生成的 Triton kernel 数：
- path B / NIR-compile / SJ-direct：约 42-54 个 kernel（13 conv tem + 13 BN/conv epilogue + 5 pool + ...），**外加 15 次 SJ LIF custom_op 调用**
- PrefixSumIF-compile：仅 ~35 个 kernel，**无任何 custom_op 调用**

---

## 5. 局限与适用性

### 5.1 算法局限

| 局限 | 影响 |
|---|---|
| Soft-reset 与 hard-reset **数值不等价** | 在用预训练 LIF 权重时，输出不会与原模型逐位一致；需要重新训练 SNN（用 soft-reset IF 训）或仅用作架构对照 |
| **不支持 decay**（τ→∞，纯 IF） | 失去 LIF 的"漏电"动力学；对部分需要时间常数表达短期记忆的任务可能下降精度（视任务而定） |
| 严格遵循"每步至多 1 spike"规则 | 与标准实现一致；若希望支持单步多 spike（即 `floor(v / threshold)`），可以一次发完，但需替换比较逻辑 |
| **本实验只测推理延迟** | 训练路径需要 backward；cumsum 与 stack 在反向通路上的 Inductor 行为未测 |

### 5.2 适用范围

prefix-sum 形式在以下场景特别有优势：

- **T 较大**（≥ 8）：cumsum 的 log-T 深度并行优势随 T 增大而显著；
- **训练 / 微调阶段需要省显存**：减少 8 GiB 峰值意味着 batch / 模型放大有更多空间；
- **需要让自定义 Pass 看到神经元逻辑**：custom_op 是黑盒 launcher，Pass 看不进去；普通 ATen op 链能让 SNN Pass 在 Triton 编译时看到完整数据流。

不适用：
- 需要**严格 LIF（decay + hard-reset）数值复现已训好模型**的部署 —— 此时只能继续走 SJ LIFNode。

---

## 6. 横向数据汇总（10024 样本冷启动）

来自 `/tmp/cold_start_results.jsonl` 的真实记录（已同步到 [`Document/IR-Trace/perf_breakdown/cold_start_results.jsonl`](../IR-Trace/perf_breakdown/cold_start_results.jsonl) 的下一次提交时一并固化）：

```
mode                 batch  ms/张   张/s    peak_mem  compile_s
─────────────────────────────────────────────────────────────────
PrefixSumIF-eager     56    7.963   125.6   12.71 GiB   0.6 s
PrefixSumIF-compile   56    8.241   121.3    6.00 GiB  89.7 s   ★ 最省显存
path B (compile)      56    9.305   107.4   14.04 GiB  95.6 s
NIR-compile           56    9.297   107.5   14.04 GiB  96.1 s
SJ-direct (compile)   56    9.394   106.4   14.04 GiB  95.1 s
```

### 关键收益（相对 baseline 中最佳的 NIR-compile）

- **延迟 -11.4%**（compile）/ **-14.4%**（eager）
- **峰值显存 -57%**（compile 模式）
- **编译时间持平**（PrefixSumIF compile 89.7 s vs NIR-compile 96.1 s）

---

## 7. 复现命令

```bash
cd /home/charlley/Code/Triton-Pass-Analysis

# 全 Triton 编译路径（PrefixSumIF-compile）
rm -rf ~/.triton/cache && mkdir -p ~/.triton/cache
MODE=compile BATCH=56 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    conda run -n triton-dev-cuda131 python examples/vgg16_snn/prefix_sum_snn.py

# eager 路径（PrefixSumIF-eager）
rm -rf ~/.triton/cache && mkdir -p ~/.triton/cache
MODE=eager BATCH=56 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    conda run -n triton-dev-cuda131 python examples/vgg16_snn/prefix_sum_snn.py

# 结果会追加到 /tmp/cold_start_results.jsonl，可用 jq 看：
jq -c '{mode, batch, mean_per_img_ms, peak_mem_gib, compile_s}' /tmp/cold_start_results.jsonl
```

---

## 8. 结论

把 LIF 神经元用 **prefix-sum + soft-reset** 重新表达后，VGG16-SNN 在两种推理路径上都更快：

- **eager 路径 7.96 ms / 张**（最快，-14% vs NIR-compile）
- **compile 路径 8.24 ms / 张**（-11% vs NIR-compile，**且峰值显存只 6 GiB**）

性能与显存收益的根本原因**不在算法本身的并行性**（T=4 的 cumsum 与 4 步 elementwise 几乎与顺序 LIF 等价），而在 **把神经元逻辑从 `torch.library.custom_op` 黑盒改回普通 ATen op chain** 后：

1. **Inductor 拿到了全程 fuse + buffer reuse 权限**，原来 SJ LIF kernel `@triton.autotune restore_value` 的 ~3 GiB clone 不见了；
2. **15 个 LIF custom_op 调用换成可融合的 elementwise op 流**，Inductor 可与上游 conv epilogue / 下游 view 交叉融合，省下若干 kernel launch。

这给"自定义 Triton Pass 介入 SNN 神经元逻辑"留下了通道 —— 不再被 custom_op 边界挡住。
