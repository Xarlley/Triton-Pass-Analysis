# snn_compiler 对脉冲 Transformer 的推理优化：原理与实测

> 本文系统阐述 snn_compiler 如何**探测并融合脉冲注意力块**（即脉冲 Transformer 中的注意力模块）。
> 在「探测并融合卷积型脉冲神经网络」这一已有能力之外，本文展示 snn_compiler 如何把同一套优化方法推广到注意力机制，并附上在 A100 显卡上的真实实测数据。
> 本轮工作的逐步探索过程记录在 [`Document/Exploration/spiking-attention-optimization-journal.md`](../Exploration/spiking-attention-optimization-journal.md) 中；简明的使用方法见 [snn_compiler/README.md 第 6.5 节](../../snn_compiler/README.md) 与 [snn-compiler-pattern-to-action.md 第 5.5 节](./snn-compiler-pattern-to-action.md)。
>
> **测量环境**：所有测量都在一块 NVIDIA A100-SXM4-40GB 显卡（单卡，且是与他人共享的机器）上完成，使用的软件环境是 `triton-src`（其中 PyTorch 版本为 2.12，Triton 为从源码编译的 3.7 版本）。
> 本文使用的两个真实模型是 Spikingformer-8-768（时间步数 T=4，通道数 C=768，注意力头数为 8，每头维度 d=96）与 Meta-SpikeFormer-55M（又称 Spike-Driven-Transformer-V2，时间步数 T=4，通道数 C=512，注意力头数为 8，每头维度 d=64）；两个模型的 token 数都是 N=196（对应 14×14 的空间分辨率）。
> 本文的每一次测速都遵守两条纪律：第一，**GPU 占用门控**——在测量之前先用 `nvidia-smi` 确认没有其他人的进程占用显卡（本轮全部测量时他人占用均为 0 MiB）；第二，**冷启动感知**——每次测量先预热至少 20 次（以吸收 Triton 自动调优、即时编译、cuDNN 算法选择、显存分配器的首次开销），再在 `cuda.synchronize` 之后取多次迭代的中位数，同时记录第 10 与第 90 百分位以观察抖动。
> snn_compiler 在整个工作中保持独立：它只把 Triton 当作库来使用（通过 `@triton.jit` 装饰器编写自己的 kernel），**没有修改任何 Triton 源码**。

---

## 1. 为什么脉冲注意力可以「像卷积一样」被通用优化

### 1.1 两个主流脉冲 Transformer 的注意力核心几乎逐字相同

我们对比了 examples 中两个真实模型的注意力前向代码，它们分别是 Spikingformer 的 `SpikingSelfAttention`（脉冲自注意力）模块与 Spike-Driven-Transformer-V2 的 `MS_Attention_RepConv_qkv_id` 模块。这两个模块**最关键的两行代码完全一致**：

```python
# 这两个模型都用下面两行来计算注意力 —— 它们逐字相同
x = k.transpose(-2, -1) @ v      # 先算 K 的转置与 V 的矩阵乘，得到 KᵀV
x = (q @ x) * self.scale         # 再算 Q 与 (KᵀV) 的矩阵乘，并乘以缩放系数（两个模型都取 0.125）
```

这两个模块的整条数据流也是同构的，如下表所示：

| 阶段 | Spikingformer 的 SpikingSelfAttention | Spike-Driven-Transformer-V2 的 MS_Attention | 本质 |
|---|---|---|---|
| 输入脉冲 | `proj_lif(x)` | `head_lif(x)` | 一个 LIF 神经元把输入转成脉冲 |
| 生成 Q、K、V | `lif(BN(Conv1d_1×1(x)))`，重复三次 | `lif(BN(RepConv(x)))`，重复三次 | 投影层接批归一化层再接 LIF 神经元，产生三个脉冲张量 Q、K、V |
| 组合 | `(q @ (kᵀ@v)) * 0.125` | `(q @ (kᵀ@v)) * 0.125` | 不含 softmax 的、按线性顺序排列的矩阵乘 |
| 非线性 | `attn_lif`（发放阈值 0.5） | `attn_lif`（发放阈值 0.5） | 一个 LIF 神经元 |
| 输出投影 | `BN(Conv1d(x))` | `BN(RepConv(x))` | 一个线性投影层 |

这两个模块的唯一差别在于投影算子的形态：Spikingformer 用的是 1×1 的一维卷积（Conv1d），而 Spike-Driven-Transformer-V2 用的是可重参数化卷积 RepConv。除此之外，这两个模块的注意力机制本身是同一个。

（说明：LIF 神经元指的是漏积分发放（Leaky Integrate-and-Fire）神经元；BN 指批归一化层（BatchNorm）；softmax 指标准 Transformer 注意力里的归一化指数函数；本文后续会沿用这些名称。）

### 1.2 共同原理（这是可以通用优化的根据）

这两个模型的注意力共享四条性质：

1. Q、K、V 这三个张量全都是脉冲张量，其每个元素的取值都是 0 或 1（因为它们都由「投影层接批归一化层再接 LIF 神经元」产生）。
2. 这两个模型的注意力里都**没有 softmax**。这是最深层的结构性共同点：脉冲张量本身就是非负且稀疏的，所以注意力的归一化作用被交给了后面的 `attn_lif` 神经元来承担。
3. 正因为没有 softmax，这两个矩阵乘可以利用结合律重排成线性顺序，即先算 `KᵀV` 再算 `Q(KᵀV)`，从而避开 N×N 的注意力矩阵，使计算复杂度降到 O(N·d²)。这两个模型的源码里本就是这样写的。
4. 这两个模型注意力里唯一的非线性运算仍然是 LIF 神经元。

「没有 softmax」是这一切的钥匙：它同时保证了三件事——其一，矩阵乘的结合律重排在数学上成立；其二，两个矩阵乘的操作数始终保持为脉冲或小整数，因而可以用门控累加或二进制运算来替代浮点的通用矩阵乘；其三，缩放系数是一个常数，可以折叠进 LIF 神经元。由于这一族的脉冲 Transformer 都共享上述性质，所以这套优化是可以通用的。

### 1.3 时间都花在哪里（阿姆达尔定律决定了优化的优先级）

我们对加载了真实权重的模型逐层挂上钩子，并用「按实测发放率生成的伯努利脉冲」作为输入来对各个组件单独计时（配置为 Spikingformer 形态，批大小 B=16，每头维度 d=96）。结果如下表：

| 组件 | 中位数耗时 | 在注意力核心中的占比 |
|---|---:|---|
| KᵀV 矩阵乘 | 0.124 毫秒 | —— |
| 完整的 `(q@(kᵀv))*scale`（包含两个矩阵乘） | 0.282 毫秒 | 矩阵乘约占 44% |
| `attn_lif`（SpikingJelly 的 eager 模式、torch 后端，本质是 Python 层的时间步循环） | 0.434 毫秒 | —— |
| `attn_lif`（snn_compiler 的 Triton LIF kernel） | **0.186 毫秒** | **比 eager 模式快 2.3 倍** |

这里有一个关键发现：整个注意力块在 eager 模式下约耗时 2.6 毫秒，但「两个矩阵乘加上 `attn_lif`」这个核心部分只占约 0.72 毫秒。这意味着，整个注意力块约 70% 的时间花在了 Q、K、V 与输出这四个投影上（每个投影都是「投影层接批归一化层再接 eager 模式的 LIF」）。而这恰恰是 snn_compiler 在卷积侧用 `FusedConvBNNeuron` 与 `FusedLinearNeuron` 已经能够优化的东西。因此，**最大的、而且是本框架已经具备的优化收益落在投影上，而不在矩阵乘上**。

我们还测量了真实的脉冲发放率（加载训练权重，对 8 个注意力块取跨块均值）：

| 模型 | q_lif 发放率 | k_lif 发放率 | v_lif 发放率 | attn_lif 发放率 |
|---|---|---|---|---|
| Spikingformer | 0.072 | **0.026** | 0.047 | 0.159 |
| Spike-Driven-Transformer-V2 | 0.166 | **0.034** | 0.078 | 0.138 |

由此可见，Q、K、V 三个张量都极其稀疏（其中 K 的发放率仅约 3%）。在计算 KᵀV 时，这个矩阵乘的两个操作数的发放率都只有约 3% 到 8%，所以稠密的浮点通用矩阵乘会浪费掉超过 99% 的浮点运算量。这说明用二进制运算或 popcount 来计算 KᵀV 有很大的优化空间（详见第 4 节）。

---

## 2. 优化的「模式 → 动作」

snn_compiler 对脉冲注意力的优化遵循与卷积侧相同的方法论：先识别模式，再融合，再把常数折叠进神经元，再把时间步循环放进寄存器，最后用编译期常量（constexpr）做特化。本框架把整个脉冲注意力块整体识别出来，并对其中三类子结构分别执行三类动作，如下表：

| 识别到的模式 | 本框架执行的动作 | 对应代码 |
|---|---|---|
| 「投影层（Conv1d 或 RepConv）接批归一化层再接 LIF」，用于生成脉冲 Q、K、V | 本框架把这里的 LIF 替换为 snn_compiler 的 Triton LIF kernel（它与 SpikingJelly 的 LIF 逐位一致），并且可选地把批归一化折叠进卷积（由参数 `fold_bn` 控制） | 复用 `FusedConvBNNeuron`，包装在 `nn/attention.py` 中 |
| `kᵀ@v`（脉冲与脉冲相乘） | 本框架默认用 `torch.bmm`（即 cutlass 库的批量矩阵乘），也可以改用 `spike_ktv_popcount`（先做二进制位打包再做 popcount，速度更快且逐位精确，详见第 4 节） | `kernels/attention.py` |
| `(q@kv)*scale → attn_lif` | 本框架把这一段融合成一个 kernel `spike_av_lif`：它让膜电位常驻寄存器并跨整个时间步循环、从不把注意力图写回显存、并把缩放系数折叠进 LIF 的输入尺度 | `kernels/attention.py` |

入口函数 `fuse_spiking_attention(model)` 会遍历模型的所有子模块，用鸭子类型（duck typing）的方式识别脉冲注意力块（它能同时识别 SpikingSelfAttention 与 MS_Attention 两种形态），并就地把它们替换为 `FusedSpikeAttention` 模块。

### 2.1 核心融合 kernel `spike_av_lif`

本框架把第二个矩阵乘、缩放、以及 `attn_lif` 神经元这三步融合成**一个** kernel。对于每一个「批次×注意力头」的组合，这个 kernel 取出一块 token，沿着时间步维度，在**寄存器**里维持膜电位 `v[BLOCK_N, d]`；在每一个时间步上，它现场计算 `a[t] = q[t] @ kv[t]`、更新膜电位、并直接写出脉冲。这个 kernel **从不把 `[T, B, heads, N, d]` 形状的注意力图写回显存**（相比之下，朴素的 `torch.bmm` 方案会先把这张图写进显存，之后再读回来）。下面是这个 kernel 的伪代码：

```python
v_mem = 0                                   # 膜电位常驻寄存器
for t in tl.static_range(0, T):             # 时间步维度在 kernel 内被完全展开
    q_t  = load(q[t, bh, n_tile, :])        # 取出脉冲 Q（取值为 0 或 1）
    kv_t = load(kv[t, bh, :, :])            # 取出 KᵀV（它是二值矩阵内积得到的小整数）
    a    = tl.dot(q_t, kv_t)                # 计算 q@kv（这是精确的整数运算）
    v_mem = decay * v_mem + input_scale * a # LIF 充电；这里 input_scale = (1/τ)*scale，已把缩放系数折进来
    spike = (v_mem >= v_th)                 # 判断是否发放脉冲
    v_mem = v_mem * (1 - spike) + spike * v_reset   # 硬复位
    store(spike[t, bh, n_tile, :], spike)   # 直接写出脉冲
```

这个 kernel 与卷积侧的「卷积输出接 LIF」融合是同构的，区别仅在于：这里循环体内做的是一个小矩阵乘，而卷积侧做的是卷积。

### 2.2 为什么这个融合是逐位精确的

在这个 kernel 里，Q 的每个元素取值为 0 或 1，而 `kv`（即 KᵀV）是两个二值矩阵做内积得到的结果，所以 `kv` 的每个元素都是不超过 N 的小整数（N=196，小于 2 的 10 次方）。因此，`q@kv` 这个矩阵乘全程都是**精确的整数运算**（它的部分和不超过 d 乘 N，约为 1.9 万，小于 2 的 24 次方，所以 32 位浮点累加是精确的；即使输入被转成 TensorFloat-32 格式，对于不超过 2048 的整数也是精确的）。此外，缩放系数 0.125 等于 2 的负 3 次方、衰减项里的 1/τ 等于 0.5，它们都是 2 的整数次幂，所以折叠时不产生任何舍入误差。LIF 神经元本身是逐元素运算。综合这几点，`spike_av_lif` 与「`torch.bmm` 加上朴素 LIF」这条参考路径是**逐位一致**的（实测最大绝对误差为 0）。同理，snn_compiler 的 Triton LIF 与 SpikingJelly 的 `MultiStepLIFNode` 也是逐位一致的（在复位电位为 0、采用硬复位、且 decay_input 为真的设定下，SpikingJelly 的充电公式 `H = V(1-1/τ) + X/τ` 与本框架的实现等价）。

---

## 3. 集成进 snn_compiler 的接口

下面是把这套优化用到一个模型上的最小代码：

```python
from snn_compiler.passes import fuse_spiking_attention
from snn_compiler import assert_equivalent

# 探测并就地替换所有脉冲注意力块（鸭子类型能同时识别 Spikingformer 与 SDT-V2 两种形态）
n = fuse_spiking_attention(model, fold_bn=False, ktv_mode="bmm")   # 该函数返回被替换的块数
assert_equivalent(reference_model, model, x)   # 在信任加速结果之前先做校验（默认要求逐位一致）
```

关于这些参数：

- 参数 `fold_bn` 默认为 `False`，此时本框架保证逐位一致；若设为 `True`，本框架会更快，但批归一化的折叠会翻转个别处于阈值边界的脉冲（这个现象与卷积侧完全相同）。
- 参数 `ktv_mode` 默认为 `'bmm'`（即用 cutlass 的批量矩阵乘，稳妥）；也可设为 `'popcount'`（即用二进制位打包加 popcount 的方式计算 KᵀV，它逐位精确而且更快，详见第 4 节）。
- 如果用户想单独构造一个融合模块，可以调用 `FusedSpikeAttention.from_reference(ref_block, fold_bn=False, ktv_mode='popcount')`。
- 相关代码位于 `kernels/attention.py`、`nn/attention.py`、`passes/attention_fuse.py`；对应的测试位于 `tests/test_spike_attention.py`。

---

## 4. 用脉冲二值性加速 KᵀV：bit-pack 加 popcount（一个由负转正的结果）

KᵀV 这个矩阵乘可以改写成 popcount 的形式：`KᵀV[i,j] = Σ_n K[n,i]·V[n,j] = Σ_w popcount(Kpack[i,w] & Vpack[j,w])`。这里的思路是，本框架把 token 维度 N 上每 32 个脉冲打包进一个 32 位整数（于是字数 W 等于 196 除以 32 向上取整，即 7），这样矩阵乘里沿 N 维的收缩就从 196 步降到了 7 步（约为原来的二十八分之一）。其中 popcount 指的是统计一个整数的二进制表示里 1 的个数。

本框架对这个想法的探索分成几步：

- 第一步，本框架先写了一个朴素的 Triton 二值矩阵乘 kernel，但它**打不过 cutlass 的 `torch.bmm`**（这是一个负结果，在 d=96 时只有 cutlass 的 0.58 倍速度），原因是 cutlass 的张量核心（tensor core）实现太强。
- 第二步，本框架改用 popcount 方案：单看 popcount kernel 本身，它比 `torch.bmm` 快 4 到 5 倍，而且逐位精确；**但是用 torch 来做二进制位打包的那一步很慢**，把这部分收益吃掉了，导致包含打包在内的完整路径反而更慢。
- 第三步，本框架另写了一个受显存带宽限制的 Triton 打包 kernel 来替代 torch 的打包之后，整条完整路径就**净胜** `torch.bmm` 了（具体数据见第 5.5 节）。

因此，本框架把 `spike_ktv_popcount` 作为一个**可选项**收进了框架（默认仍然用 `torch.bmm`；在那些 `libdevice.popc` 不可用的 Triton 构建上，本框架会自动退回到 `torch.bmm`）。需要说明的是，由于 KᵀV 只占整个注意力块约 5% 的时间（这是阿姆达尔定律的结论），所以这个优化在端到端层面只带来约 2% 到 3% 的提升，但它是免费的，而且逐位精确。

---

## 5. 实测实验数据

### 5.1 正确性（参数 fold_bn 为 False，使用真实预训练权重）

| 检查项 | Spikingformer 的 SpikingSelfAttention | Spike-Driven-Transformer-V2 的 MS_Attention |
|---|---|---|
| 鸭子类型探测 | 8 个块全部识别 | 8 个块全部识别 |
| 单块的 `assert_equivalent` | 最大绝对误差为 **0**（逐位一致） | 最大绝对误差为 **0** |
| **替换全部块之后的整模型端到端对比** | 最大绝对误差为 **0**，top-1 一致率 **100%** | 最大绝对误差为 **0**，top-1 一致率 **100%** |
| 确定性基线（原模型连跑两次的差异） | 该模型的卷积路径存在 cutlass 非确定性抖动，详见第 6 节方法学 | **0**（该模型路径是确定的），这证明本融合是干净的 |

在更细的层面：`spike_av_lif` 与「`torch.bmm` 加 LIF」的最大绝对误差为 0；`spike_ktv_popcount` 与 `torch.bmm` 的最大绝对误差为 0（这个对比也包含了 N 不是 32 的整数倍、需要做掩码的情形）。

### 5.2 单块测速：融合版对比 eager 版与 torch.compile 版（时间步数 T=4，批大小 B=16）

| 模型 | eager 版 | FusedSpikeAttention（本框架） | torch.compile 版 | 融合版相对 eager 版 | 融合版相对 torch.compile 版 |
|---|---:|---:|---:|---:|---:|
| Spikingformer 的 SpikingSelfAttention | 2.54 毫秒 | **1.34 毫秒** | 1.72 毫秒 | **快 1.90 倍** | **快 1.29 倍** |
| Spike-Driven-Transformer-V2 的 MS_Attention | 4.62 毫秒 | **3.01 毫秒** | —— | **快 1.53 倍** | —— |

需要强调的是，FusedSpikeAttention 在更快的同时还保持逐位一致，而 torch.compile 并不保证这一点。

### 5.3 批大小扫描（时间步数 T=4，单块，单位为毫秒中位数）

下面是 Spikingformer 的 SpikingSelfAttention（通道数 C=768）在不同批大小下的结果：

| 批大小 B | eager 版 | 融合版 | torch.compile 版 | 融合版相对 eager 版 | 融合版相对 torch.compile 版 |
|---:|---:|---:|---:|---:|---:|
| 8  | 2.928 | 1.511 | —— | **快 1.94 倍** | —— |
| 16 | 3.291 | 2.285 | 2.659 | 快 1.44 倍 | 快 1.16 倍 |
| 32 | 5.515 | 4.059 | —— | 快 1.36 倍 | —— |
| 64 | 10.431 | 7.749 | 9.299 | 快 1.35 倍 | 快 1.20 倍 |

下面是 Spike-Driven-Transformer-V2 的 MS_Attention（通道数 C=512）在不同批大小下的结果：

| 批大小 B | eager 版 | 融合版 | torch.compile 版 | 融合版相对 eager 版 | 融合版相对 torch.compile 版 |
|---:|---:|---:|---:|---:|---:|
| 8  | 4.222 | 2.820 | —— | **快 1.50 倍** | —— |
| 16 | 4.612 | 3.227 | 6.349 | 快 1.43 倍 | **快 1.97 倍** |
| 32 | 7.275 | 6.065 | —— | 快 1.20 倍 | —— |
| 64 | 13.889 | 11.813 | 18.380 | 快 1.18 倍 | 快 1.56 倍 |

从这两张表可以看出：融合版相对 eager 版快 1.18 到 1.94 倍，相对 torch.compile 版快 1.16 到 1.97 倍，也就是说融合版相对这两条基线都更快。融合版在批大小较小时优势最大（因为此时 eager 模式 LIF 的启动开销在总时间里占比更大），在批大小较大时优势收窄（因为此时计算量增大，运算变得更受算力限制）。

### 5.4 时间步数扫描（Spikingformer 的 SpikingSelfAttention，批大小 B=16）

| 时间步数 T | eager 版 | 融合版（用 bmm） | 融合版（用 popcount） | 融合版相对 eager 版 | popcount 版相对 eager 版 | 是否逐位一致 |
|---:|---:|---:|---:|---:|---:|---|
| 4  | 3.708 | 2.274 | 2.269 | **快 1.63 倍** | **快 1.63 倍** | 是 / 是 |
| 8  | 5.607 | 4.308 | 4.094 | 快 1.30 倍 | 快 1.37 倍 | 是 / 是 |
| 16 | 10.317 | 8.137 | 7.869 | 快 1.27 倍 | 快 1.31 倍 | 是 / 是 |

从这张表可以看出：融合版的优势在时间步数较小时最大（在 T=4 时达到 1.63 倍），在时间步数较大时收窄到约 1.27 到 1.31 倍。原因在于，融合带来的收益主要是节省 kernel 启动开销，而这部分开销在时间步数较大时会被摊薄（这与卷积侧「启动开销在小问题上占比更大」的规律是一致的）。这两种 KᵀV 模式在所有时间步数上都保持逐位一致（因为本框架的 kernel 用 `static_range(0, T)` 展开时间步循环，能适配任意时间步数）。

需要补充的是：这个时间步数扫描的结果与我们最初的假设是相反的。我们最初猜测融合版的优势会随时间步数增大而增大，但实测表明优势随时间步数增大而减小。我们如实修正了这个判断，原因正如上一段所述。

### 5.5 popcount 版的 KᵀV 对比 cutlass 的 bmm（完整路径，包含 Triton 打包 kernel）

| 配置 | bmm（cutlass） | 打包 kernel（两个中的一个） | **完整路径（两次打包加一次 popcount）** | **完整路径相对 bmm** | 正确性 |
|---|---:|---:|---:|---:|---|
| Spikingformer 形态，批大小 16，每头维度 96 | 0.285 毫秒 | 0.090 毫秒 | **0.203 毫秒** | **快 1.41 倍** | 最大绝对误差为 0 |
| SDT-V2 形态，批大小 16，每头维度 64 | 0.293 毫秒 | 0.066 毫秒 | **0.148 毫秒** | **快 1.99 倍** | 最大绝对误差为 0 |
| Spikingformer 形态，批大小 64，每头维度 96 | 0.963 毫秒 | 0.167 毫秒 | **0.432 毫秒** | **快 2.23 倍** | 最大绝对误差为 0 |

从这张表可以看出：逐位精确的 popcount 版 KᵀV 净胜 cutlass 的 `torch.bmm` 1.4 到 2.2 倍，而且优势随批大小增大而增大。若只看 popcount kernel 本身（不计打包的耗时），它比 `torch.bmm` 快 4 到 5 倍。

---

## 6. 方法学（为什么这些数据是可信的）

本文的数据可信，是因为我们在测量时遵守了以下几条做法：

- **共享机的占用门控**：本机是与他人共享的 A100，所以我们在每次测速之前都用 `nvidia-smi` 确认了没有其他人的进程占用显卡；本轮所有测量时，他人进程占用的显存都是 0 MiB。
- **冷启动感知**：Triton 的自动调优与即时编译、cuDNN 的算法选择、以及显存分配器的开销都集中在前几次调用中发生，所以我们每次测量都先预热至少 20 次，再在 `cuda.synchronize` 之后取多次迭代的中位数，同时记录第 10 与第 90 百分位以观察抖动。
- **确定性基线**：我们用「让原模型在同一个输入上连跑两次」的方式来量化基线的非确定性。对于 Spike-Driven-Transformer-V2，这个基线差异是 0，因此本融合产出的结果是干净的逐位一致；而对于 Spikingformer，整模型偶尔出现的约 0.3 的差异来自**没有被替换的卷积与多层感知机路径上的 cutlass 原子操作抖动**（让原模型连跑两次也会出现同样的差异），它并不来自本融合——因为单块已经证明了逐位一致。
- **跨环境验证**：本框架最终的代码在 Triton 3.5.1（本地的 RTX 5070 Ti 显卡）与 Triton 3.7（A100 显卡）上都保持逐位一致；本地的 pytest 全套件有 17 个用例全部通过。

---

## 7. 边界与未来工作

我们如实说明这套优化的边界：

- **端到端的加速不等于单块的加速**：上面各表给出的多数是单个注意力块的数字。注意力块只是整个网络的一部分，所以整网的加速会受阿姆达尔定律的限制；而 KᵀV 又只占注意力块约 5% 的时间，所以 popcount 优化在端到端层面带来的提升较小（但它是免费的、且逐位精确）。
- **RepConv 的结构重参数化**（针对 Spike-Driven-Transformer-V2 的投影，其结构为「1×1 卷积接带填充的批归一化层，再接 3×3 深度卷积接 1×1 卷积接批归一化层」）已经被我们评估过，但**暂时搁置**。原因有三：其一，其中的 `BNAndPadLayer` 用批归一化算出的值来做边界填充，把整条链折叠成一个卷积并不简单、容易引入数值差异；其二，这个投影最主要的优化收益（也就是把 eager 模式的 LIF 换成 Triton LIF）已经被 `variant='ms'` 这条路径捕获了；其三，这项改动复杂度高、风险大，而增益是边际的。
- **把打包融进上游 LIF（fused-LIF-pack）**：本框架可以把 popcount 方案里的二进制位打包进一步融进上游产生 K、V 的那个 LIF kernel（也就是在神经元发放脉冲的同时就直接打包），从而再省去一次显存读写。我们已经验证了打包可以由 Triton 高效完成，这项工作留待后续。
- 此外，未来还可以把这套优化推广到更多脉冲 Transformer（例如 Spike-Driven-Transformer-V3、E-SpikeFormer 等），并把 `fuse_spiking_attention` 接进整网级的 pass 以及 examples 中的推理流水线。

---

## 8. 复现方式

下面给出复现本文实验的具体命令：

```bash
# 在 A100 上、使用 triton-src 环境
ssh -F ~/.ssh/config.a100 a100
conda activate triton-src
cd ~/charlley/snn_compiler_attn        # 该目录由 snn_compiler/explore/attention/push_to_a100.sh 推送得到
SJ_NEURON_BACKEND=torch python snn_compiler/explore/attention/phase5_integration.py   # 探测、整模型逐位一致、单块测速
SJ_NEURON_BACKEND=torch python snn_compiler/explore/attention/phase6_sdtv2.py          # 向 SDT-V2 泛化、确定性基线
python snn_compiler/explore/attention/phase4b_sweep.py                                 # 批大小扫描
SJ_NEURON_BACKEND=torch python snn_compiler/explore/attention/phase8_tsweep.py         # 时间步数扫描
python snn_compiler/explore/attention/phase6c_packkernel.py                            # popcount 版 KᵀV 对比 bmm
# 在本地（任意带 CUDA 与 Triton 的机器上）
python -m pytest snn_compiler/tests/test_spike_attention.py -q
```

上述每个实验脚本都内置了 GPU 占用门控与冷启动感知的中位数计时。逐步的探索过程以及每一步的原始数据，都记录在 [探索日志](../Exploration/spiking-attention-optimization-journal.md) 中。
