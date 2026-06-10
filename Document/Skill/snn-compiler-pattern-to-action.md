# snn_compiler 工作原理：识别什么模式 → 执行什么动作

本文用**具体可复跑的例子**说明 snn_compiler 到底做了什么。它的加速分**两层**，
每层都是一组「模式 → 动作」的规则：

| 层 | 输入模式 | 动作 | 在哪 |
|---|---|---|---|
| **① 图重写**（Python/模块级） | `nn.Module` 里相邻的 `Conv/Linear (→BN) → 脉冲神经元` | 把这几个子 module **替换**成一个融合 module | [`passes/fuse.py`](../../snn_compiler/passes/fuse.py)、[`nn/modules.py`](../../snn_compiler/nn/modules.py) |
| **② kernel 特化**（Triton/编译级） | 融合 module 的属性（有无 bias / 残差 / 硬软复位 / 阈值形态 / 内存布局 / 神经元类型） | 把这些特征翻译成 **`tl.constexpr`**，让 Triton JIT 出**只含该分支**的专用 kernel | [`kernels/fused.py`](../../snn_compiler/kernels/fused.py) |

下面逐层讲，每条规则都给出**触发条件、动作、源码位置**，并附**真实运行输出**。

> 想要的只是"怎么用"，看 [使用指南](./snn-compiler-usage-guide.md)；想"如何确保没算错"，
> 看 [§7 自检](#7-动作之外先自检verify)。本文讲的是"内部发生了什么"。

---

## 1. 第一层：图重写——识别相邻模式，替换为融合 module

### 1.1 先定义：什么算"脉冲神经元"？

替换前要先认出哪个 module 是神经元。规则（[`fuse.py` `_is_neuron`](../../snn_compiler/passes/fuse.py)）：

- 是本框架的 `IFNode`/`LIFNode`/`CubaLIFNode`/`EIFNode` → 直接认；
- 或 **duck-typing**：类名叫 `IFNode`/`LIFNode` 且有 `v_threshold`（LIF 还需 `tau`）属性
  → 认（这样 **SpikingJelly 的神经元**也能被识别）。

认出后，`_neuron_kwargs` 从它身上**抽取动力学参数**交给融合 module：
`v_threshold`、`v_reset`（SJ 用 `None` 表示软复位 → 翻成 `soft_reset=True`）、
`tau`、`decay`、`decay_input`、`soft_reset`。

### 1.2 三条替换规则（在 `nn.Sequential` 内逐位扫描）

`fuse_snn_model` 顺序遍历 `nn.Sequential` 的子 module，按下表匹配相邻模式
（源码 [`_fuse_seq`](../../snn_compiler/passes/fuse.py)）：

| # | 识别到的模式（相邻） | 执行的动作（替换为） | 条件 |
|---|---|---|---|
| 1 | `Conv2d → BatchNorm2d → 神经元` | **`FusedConvBNNeuron`** | 三者相邻且第三个是神经元 |
| 2 | `Conv2d → 神经元`（无 BN） | **`FusedConvNeuron`** | 两者相邻 |
| 3 | `Linear → 神经元` | **`FusedLinearNeuron`** | 两者相邻 |
| — | 其它（含 `nn.Sequential` 子块） | 递归进入；匹配不上的原样保留 | — |

匹配成功就**跳过被吃掉的几层**、把融合 module 接到新序列里，并 `n_fused += 1`。
返回 `(新模型, n_fused)`——`n_fused` 就是"融了几处"。

### 1.3 真实例子 A：VGG 式 Sequential

```python
import torch, torch.nn as nn
from snn_compiler.nn import IFNode
from snn_compiler.passes import fuse_snn_model

model = nn.Sequential(
    nn.Conv2d(3,16,3,padding=1,bias=False), nn.BatchNorm2d(16), IFNode(),  # 模式1
    nn.Conv2d(16,16,3,padding=1,bias=False), IFNode(),                     # 模式2（无BN）
    nn.AvgPool2d(2), nn.Flatten(2),
).eval().cuda()
fused, n = fuse_snn_model(model, layout="NHWC")
```

**实际输出**（`tests` 同款环境，RTX 5070 Ti）：

```text
BEFORE:                          AFTER:
  (0): Conv2d                      (0): FusedConvBNNeuron   ← 模式1：Conv+BN+IF 三合一
  (1): BatchNorm2d                 (1): FusedConvNeuron     ← 模式2：Conv+IF 二合一
  (2): IFNode                      (2): AvgPool2d           ← 没匹配上，原样保留
  (3): Conv2d                      (3): Flatten
  (4): IFNode
  (5): AvgPool2d            fuse_snn_model -> n_fused = 2
  (6): Flatten             fused[0] = FusedConvBNNeuron: fold_bn=True neuron='if' layout='NHWC'
```

7 个子 module → 4 个；两处相邻模式被融合，`AvgPool2d`/`Flatten` 因匹配不上而保留。

### 1.4 残差 / 非顺序拓扑：动作要显式选

`fuse_snn_model` **只认 `nn.Sequential` 内相邻的线性链**。残差/分支不是顺序结构，
要用另外两个入口：

| 入口 | 适用 | 动作 |
|---|---|---|
| [`fuse_modules_path(model, [(conv,bn,neuron),…])`](../../snn_compiler/passes/fuse.py) | 已有的 block 类，forward 写法是 `neuron(bn(conv(x)))` | 按**路径**把指定三元组换成 `FusedConvBNNeuron`，被吃掉的 conv/bn 设为 `nn.Identity()`（forward 自动等价直通） |
| 直接构造 `FusedConvBNAddNeuron` / `FusedAddNeuron` | 残差/多分支合流 | 把 `+残差 → 神经元` 一步融进**一个** kernel（见 §2） |

### 1.5 真实例子 B：标准 ResNet-18（残差 = 加在神经元之前）

zoo 的 `resnet18_snn(fused=True)` 对每个 BasicBlock 调用 `block.fuse()`：
`block1 = FusedConvBNNeuron`、`block2 = FusedConvBNAddNeuron`（第二路把 `+identity`
吸进神经元 kernel）。**实际模块统计**：

```text
BEFORE: Conv2d:20  BatchNorm2d:20  IFNode:17  FusedConvBNNeuron:0  FusedConvBNAddNeuron:0
AFTER : Conv2d:3   BatchNorm2d:3   IFNode:0   FusedConvBNNeuron:9  FusedConvBNAddNeuron:8
```

读法：17 个神经元全被吸收（`IFNode:0`）；`9 = 8 个 block 的第一路 + 1 个 stem`，
`8 = 8 个 block 的第二路`。**为什么还剩 3 个 `Conv2d`+`BatchNorm2d`？** 那是
layer2/3/4 的 **downsample 分支**——它只有 conv+bn、**后面没有神经元**，不匹配任何
模式，所以原样保留（它的输出作为 `FusedConvBNAddNeuron` 的 `residual` 传入）。

> 这正好对照 [SEW-ResNet](#5-专题sew-resnet为什么不能直接套)：SEW 的 downsample **自带神经元**，
> 残差**加在神经元之后**，所以动作完全不同。

---

## 2. 第二层：kernel 特化——把模块属性翻译成 constexpr

融合 module 的 `forward` 做两件事：(1) 算 conv（**不加 bias**）；(2) 调
`fused_bias_if_lif(...)`，把"这是什么神经元、有没有 bias/残差、怎么复位、阈值什么形态、
什么布局"翻译成一组 `tl.constexpr`，Triton 据此 JIT 出**专用** kernel。

### 2.1 模块属性 → constexpr / 标量 的映射

源码 [`fused_bias_if_lif`](../../snn_compiler/kernels/fused.py)：

| 模块/调用的特征 | 翻译成 | kernel 里的动作 |
|---|---|---|
| 神经元 `if` | `decay_factor=1.0, input_scale=1.0` | `v = v + (y+bias)` |
| 神经元 `lif` | `decay_factor=1-1/τ`，`input_scale=1/τ`（`decay_input=False` 则 `=1`） | `v = decay·v + scale·(y+bias)` |
| `bias is None`？ | `HAS_BIAS: constexpr` | 编译期决定是否 `load(bias)`；无 bias 时该分支整段消失 |
| `residual` 传了吗？ | `HAS_RESIDUAL: constexpr` | 决定是否 `load(residual)` 并 `+r_t` |
| `soft_reset`？ | `RESET_MODE: constexpr`（0 软 / 1 硬） | 软：`v -= spike·v_th`；硬：`v = v·(1-spike) + spike·v_reset` |
| `v_threshold` 形态 | `THR_MODE: constexpr`（0 标量 / 1 per-channel `[C]` / 2 per-neuron `[NCL]`） | 决定阈值从常量 / `v_th[c]` / `v_th[n]` 取 |
| `layout` | `CHANNEL_LAST: constexpr` | 通道索引 NHWC `n%C` vs NCHW `(n//HW)%C` |
| `T`（时间步） | `T: constexpr` | `tl.static_range(0,T)` 把时间循环**编译期完全展开** |

因为这些都是 `constexpr`，Triton 对每种组合**特化出独立 kernel**：没残差的版本里
连 `residual` 的 load 都不存在——所以融合"零抽象税"。`@triton.autotune` 再按
`key=[T,NCL,C,THR_MODE,RESET_MODE,CHANNEL_LAST,HAS_BIAS,HAS_RESIDUAL]` 在
`BLOCK_NCL∈{128…1024}`、`num_warps∈{4,8}` 里选最快配置。

### 2.2 kernel 核心：外层并行 + 寄存器里跑时间循环

```python
# _bias_if_lif_kernel（节选，已去 mask/偏移细节）
v = 0.0                                  # 膜电位常驻寄存器
for t in tl.static_range(0, T):          # ← T 步在 kernel 内展开，v 不落显存
    y_t = load(y_ptr + t*NCL + idx)      # conv 输出（无 bias）
    if HAS_RESIDUAL: y_t += load(residual_ptr + t*NCL + idx)
    v = decay_factor * v + input_scale * (y_t + bias)     # 充电
    spike = (v >= v_th)                                   # 发放
    if RESET_MODE == 0: v -= spike * v_th                 # 软复位
    else:               v  = v*(1-spike) + spike*v_reset  # 硬复位
    store(spike_ptr + t*NCL + idx, spike)                 # 写脉冲
```

关键动作：**膜电位 `v` 全程在寄存器**（朴素实现每步都要 `load v`/`store v`），
**grid 沿 B·C·H·W 并行**（每个神经元独立、无需同步），**i64 偏移**避免
`T·NCL` 超 2³¹ 时的地址回绕。这套模板已打满显存带宽，所以加速主要来自**省 launch**
（Conv-BN-Neuron 三次启动 → 一次）而非把神经元算得更快。

### 2.3 其它 kernel 变体（同一套「模式→动作」思路）

| 模式 | kernel | 动作 |
|---|---|---|
| 最后一层 LIF 是"投票" | `_bias_if_lif_rate_kernel` | T 步内累加 spike-count，只写一次 `[NCL]`（省 T/2× 写带宽） |
| 大 T 显存放不下 | `_bias_if_lif_stateful_kernel` | 带 `v_init`/`v_out`，分块串接 v 状态 |

---

## 3. `fold_bn`：同一个 `FusedConvBNNeuron`，两种动作

`FusedConvBNNeuron`/`FusedConvBNAddNeuron` 用 `fold_bn` 选**怎么处理 BN**：

| `fold_bn` | 动作 | 数值 |
|---|---|---|
| `True`（默认，最快） | 构造时把 BN **折进 conv 的 weight/bias**（[`fold_conv_bn`](../../snn_compiler/kernels/fused.py)：`W'=γ/√(σ²+ε)·W`，`b'=γ/√(σ²+ε)·(b-μ)+β`），运行时只剩 conv+neuron | 数学等价，但有 ~1e-3 扰动 → 脉冲硬阈值下会翻转个别边界脉冲，**非逐位一致** |
| `False`（逐位精确） | conv 与 BN 仍是**两个独立 eager 算子**（`F.conv2d` 后 `F.batch_norm`），只融神经元 | 与原网络**逐位一致**，代价是多一次 BN kernel |

折叠把 BN 的 affine 吸进 conv bias，所以折叠版 `bias = b'`（`HAS_BIAS=True`）；
不折叠版 BN 自己算完，神经元 kernel 收到 `bias=None`（`HAS_BIAS=False`）。

---

## 4. 端到端：例子 A 的一次 forward 都发生了什么

```
x_seq [T,B,3,H,W]
   └─(0) FusedConvBNNeuron.forward
        ├─ F.conv2d(x, W', bias=None)              # W' 已折 BN（fold_bn=True）
        └─ fused_bias_if_lif(y, bias=b', neuron='if', soft_reset=False, layout='NHWC')
              → constexpr: HAS_BIAS=1, HAS_RESIDUAL=0, RESET_MODE=1(hard),
                            THR_MODE=0(scalar), CHANNEL_LAST=1, T=T
              → Triton JIT 出"有 bias / 无残差 / 硬复位 / 标量阈值 / NHWC"专用 kernel
              → 一个 kernel 内跑完 T 步充电-发放-复位，输出 spike [T,B,16,H,W]
   └─(1) FusedConvNeuron.forward                    # 同上，但 HAS_BIAS=0（无 BN/无 bias）
   └─(2) AvgPool2d  └─(3) Flatten                   # 原样
```

朴素实现这里要启动：conv、BN、bias-add、neuron 充电、发放、复位…多个 kernel；
融合后 **(0) 只启动 `conv` + 1 个神经元 kernel**。

---

## 5. 专题：SEW-ResNet 为什么不能直接套

这是最容易**静默算错**的拓扑，也是"模式不同 → 动作必须不同"的最佳示例：

| | 标准 ResNet（§1.5） | **SEW-ResNet** |
|---|---|---|
| 残差相加 | 神经元**之前**：`neuron(conv_bn(x) + identity)` → 用 `FusedConvBNAddNeuron`（`HAS_RESIDUAL=1`） | 神经元**之后**：`neuron(conv_bn(x)) ⊕ identity` → 用 `FusedConvBNNeuron` + 普通 `⊕` |
| downsample | 仅 conv+bn（不融，作 residual） | conv+bn **+ 自带神经元** → 也是一个 `FusedConvBNNeuron` |

用错（拿标准 ResNet 的 `FusedConvBNAddNeuron` 接 SEW）会**算成另一个网络且不报错**。
因此 SEW 现在是**一等支持**：[`zoo/sew_resnet.py`](../../snn_compiler/zoo/sew_resnet.py) 的
`sew_resnet18/34_snn(connect_f=ADD|AND|IAND, fused=, fold_bn=)`，其 `fuse()` 对
三段 conv-bn-neuron（含 downsample 分支）各建一个 `FusedConvBNNeuron`，`⊕` 保持普通逐元素。

---

## 5.5 专题：脉冲注意力（把同一套配方搬到 Transformer）

卷积型的配方是「识别 `线性→BN→神经元` → 融合 + 折常数 + T 维寄存器循环 + constexpr 特化」。
**脉冲注意力可套同一配方**——因为它**无 softmax**（脉冲 Q/K/V 非负稀疏，归一化交给 LIF）：

| 识别到的模式 | 执行的动作 |
|---|---|
| `投影(Conv1d/RepConv)→BN→LIF`（产生脉冲 Q/K/V） | 复用卷积侧的「投影+BN+LIF」融合：LIF→Triton kernel、BN 可折 |
| `kᵀ@v`（脉冲×脉冲） | `torch.bmm`（cutlass；朴素 Triton 二值 GEMM 实测打不过它） |
| `(q@kv)*scale → attn_lif` | **融合 `spike_av_lif`**：膜电位 `v[N,d]` 寄存器跨 T、每 t 现算 `q[t]@kv[t]`、**不落注意力图**、scale 折进输入尺度 |

`fuse_spiking_attention(model)` 探测整块替换（duck-type 同时认 Spikingformer SSA 与 SDT-V2 MS 两种形态）。
因 q∈{0,1}、kv 为小整数，`q@kv` 全程精确整数 → `spike_av_lif` 与 `bmm`+LIF **逐位一致**。
实测两类脉冲 transformer 全模型逐位一致、单块 1.5–1.9×（详见 [README §6.5](../../snn_compiler/README.md) 与
[探索日志](../Exploration/spiking-attention-optimization-journal.md)）。这正是「卷积型优化方法可推广到脉冲注意力」的落地。

---

## 6. 决策表：你的模式 → 该用的动作

| 你的网络长这样 | 动作 |
|---|---|
| 纯 `nn.Sequential`（VGG 式） | `fuse_snn_model(model, layout=, fold_bn=)` |
| 已有 block 类、forward 是 `neuron(bn(conv(x)))` 串行 | `fuse_modules_path(block, [(…)] )` |
| 标准 ResNet 残差（加在神经元前、downsample 无神经元） | 每路 `FusedConvBNNeuron` + 第二路 `FusedConvBNAddNeuron`（或用 `resnet18/34_snn`） |
| **SEW** 残差（加在神经元后、downsample 自带神经元） | `sew_resnet18/34_snn` 或 `FusedConvBNNeuron`×N + 普通 `⊕` |
| 多分支合流 `neuron(a+b)` | `FusedAddNeuron` |
| 最后一层 LIF 是 rate 投票 | `RateCodedLIFNode`（rate kernel） |
| **脉冲注意力块**（脉冲 Transformer：无 softmax 的脉冲 Q/K/V + `(q@(kᵀ@v))*scale` + LIF） | `fuse_spiking_attention(model)` 探测并整块替换为 `FusedSpikeAttention`（投影 LIF→Triton、KᵀV=bmm、核=融合 `spike_av_lif`）。两类实现（Spikingformer SSA / SDT-V2 MS）逐位一致，单块 1.5–1.9×。见 README §6.5 与 [探索日志](../Exploration/spiking-attention-optimization-journal.md) |
| 其它注意力/矩阵乘（含 softmax 或非脉冲 Q/K/V） | 不在融合 pattern 内，走 `torch.compile`（见 [`examples/snn_triton_pipeline`](../../examples/snn_triton_pipeline)） |

---

## 7. 动作之外：先自检（verify）

任何融合都可能改变结果（拓扑接错 / BN 折叠）。**信任加速模型前跑一行**：

```python
from snn_compiler import assert_equivalent
assert_equivalent(reference_model, fast_model, x)   # 默认要求逐位一致；不等价显式报错
```

**真实输出**（SEW-ResNet-18，`fold_bn=False`，bit-exact 路径）：

```text
[snn_compiler.verify] shape_match=True  max|Δ|=0.000e+00  rel=0.000e+00  top1-agree=100.00%
```

它把原网络与加速网络喂同一输入逐元素比对，超容差就报错并按相对误差判别原因
（拓扑接错 / BN 折叠）。详见 [verify.py](../../snn_compiler/verify.py) 与
[使用指南 §6](./snn-compiler-usage-guide.md)。

---

## 8. 源码地图

| 关心什么 | 看哪 |
|---|---|
| 模式识别 + 替换（动作①） | [`passes/fuse.py`](../../snn_compiler/passes/fuse.py)（`_is_neuron`/`_neuron_kwargs`/`_fuse_seq`/`fuse_modules_path`） |
| 融合 module 的 forward + `fold_bn` | [`nn/modules.py`](../../snn_compiler/nn/modules.py) |
| 属性→constexpr→kernel（动作②） | [`kernels/fused.py`](../../snn_compiler/kernels/fused.py)（`fused_bias_if_lif`/`_bias_if_lif_kernel`/`fold_conv_bn`） |
| 朴素参考实现（对拍用） | [`kernels/neurons.py`](../../snn_compiler/kernels/neurons.py)（`naive_if_lif`） |
| SEW 一等支持 | [`zoo/sew_resnet.py`](../../snn_compiler/zoo/sew_resnet.py) |
| 自检 | [`verify.py`](../../snn_compiler/verify.py) |
| 本文例子的可复跑测试 | [`tests/test_safe_and_sew.py`](../../snn_compiler/tests/test_safe_and_sew.py)、[`tests/test_residual_and_zoo.py`](../../snn_compiler/tests/test_residual_and_zoo.py) |
