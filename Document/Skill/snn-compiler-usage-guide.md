# snn_compiler 使用指南：VGG / ResNet / MobileNet 与任意 SNN 的加速

> 适用版本：本仓库 `snn_compiler/` 含 `zoo/`、`passes/fuse.py` 含 `fuse_modules_path` 与
> `fuse_conv_bn_add_neuron_path` 之后。

本文档总结**如何把现有 SNN 套上 snn_compiler 取得加速**，按拓扑复杂度分四级：

1. 纯 `nn.Sequential` 网络（VGG 系）—— 一行 `fuse_snn_model`
2. 含残差合流的网络（ResNet / MobileNet-V2 倒残差 / DenseNet）—— 路径式 fuse + `FusedConvBNAddNeuron`
3. 多分支 / 门控网络 —— 直接构造 `FusedAddNeuron` 与其它融合 module
4. SpikingJelly 现有模型迁移 —— duck-typing 自动识别 + 包装

并给出本框架的**技术路径**：从单 kernel 出发，逐层扩展到多 neuron、多 reset、含残差融合、跨架构通用 pass。

---

## 目录

- [1. 五分钟起步](#1-五分钟起步)
- [2. 支持的架构与已实测加速](#2-支持的架构与已实测加速)
- [3. 如何把任意 SNN 接上本框架](#3-如何把任意-snn-接上本框架)
- [4. 关键 API 参考](#4-关键-api-参考)
- [5. 技术路径与设计原则](#5-技术路径与设计原则)
- [6. 调试与常见问题](#6-调试与常见问题)

---

## 1. 五分钟起步

```python
import torch
from snn_compiler.zoo import vgg16_snn, resnet18_snn, mobilenet_v2_snn

# 三种架构都用同一个 API：fused=True 直接出融合版
m = resnet18_snn(num_classes=1000, neuron="lif", tau=2.0,
                  soft_reset=False, v_threshold=1.0, v_reset=0.0,
                  layout="NHWC", fused=True).cuda().eval().to(torch.bfloat16)
x = torch.randn(4, 16, 3, 224, 224, device="cuda", dtype=torch.bfloat16)
with torch.no_grad():
    spike = m(x)        # [T=4, B=16, num_classes=1000]
```

需要拿自家模型替换的话，去 [§3](#3-如何把任意-snn-接上本框架)。

---

## 2. 支持的架构与已实测加速

`snn_compiler/zoo/` 提供五个引用实现，均通过 fused vs naive 的输出 `max|diff|=0` 验证（[snn_compiler/tests/test_residual_and_zoo.py](../../snn_compiler/tests/test_residual_and_zoo.py)）。

实测（RTX 5070 Ti，BATCH=16，T=4，H=W=224，LIF / hard reset，bf16+NHWC）：

| 架构 | naive (ms/img) | fused (ms/img) | 加速 |
|---|---|---|---|
| VGG-11 SNN | 2.10 | 1.14 | **1.84×** |
| VGG-16 SNN | 3.91 | 1.97 | **1.99×** |
| ResNet-18 SNN | 0.587 | 0.307 | **1.91×** |
| ResNet-34 SNN | 0.958 | 0.498 | **1.93×** |
| MobileNet-V2 SNN | 1.10 | 0.240 | **4.60×** |

`fp32 + NCHW` 同一矩阵：

| 架构 | naive (ms/img) | fused (ms/img) | 加速 |
|---|---|---|---|
| VGG-11 SNN | 2.534 | 1.898 | 1.34× |
| VGG-16 SNN | 4.935 | 3.786 | 1.30× |
| ResNet-18 SNN | 0.741 | 0.635 | 1.17× |
| ResNet-34 SNN | 1.233 | 1.079 | 1.14× |
| MobileNet-V2 SNN | 0.842 | 0.591 | 1.42× |

**注解**：bf16+NHWC 总是更高，因为 conv 计算变小后（tensor core），elementwise 启动税占比上升，融合收益更大。MobileNet-V2 是最大赢家，因为 depthwise/pointwise 把每个 layer 拆得更碎，启动税本身就比 VGG/ResNet 更重，融合后基本消掉。

完整数据：[snn_compiler/benchmarks/zoo_bench_results.jsonl](../../snn_compiler/benchmarks/zoo_bench_results.jsonl)。

---

## 3. 如何把任意 SNN 接上本框架

按你的网络拓扑选一条路径：

### 3.1 纯 Sequential — `fuse_snn_model` 一行收工

适用：VGG-style、纯 plain ConvNet-SNN、SpikingJelly Sequential 模型。

```python
import torch
import torch.nn as nn
from snn_compiler.nn import IFNode, LIFNode      # 或直接用 spikingjelly 的 IFNode/LIFNode
from snn_compiler.passes import fuse_snn_model

model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1, bias=False),
    nn.BatchNorm2d(64),
    LIFNode(tau=2.0, soft_reset=False),
    nn.Conv2d(64, 64, 3, padding=1, bias=False),
    nn.BatchNorm2d(64),
    LIFNode(tau=2.0, soft_reset=False),
    nn.AvgPool2d(2, 2),
    # ...
).eval().cuda()

fused, n_fused = fuse_snn_model(model, layout="NHWC")
print(f"fused {n_fused} Conv-BN-Neuron patterns")
# fused 是新的 nn.Sequential，可直接 forward
y_seq = fused(x_seq)    # x_seq: [T, B, 3, H, W]
```

`fuse_snn_model` 自动识别：

- `Conv2d → BN → IF/LIF` → `FusedConvBNNeuron`
- `Conv2d → IF/LIF` → `FusedConvNeuron`
- `Linear → IF/LIF` → `FusedLinearNeuron`

并兼容 SpikingJelly 的 `IFNode` / `LIFNode`（duck-typing）。

### 3.2 ResNet-style — `block.fuse()` 或路径式 fuse

ResNet/BasicBlock 有残差合流（`out = neuron2(conv2_bn2(neuron1(conv1_bn1(x))) + identity)`），不是 Sequential，`fuse_snn_model` 处理不了。三种做法：

**A. 用 zoo 现成实现**（推荐起步）

```python
from snn_compiler.zoo import resnet18_snn
m = resnet18_snn(num_classes=1000, fused=True).cuda().eval()
```

代码见 [snn_compiler/zoo/resnet.py](../../snn_compiler/zoo/resnet.py)，关键是每个 `BasicBlockSNN` 有 `fuse()` 方法。

**B. 已有 BasicBlock 类，手动 fuse 单个 block**

```python
from snn_compiler.nn import FusedConvBNNeuron, FusedConvBNAddNeuron
from snn_compiler.passes import _neuron_kwargs   # 用于从 neuron module 抽参数

def fuse_basicblock(block: nn.Module, layout="NHWC"):
    """block 必须有 conv1/bn1/neuron1/conv2/bn2/neuron2 这六个属性。"""
    kw1 = _neuron_kwargs(block.neuron1); kw1["layout"] = layout
    kw2 = _neuron_kwargs(block.neuron2); kw2["layout"] = layout
    dev, dt = block.conv1.weight.device, block.conv1.weight.dtype
    block.block1 = FusedConvBNNeuron(
        block.conv1.eval(), block.bn1.eval(), **kw1
    ).to(device=dev, dtype=dt)
    block.block2 = FusedConvBNAddNeuron(
        block.conv2.eval(), block.bn2.eval(), **kw2
    ).to(device=dev, dtype=dt)
    block.conv1 = block.bn1 = block.neuron1 = nn.Identity()
    block.conv2 = block.bn2 = block.neuron2 = nn.Identity()
    # 再改一下 block.forward 让它走 block1 / block2
```

`block.forward` 也要相应改成：

```python
def forward(self, x):
    identity = x if self.downsample is None else self.downsample(x_4d_view(x))
    out = self.block1(x)
    out = self.block2(out, identity)   # 注意：FusedConvBNAddNeuron.forward(x, residual)
    return out
```

**C. 不改类、不改 forward —— 用 `fuse_modules_path`**

```python
from snn_compiler.passes import fuse_modules_path

# 只融合"conv1-bn1-neuron1"半段；conv2-bn2-neuron2 残差路径仍走朴素加法
n = fuse_modules_path(block, [
    ("conv1", "bn1", "neuron1"),
], layout="NHWC")
# 之后 block.neuron1 是一个 FusedConvBNNeuron；block.conv1/bn1 已变成 nn.Identity
# block.forward 原本写的就是 x = self.neuron1(self.bn1(self.conv1(x)))，自动等价
```

这一招对**任何已有的 ResNet/Bottleneck/DenseBlock/InvertedResidual 类都生效**，前提是原 forward 写法是 `out = neuron(bn(conv(x)))`（顺序串行调用）—— 融合后 `bn(conv(x))` 变 Identity 链，等价直通。

### 3.3 多分支 / 门控 / 自定义合流

如果你的 block 有多条分支 `c = neuron(a + b + ...)`：

```python
from snn_compiler.nn import FusedAddNeuron

self.merge_neuron = FusedAddNeuron(neuron="lif", tau=2.0,
                                     soft_reset=False, layout="NHWC")
# forward:
spike = self.merge_neuron(branch_a, branch_b)
```

`FusedAddNeuron.forward(a_seq, b_seq)` 在单 Triton kernel 内 fused `a + b → IF/LIF`。两条以上分支，可在 Python 侧先 `a + b`，再 `FusedAddNeuron(a+b, c)`，每次只多一个 elementwise add。

### 3.4 SpikingJelly 已有模型迁移

`fuse_snn_model` 的 `_neuron_kwargs` 通过 duck-typing 识别非本框架的 `IFNode`/`LIFNode`：

```python
import spikingjelly
from spikingjelly.activation_based.layer import Conv2d, BatchNorm2d
from spikingjelly.activation_based.neuron import LIFNode as SJLIFNode
from snn_compiler.passes import fuse_snn_model

model = nn.Sequential(
    Conv2d(3, 64, 3, padding=1),
    BatchNorm2d(64),
    SJLIFNode(tau=2.0, v_threshold=1.0),
    # ...
).eval().cuda()
fused, n = fuse_snn_model(model)
```

注意：SJ `Conv2d` / `BatchNorm2d` 是 wrap 过的 `nn.Conv2d`/`nn.BatchNorm2d`，本 pass 的 `isinstance(a, nn.Conv2d)` 在大多数 SJ 版本下仍判 True；如果 SJ 改用 mix-in 不继承 nn.Conv2d，把它换回 `nn.Conv2d(...)` 即可。

`SJLIFNode` 的 `v_reset=None` 表示 soft reset，本 pass 已识别（见 [passes/fuse.py](../../snn_compiler/passes/fuse.py)）。

---

## 4. 关键 API 参考

### 4.1 入口：`snn_compiler.passes`

| API | 用途 |
|---|---|
| `fuse_snn_model(model, *, layout)` | Sequential 全自动模式匹配，最常用 |
| `fuse_modules_path(model, groups, *, layout)` | 路径式 fuse；不改 forward，不改类 |
| `fuse_conv_bn_add_neuron_path(model, conv_p, bn_p, neuron_p, *, layout)` | 显式把 (Conv, BN, Neuron) 替换为 FusedConvBNAddNeuron，用于 ResNet 残差 |

### 4.2 融合 module：`snn_compiler.nn`

| 类 | forward 签名 | 适用 |
|---|---|---|
| `IFNode` | `(x_seq) → spike_seq` | 纯 IF |
| `LIFNode` | `(x_seq) → spike_seq` | 纯 LIF |
| `CubaLIFNode` | `(x_seq) → spike_seq` | 二阶 LIF |
| `EIFNode` | `(x_seq) → spike_seq` | 指数 IF |
| `FusedConvNeuron` | `(x_seq) → spike_seq` | Conv → Neuron |
| `FusedConvBNNeuron` | `(x_seq) → spike_seq` | Conv → BN → Neuron |
| `FusedConvBNAddNeuron` | `(x_seq, residual_seq) → spike_seq` | Conv → BN → +Residual → Neuron （ResNet 残差） |
| `FusedAddNeuron` | `(a_seq, b_seq) → spike_seq` | +合流 → Neuron（无 conv） |
| `FusedLinearNeuron` | `(x_seq) → spike_seq` | Linear → Neuron |

所有 module 都支持：`decay`/`tau`、`decay_input`、`soft_reset`、`v_threshold`（scalar / per-C / per-N tensor）、`v_reset`、`layout`（NCHW/NHWC）。

### 4.3 底层 kernel 入口：`snn_compiler.kernels`

| API | 用途 |
|---|---|
| `if_lif(x_seq, *, neuron, ...)` | 直接调用 IF/LIF kernel |
| `cuba_lif(x_seq, *, tau_syn, tau_mem, ...)` | CubaLIF kernel |
| `eif(x_seq, *, tau, delta_T, ...)` | EIF kernel |
| `fused_bias_if_lif(y_seq, bias, *, residual, ...)` | Conv 输出+bias[+residual]→Neuron 融合 |
| `conv_neuron(x_seq, weight, bias, *, ...)` | Conv+Neuron 端到端 |
| `linear_neuron(x_seq, weight, bias, *, ...)` | Linear+Neuron 端到端 |
| `fold_conv_bn(...)` | BN→Conv 折叠工具函数 |

每个 neuron 都允许 `decay=...`（参考 [第 11 阶段 journal](../Exploration/mlir-perf-exploration-journal.md)）。

### 4.4 现成架构：`snn_compiler.zoo`

| 工厂 | 描述 |
|---|---|
| `vgg11_snn / vgg13_snn / vgg16_snn / vgg19_snn` | VGG-style，可选 BN，可选 pool 类型 |
| `resnet18_snn / resnet34_snn` | ResNet-BasicBlock 系列 |
| `mobilenet_v2_snn` | MobileNet-V2（倒残差 + depthwise） |

所有工厂都接受 `(num_classes, neuron, tau, decay_input, soft_reset, v_threshold, v_reset, layout, fused=True/False, init_bn=True/False)`。

---

## 5. 技术路径与设计原则

### 5.1 一个 MLIR-level pattern → 所有 neuron 模型

`v_t = f(v_{t-1}, x_t)`，`spike_t = (v_t ≥ θ)`，`v_t = reset(...)` —— per-position state-recurrent，沿 T 顺序递推。对应 kernel 模式：

```
grid = (ceil(NCL / BLOCK_NCL),)     # NCL = B*C*H*W，outer-parallel
for t in tl.static_range(0, T, 1):   # T-register-loop
    x_t = tl.load(...)
    v   = step(v, x_t, params)
    spike = (v ≥ v_th)
    v   = reset(v, spike, v_reset, RESET_MODE)
    tl.store(...)
```

详见 [Document/Paper/snn_compiler_paper.md §3.1](../Paper/snn_compiler_paper.md) 与 [journal §9](../Exploration/mlir-perf-exploration-journal.md)。

### 5.2 残差合流自然嵌入同一 kernel

ResNet 标志性的 `neuron(conv_bn(x) + identity)`，加号两边都是 `[T, B, C, H, W]`，对位独立。Triton kernel 在 T 循环每步多 load 一个 residual_ptr 即完成融合：

```python
for t in tl.static_range(0, T, 1):
    y_t = tl.load(y_ptr + t * NCL + idx)
    if HAS_RESIDUAL:
        r_t = tl.load(residual_ptr + t * NCL + idx)
        v = decay * v + scale * (y_t + bias + r_t)
    else:
        v = decay * v + scale * (y_t + bias)
    ...
```

`HAS_RESIDUAL` 是 constexpr，Triton 编译时按需 specialization；无 residual 路径与原版完全等价。

### 5.3 三类拓扑的同一抽象

| 拓扑 | 融合策略 | API |
|---|---|---|
| 顺序串联（VGG）| Sequential 模式匹配 | `fuse_snn_model` |
| 树状残差（ResNet/MobileNet）| 第二支 conv 接 residual 合流 | `FusedConvBNAddNeuron` |
| 任意分支合流 | 把 add 与 neuron 看成同一 kernel | `FusedAddNeuron` |

三者共用同一个底层 kernel `_bias_if_lif_kernel`，只在 Python 包装层区分 `HAS_BIAS`/`HAS_RESIDUAL` 两个 constexpr 开关。这是为什么 MobileNet-V2 的加速比（4.60×）能比 ResNet（1.93×）大 — 拓扑越碎，融合收益越多，但所用代码 100% 是同一份 kernel template。

### 5.4 何时不会有加速

| 情况 | 原因 |
|---|---|
| Inductor `torch.compile` 已经 fuse 了这条链 | 框架与 Inductor 收益重合；本框架在 SNN 上仍有结构性优势（journal §9 三大 gap），但 ANN 场景下差距小 |
| 卷积本身是大部分时间 | 融合节省的是 elementwise 启动税；如果 conv 自己已耗 95%+，加速会被卡到 1.05× |
| 模型只跑 T=1 | 注意：本框架对任意 T≥1 都正确；但 T=1 时退化成普通 conv-bn-neuron，无多步加速 |

测出加速比时建议把 `BATCH × T × HW` 都打满，确保不是 launch 数太少导致的「随机加速」。

---

## 6. 调试与常见问题

**Q1. `bit-equal` 验证失败 / fused 与 naive 输出差异 > 0**

- 检查 BN 是不是 `.eval()` 状态（`fuse_conv_bn` 用 running stats，train 模式不对）。
- 如果用 bf16，`naive` 路径会有 ULP 漂移；测试请用 fp32，或用 `same_spikes = (out > 0).eq(ref > 0).all()` 判定。
- `v_reset` 类型必须是 Python float，不能是 tensor（kernel 把它当 constexpr）。

**Q2. CUDA OOM**

- 把 BATCH 调小；融合后 peak 通常持平或略低，但 autotune 阶段 `restore_value` 仍要 clone。
- `bf16 + NHWC` 比 `fp32 + NCHW` 省 ~ 2× 显存。

**Q3. 多个 conv 之间 view 报错 `non-contiguous`**

- 在 NHWC pipeline 中，所有 conv 输出要 `.contiguous(memory_format=torch.channels_last)` 之后再 view 回 5D。
- 框架的 `FusedConvBNNeuron` 已经处理；自己写 module 时容易漏。

**Q4. SpikingJelly model fuse 后输出是 0 / NaN**

- 大概率是 `BatchNorm2d.running_var = 1.0`（未训练）但 `running_mean ≠ 0`，折叠后偏置被放大。框架对 BN running stats 不做安全检查 — 模型 fuse 前请确保 running_var 已稳定（≥ 0.01）。

**Q5. 如何看融合到底融了几层？**

- `fuse_snn_model` 返回 `(model, n_fused)`，直接看返回值。
- 对 ResNet/MobileNet，可在 `block.fuse()` 末尾 print 一下 `_fused` 标志。
- 也可以 `for name, m in model.named_modules(): if 'Fused' in type(m).__name__: print(name)`。

**Q6. 如何在自己的 SNN 上跑一个 sanity benchmark？**

```python
import torch, time
m_naive = my_snn_model().cuda().eval()
m_fused = fuse_snn_model(my_snn_model().cuda().eval())[0]
x = torch.randn(T, BATCH, ..., device="cuda")
for m, n in [(m_naive, "naive"), (m_fused, "fused")]:
    for _ in range(5): m(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50): m(x)
    torch.cuda.synchronize()
    print(f"{n}: {(time.perf_counter()-t0)*1000/50:.2f} ms/iter")
```

---

## 参考

- **架构与方法**：[Document/Paper/snn_compiler_paper.md](../Paper/snn_compiler_paper.md)
- **探索过程**：[Document/Exploration/mlir-perf-exploration-journal.md](../Exploration/mlir-perf-exploration-journal.md)，特别是 §10 / §11 / §12
- **API README**：[snn_compiler/README.md](../../snn_compiler/README.md)
- **测试**：
  - [snn_compiler/tests/test_correctness.py](../../snn_compiler/tests/test_correctness.py)（177 个 bit-equal 用例）
  - [snn_compiler/tests/test_graph_pass.py](../../snn_compiler/tests/test_graph_pass.py)（Sequential pass）
  - [snn_compiler/tests/test_residual_and_zoo.py](../../snn_compiler/tests/test_residual_and_zoo.py)（残差 + zoo）
- **Benchmark**：
  - [snn_compiler/benchmarks/bench_vgg16.py](../../snn_compiler/benchmarks/bench_vgg16.py)
  - [snn_compiler/benchmarks/bench_zoo.py](../../snn_compiler/benchmarks/bench_zoo.py)
  - [snn_compiler/benchmarks/sweep_all.sh](../../snn_compiler/benchmarks/sweep_all.sh)
