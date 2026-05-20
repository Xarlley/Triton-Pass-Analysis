# 让 SpikingJelly SNN 完整走 Triton 编译：诊断与调优

> 本文记录把一个 SpikingJelly 多步 SNN（VGG16）通过 `torch.compile` **完整、可靠地**
> 下沉到 Triton 编译的全过程：所需的背景知识、诊断方法，以及基于 SpikingJelly
> 源码对 `recompile_limit` 隐患的根因分析与修复。
>
> 适用对象：需要让自定义 Triton GPU Pass 作用到整个 SNN 的编译器开发场景。
> 验证环境：torch 2.11.0+cu130、spikingjelly 0.0.0.0.14、RTX 5070 Ti、Triton 自定义构建。

---

## 1. 背景：为什么必须"完整走 Triton 编译"

本项目要为 SNN 开发自定义的 TritonGPU Pass。Pass 运行在 Triton 的编译流水线里，
**只能作用于真正经由 Triton 编译的 kernel**。如果模型的一部分：

- 回退到 **eager**（根本不进 `torch.compile` 的图），或
- 被 Inductor 降级为 **extern kernel**（cuDNN / cuBLAS 的预编译库调用），

那这部分代码就绕开了 Triton，自定义 Pass 永远看不到它。因此目标是：
**整个 SNN 的每一个算子，每次运行都经过 Triton 编译**。

初次用 `torch.compile` 跑 VGG16-SNN 时，控制台出现：

```
torch._dynamo hit config.recompile_limit (8)
   function: 'seq_to_ann_forward' (spikingjelly/activation_based/functional.py:653)
```

最终只有约 4 个 Triton kernel 被编译，模型大部分在 eager 下执行。这就是
"recompile_limit 隐患"。

---

## 2. 必备知识：`torch.compile` 的编译栈

### 2.1 四级流水线

```
torch.compile(model)(x)
  └─ TorchDynamo     —— 字节码级别捕获 Python，产出 FX Graph
       └─ AOTAutograd —— 算子分解、（必要时）生成反向图
            └─ TorchInductor —— 把 FX Graph 下降为 kernel；GPU 上生成 Triton 代码
                 └─ Triton —— 把 Triton DSL 编译为 PTX/cubin（自定义 Pass 在此介入）
```

要"完整走 Triton"，必须同时满足两点：**(A) Dynamo 把整个模型都捕获进图**、
**(B) Inductor 把所有算子都降级为 Triton kernel 而非 extern**。

### 2.2 图中断（graph break）

Dynamo 逐字节码翻译 Python。遇到无法静态分析的构造（动态控制流、无法确定类型的
`isinstance`、调用不可追踪的 C 扩展等）时，它会**图中断**：把已追踪的部分编成一张
子图，中断点处的代码退回 **eager Python** 执行，之后再重新进入追踪。一次中断把模型
切成两段子图。中断越多，图越碎，跨算子融合越差。

关键事实：**如果 Dynamo 在内联函数 `f` 的过程中发生图中断，它就无法把 `f` 内联进
父图**——`f` 会被当作一个独立的编译帧（code object）单独处理。这条规则是后文因果链
的核心。

### 2.3 重编译（recompile）与 `recompile_limit`

Dynamo 以 **Python 帧（code object）** 为单位编译，并为每个编译结果附加一组
**guard**（如输入张量的 dtype/shape、参数对象的类型）。同一个 code object 被以不满足
已有 guard 的参数再次调用时，Dynamo 会**重编译**出一个新版本。

`torch._dynamo.config.recompile_limit`（旧名 `cache_size_limit`，默认 **8**）限制
**单个 code object** 能缓存的版本数。一旦某个 code object 的版本数超过该上限，Dynamo
就**放弃它、把它加入跳过名单**——该函数此后**永远以纯 eager 执行**。

这正是隐患机理：某个函数被以多种参数反复重编译触顶后，它（及它调用的所有算子）就
彻底离开了 Triton。

### 2.4 Inductor：extern kernel vs Triton 模板

Inductor 把 FX 图里的算子降级为 kernel。逐元素 / 规约 / 池化 / BatchNorm 等会原生
生成 **Triton kernel**；但**卷积和矩阵乘法默认降级为 extern kernel**——即直接调用
`extern_kernels.convolution`（cuDNN）、`extern_kernels.addmm`（cuBLAS），**不经过
Triton**。

要让卷积 / 矩阵乘法也走 Triton，必须开启 `max_autotune`，它会为 conv/GEMM 生成
Triton 模板并自动调优；再把候选后端限定为 `TRITON`，排除 `ATEN`，Inductor 就只能
选用 Triton 模板。

---

## 3. 诊断工具与方法

| 工具 | 用途 |
|---|---|
| `TORCH_LOGS=output_code` | 打印 Inductor 生成的 wrapper。`grep async_compile.triton` 数 Triton kernel；`grep extern_kernels.` 数 extern 调用 |
| `TORCH_LOGS=graph_breaks` | 逐条打印图中断的**位置与原因**（含用户栈） |
| `TORCH_LOGS=recompiles` | 打印每次重编译的触发原因（哪条 guard 失效） |
| `torch._dynamo.explain(model)(x)` | 返回 `graph_count` / `graph_break_count` / `op_count`，一眼看出模型被切成几张图 |
| `torch._dynamo.utils.counters["graph_break"]` | 运行后统计图中断总数，可写进脚本做断言 |
| `TORCHINDUCTOR_CACHE_DIR=/tmp/fresh` | 指向空目录，强制冷编译，确保测量的是真实编译行为而非缓存 |

诊断套路：**先用 `explain` 看图被切成几张**，再用 `graph_breaks` 日志定位**每个中断
的源码行**，最后用 `output_code` 日志确认 **extern kernel 是否清零**。

---

## 4. 问题定位：基于 SpikingJelly 源码

### 4.1 多步层如何分发

调用 `functional.set_step_mode(model, 'm')` 后，模型里的 `layer.Conv2d`、
`layer.BatchNorm2d`、`layer.MaxPool2d`、`layer.Linear`、`layer.Flatten` 都进入
**多步模式**。以 `layer.Conv2d` 为例（`spikingjelly/activation_based/layer.py`）：

```python
class Conv2d(nn.Conv2d, base.StepModule):
    ...
    def forward(self, x: Tensor):
        if self.step_mode == 's':
            x = super().forward(x)
        elif self.step_mode == 'm':
            if x.dim() != 5:
                raise ValueError(...)
            x = functional.seq_to_ann_forward(x, super().forward)   # 多步分发
        return x
```

多步模式下，输入是 `[T, N, C, H, W]`，该层把 **`super().forward`**（即父类
`nn.Conv2d` 的、绑定到本层实例的 forward —— 一个 **bound method**）连同输入一起
交给 `functional.seq_to_ann_forward`。BN / MaxPool / Linear / Flatten 的多步分支
完全同构。本 VGG16-SNN 共有 **35 个**这样的多步层（13 Conv + 13 BN + 5 MaxPool +
3 Linear + 1 Flatten）。

> 注：`LIFNode` 等脉冲神经元**不经由** `seq_to_ann_forward`，它们有自己的
> `multi_step_forward`，可被 Dynamo 正常追踪，不是本问题的来源。

### 4.2 `seq_to_ann_forward` 源码

`spikingjelly/activation_based/functional.py:653`：

```python
def seq_to_ann_forward(x_seq, stateless_module):
    y_shape = [x_seq.shape[0], x_seq.shape[1]]
    y = x_seq.flatten(0, 1)                                    # [T, N, ...] -> [T*N, ...]
    if isinstance(stateless_module, (list, tuple, nn.Sequential)):   # ← functional.py:682
        for m in stateless_module:
            y = m(y)
    else:
        y = stateless_module(y)
    y_shape.extend(y.shape[1:])
    return y.view(y_shape)                                     # [T*N, ...] -> [T, N, ...]
```

它把时间维折叠进 batch，对无状态 ANN 层做一次前向，再还原形状。问题出在
**第 682 行的 `isinstance`**。

### 4.3 因果链：一个 `isinstance` 引发的连锁反应

`stateless_module` 实参是 `super().forward`，一个 **bound method**。Dynamo 对
bound-method 这类对象的 `isinstance` 无法静态判定类型，`graph_breaks` 日志给出：

```
Graph break in user code at .../spikingjelly/activation_based/functional.py:682
Graph Break Reason: ... builtin isinstance() cannot determine type of argument
```

由此触发连锁反应：

1. **`isinstance` 引发图中断** —— 在 `seq_to_ann_forward` 函数体内部。
2. **中断使 `seq_to_ann_forward` 无法被内联**（见 §2.2 的关键事实）。它从"被内联进
   父图的一段代码"变成**一个独立的编译帧（code object）**。
3. 这个独立帧被全部 **35 个多步层**复用调用，而每次的 `stateless_module` 类型
   （Conv2d / BN / MaxPool / Linear …）和输入形状都不同 → Dynamo 为它缓存出
   **几十个带不同 guard 的版本**。
4. 版本数迅速超过 `recompile_limit = 8` → Dynamo **放弃该 code object** →
   `seq_to_ann_forward` 此后**纯 eager 执行** → 经它分发的所有层都不再进 Triton。
   这就是只剩约 4 个 Triton kernel 的原因。
5. 即便在触顶之前，每个已编译版本里仍带着那个图中断 → 整个模型被切成 **18 张子图**
   （`explain` 实测 `graph_count=18, graph_break_count=17`）。

**一句话**：`functional.py:682` 的一个 `isinstance` 同时制造了"图碎裂"和
"重编译触顶后回退 eager"两个症状。

---

## 5. 解决方案

### 5.1 根因修复：替换 `seq_to_ann_forward`（消除图中断）

既然根因是那个 `isinstance`，且本项目的多步层**只会以单个 Callable** 调用
`seq_to_ann_forward`，就在运行时把它替换成不含 `isinstance` 分支的等价实现：

```python
def patch_spikingjelly_for_full_graph():
    from spikingjelly.activation_based import functional as sjf

    def _seq_to_ann_forward_single_graph(x_seq, stateless_module):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y = x_seq.flatten(0, 1)
        y = stateless_module(y)        # 多步层均传入单个 Callable
        y_shape.extend(y.shape[1:])
        return y.view(y_shape)

    sjf.seq_to_ann_forward = _seq_to_ann_forward_single_graph
```

层代码以 `functional.seq_to_ann_forward` 动态查找该函数，因此在 `torch.compile`
之前替换模块属性即可全局生效。

效果（`explain` 实测）：函数体内**不再有图中断 → Dynamo 完整内联它 → 它作为独立
code object 消失 → 重编译计数随之消失**。整个 VGG16-SNN 收敛为
**`graph_count=1, graph_break_count=0, op_count=754`** —— 单一计算图。

> 这是真正的根因修复：图中断、独立帧、重编译触顶三个问题**同时**消失。

### 5.2 保险措施：提高 `recompile_limit`

```python
torch._dynamo.config.recompile_limit = 256
torch._dynamo.config.cache_size_limit = 256
```

§5.1 修复后 `seq_to_ann_forward` 已被内联、不再单独计数，本不会再触顶。但把上限
调高作为**防御性保险**：万一未来引入别的会按形状重编译的帧，也不至于悄悄回退
eager。单独使用本措施（不做 §5.1）只能消除 eager 回退，**消不掉 17 个图中断**。

### 5.3 消除 extern kernel：`max_autotune` + Triton-only 后端

```python
import torch._inductor.config as ic
ic.max_autotune = True
ic.max_autotune_gemm_backends  = "TRITON"   # 默认 "ATEN,TRITON,CPP"
ic.max_autotune_conv_backends  = "TRITON"   # 默认 "ATEN,TRITON"
```

`max_autotune` 让 Inductor 为卷积 / 矩阵乘法生成 Triton 模板并自动调优；把后端
限定为 `TRITON`（排除 `ATEN`）后，Inductor **只能**选用 Triton 模板。效果：
`extern_kernels.convolution` 从 8 降到 **0**，卷积与矩阵乘法全部成为 Triton kernel。

代价：`max_autotune` 会对每个卷积自动调优，编译耗时增加数分钟。

> 调优日志里会出现几十条 `No valid triton configs. OutOfMemoryError: out of
> resource: triton_convolution2d Required: 147456 Hardware limit: 101376`。这是
> **正常噪声**：自动调优在枚举候选配置时淘汰了共享内存超限的配置，保留可用配置，
> 不影响正确性。

### 5.4 关闭 Inductor 缓存：`force_disable_caches`

```python
ic.force_disable_caches = True
```

Inductor 的 FXGraphCache / 自动调优缓存一旦命中，会**直接返回已编译产物，跳过代码
生成与 Triton 编译**——这样自定义 SNN Pass 就不会运行。关闭缓存保证**每次运行都
真正重新生成并编译**（配合 `TRITON_ALWAYS_COMPILE=1` 时，Pass 每次都会作用到）。

代价：每次运行都重新自动调优。若只需快速校验数值、不需要 Pass 介入，可临时关掉
此项。

---

## 6. 诊断数据对照

| 阶段 | 配置 | recompile 触顶 | 图中断 | extern kernel | Triton kernel |
|---|---|---|---|---|---|
| 初始 | 默认 | **是（限 8）** | 多 | 多 | ~4（其余 eager） |
| 仅提高 limit | `recompile_limit=256` | 否 | **17（仍碎）** | 8 个 conv | 24 |
| + max_autotune | 上 + Triton-only 后端 | 否 | 17 | **0** | 31 |
| + 替换 seq_to_ann | 全部修复 | 否 | **0** | **0** | 48（单图） |

最终：`graph_count=1`、`graph_break=0`、`extern_kernels=0`、`recompile` 警告 0。
整个 VGG16-SNN 收敛为**单一、全 Triton 的计算图**。

---

## 7. 关于可复现性：SNN 脉冲量化的副作用

把卷积从 cuDNN 换成 Triton 模板会改变浮点累加顺序、产生不同的舍入。但实测推理
输出在所有配置下**逐位一致**（`sum = -0.21972893178462982` 始终不变）。

原因是网络里有 **15 层 LIF 脉冲神经元**：推理时 LIF 的前向是一个**阈值函数**——
膜电位 ≥ `v_threshold` 就发放 `1`，否则 `0`。脉冲是**离散信号**，卷积输出里
亚-ULP 量级的数值扰动通常不足以让任何神经元跨过阈值 → 脉冲序列不变 → 逐层传递
后最终 logits 不变。**脉冲量化把连续的数值噪声"吸收"掉了**，使 SNN 成为一个异常
稳健的可复现基准。

但这**不是保证**：若某次扰动恰好让一个临界神经元翻转脉冲，输出就会变。因此
`vgg16_test.py` 保存"黄金输出"并每次比对——一旦真有数值错误（例如自定义 Pass
引入了 IR bug），比对会立刻报出来。这正是该测试脚本对 §2.1 Pass 开发的价值。

---

## 8. 配置速查表

```python
# —— Dynamo：避免重编译触顶回退 eager ——
torch._dynamo.config.recompile_limit = 256
torch._dynamo.config.cache_size_limit = 256

# —— Inductor：卷积/矩阵乘法走 Triton 而非 cuDNN/cuBLAS ——
import torch._inductor.config as ic
ic.max_autotune = True
ic.max_autotune_gemm_backends = "TRITON"
ic.max_autotune_conv_backends = "TRITON"
ic.force_disable_caches = True          # 每次都真正重新编译

# —— SpikingJelly：消除多步层处的图中断（根因修复）——
#   见 §5.1，把 functional.seq_to_ann_forward 替换为无 isinstance 版本

# —— 运行时环境变量 ——
#   TRITON_ALWAYS_COMPILE=1   强制 Triton 每个 kernel 都重新编译（Pass 才会运行）
#   ENABLE_SNN_PASS=1         触发本项目自定义 SNN Pass
#   TORCH_LOGS=output_code    诊断：查看生成代码 / 统计 extern kernel
#   TORCH_LOGS=graph_breaks   诊断：定位每个图中断的源码行
```

参考实现见 [`examples/vgg16_snn/vgg16_test.py`](../../examples/vgg16_snn/vgg16_test.py)
的 `configure_full_triton_compilation()` 与 `patch_spikingjelly_for_full_graph()`。

---

## 9. 复现 / 验证命令

```bash
# 完整跑一次（脚本内已内置上述全部配置）
python examples/vgg16_snn/vgg16_test.py

# 诊断：统计 extern kernel 与 Triton kernel
TORCH_LOGS=output_code python examples/vgg16_snn/vgg16_test.py 2>&1 \
  | grep -c 'extern_kernels\.'         # 期望 0
TORCH_LOGS=output_code python examples/vgg16_snn/vgg16_test.py 2>&1 \
  | grep -c 'async_compile.triton'     # 期望 >0

# 诊断：定位图中断
TORCH_LOGS=graph_breaks python examples/vgg16_snn/vgg16_test.py 2>&1 \
  | grep 'Graph break'                 # 期望无输出
```

脚本 `[5/5]` 阶段会打印 `dynamo 图中断数: 0`，并把推理输出与黄金输出逐位比对。

---

## 10. 参考源码位置

| 内容 | 位置 |
|---|---|
| `seq_to_ann_forward` 定义 | `spikingjelly/activation_based/functional.py:653` |
| 触发图中断的 `isinstance` | `spikingjelly/activation_based/functional.py:682` |
| 多步层分发（Conv2d 为例） | `spikingjelly/activation_based/layer.py` · `Conv2d.forward` |
| Dynamo 重编译上限 | `torch._dynamo.config.recompile_limit` / `cache_size_limit` |
| Inductor 自动调优后端 | `torch._inductor.config.max_autotune{,_gemm_backends,_conv_backends}` |
| 本项目应用 | `examples/vgg16_snn/vgg16_test.py` |
