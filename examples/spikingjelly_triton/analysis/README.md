# SNN 代码到 PTX 汇编：完整编译流程解析

本目录记录了 `snn_example.py` 中 `SimpleSNN` 模型在真实运行时，经过 `torch.compile` + PyTorch Inductor + Triton 编译器逐层降级（Lowering）的每一步中间表示。

> **所有代码均来自真实运行**，从 `/tmp/torchinductor_charlley/` 和 `~/.triton/cache/` 中直接提取，绝非人工构造示例。

---

## 完整编译流程总览

```
用户 Python 代码 (SpikingJelly)
        │
        ▼ [TorchDynamo 追踪]
ATen FX Graph (AOT Autograd)
        │
        ▼ [TorchInductor 代码生成 + 算子融合]
Triton Kernel Source (.py)
        │
        ▼ [triton.compile(): ASTSource → ast_to_ttir]
Triton IR / TTIR (.ttir)               ← 进入 Triton 编译器 MLIR Pipeline
        │
        ▼ [GPU Layout Pass: convert-triton-gpu-to-ttgir]
Triton GPU IR / TTGIR (.ttgir)
        │
        ▼ [LLVM Lowering: convert-triton-to-llvm]
LLVM IR (.llir)
        │
        ▼ [LLVM NVPTX Backend]
PTX Assembly (.ptx)
        │
        ▼ [ptxas]
CUDA Binary (.cubin)  →  GPU 执行
```

---

## 第一层：用户代码 → [`01_user_code.py`](./01_user_code.py)

这是开发者编写的唯一代码层。

**关键 SpikingJelly 源码对应关系：**

| 用户代码 | SpikingJelly 源文件 | 位置 |
|---------|-------------------|------|
| `neuron.LIFNode(...)` | `spikingjelly/activation_based/neuron/lif.py` | 第 96 行 `class LIFNode(BaseNode)` |
| `surrogate.ATan()` | `spikingjelly/activation_based/surrogate.py` | 第 792 行 `class ATan(SurrogateFunctionBase)` |
| `lif(x)` 前向 | `surrogate.py` 第 776 行 `class atan(torch.autograd.Function)` → `forward()`: `return heaviside(x)` |
| `loss.backward()` | `surrogate.py` 第 770 行 `atan_backward(grad, x, alpha)` |

**LIFNode 的核心数学公式**（来自 `neuron/lif.py` 第 262 行）：
```python
# 充电方程 (decay_input=True, v_reset=0):
v = v + (x - v) / tau   # H[t] = V[t-1] + (X[t] - V[t-1]) / τ

# 发放方程 (SpikingJelly base_node.py):
spike = (v >= v_threshold).to(x)  # 前向: Heaviside (不可导)

# 重置方程 (hard reset):
v = v_reset * spike + (1.0 - spike) * v
```

**ATan 替代梯度公式**（来自 `surrogate.py` 第 770 行 `atan_backward`）：
```python
def atan_backward(grad_output, x, alpha):
    a = alpha / 2          # alpha=2.0, 故 a=1.0
    ax = math.pi * a * x   # π * x
    return a / (1 + ax * ax) * grad_output, None
    # 即: g'(x) = α / (2 * (1 + (π/2 * α * x)^2))
```

---

## 第二层：FX 图 → [`02_fx_graph.txt`](./02_fx_graph.txt)

**触发时机**：`torch.compile(model)` 后第一次调用 `compiled_model(x)` 时，TorchDynamo 追踪器启动。

**变换机制**：

1. **TorchDynamo** 拦截 Python 字节码，记录所有 Tensor 操作
2. **AOTAutograd** 展开自动微分图，将 `loss.backward()` 所需的完整计算预先展开为静态图
3. **SpikingJelly 的 `torch.autograd.Function`** 被透明展开：`atan.backward()` 中的每行 Python 算术都变成了一个独立的 `aten` 算子节点

**关键变换**（`atan_backward` 中的 `a / (1 + ax * ax)` 被分解为）：

```
aten.mul(div, 3.14159)       # ax = π * x
aten.pow(mul, 2)             # ax^2
aten.add(pow, 1)             # 1 + ax^2
aten.reciprocal(add)         # 1 / (1 + ax^2)
aten.mul(reciprocal, 1.0)    # * a (a=1.0, 无效乘)
aten.mul(mul, add_14)        # * chain_grad
aten.div(add_16, 2.0)        # /2  (对应 alpha=2 的整体因子)
```

> **重要**：`torch.autograd.Function` 的 `backward()` 是 Python 函数，TorchDynamo 会完整追踪其内容，而不仅仅是把它当做黑盒。这使得 Inductor 可以对 SpikingJelly 的替代梯度做进一步优化。

---

## 第三层：Triton Kernel → [`03_triton_kernel.py`](./03_triton_kernel.py)

**真实文件路径**：`/tmp/torchinductor_charlley/fr/cfraihlz75c32aaj5jxb5j32tha3hube67cgo2mfsx3j4awecujc.py`

**触发时机**：Inductor 分析完 FX 图后，将多个 aten 算子进行算子融合并生成此文件。

**算子融合**：本 Kernel 将以下操作融合为单一 Kernel（**省去了多次全局内存往返**）：
- LIF1 的 ATan 替代梯度（7 个 aten 算子）
- BatchNorm2d 反向传播的 Elementwise 部分（6 个 aten 算子）

**Triton 编译器入口**（对应 `triton/python/triton/compiler/compiler.py`，第 226 行 `compile()`）：

```python
# 当 Inductor 调用 async_compile.triton(...) 时，
# 最终会调用到 triton.compiler.compiler.compile(src) 其中 src 是 ASTSource
def compile(src, target=None, options=None, _env_vars=None):
    # ...
    # 获取或创建 cache_manager
    fn_cache_manager = get_cache_manager(hash)
    # 循环执行各个编译 Stage:
    for ext, compile_ir in list(stages.items())[first_stage:]:
        next_module = compile_ir(module, metadata)  # 每个 stage 产生下一级 IR
        metadata_group[ir_filename] = fn_cache_manager.put(next_module, ir_filename)
        # → 这就是为什么 /tmp/torchinductor_charlley/triton/0/HASH/ 目录下
        #   会有 .ttir, .ttgir, .llir, .ptx, .cubin 文件的原因！
```

**关键设计决策**（来自 `triton_heuristics.pointwise` 的 `inductor_meta`）：
- `'tiling_scores': {'x': 1404928}` — Inductor 评估了潜在的并行度
- `'mutated_arg_names': ['in_out_ptr0']` — 标记就地修改的参数
- `'optimize_mem': True` — 启用内存复用优化
- `'eviction_policy='evict_last'` — per-channel 统计量 ([16]) 的 L2 缓存保留策略

---

## 第四层：Inductor Wrapper → [`04_inductor_wrapper.py`](./04_inductor_wrapper.py)

**真实文件路径**：`/tmp/torchinductor_charlley/md/cmd257vas74ipq2tjgauqes7ilmrpkyrijcaqa6hoxp6g3xpvypa.py`

这是 Inductor 生成的完整反向传播调度代码，规划了 **9 个子步骤** 的精确执行顺序和显存管理策略。

**关键技术亮点**：

| 技术 | 体现 |
|-----|------|
| **显存复用** | `buf9 = buf8; del buf8  # reuse` —— 复用 `convert_element_type` 的显存给结果 |
| **形状 Guard** | `assert_size_stride(primals_1, (16,1,3,3), (9,9,3,1))` —— 保证形状与追踪时完全一致 |
| **混合后端** | 部分算子用 Triton (自定义)，矩阵乘用 `extern_kernels.mm` (cuBLAS) |
| **CUDA Stream** | `stream0 = get_raw_stream(0)` —— 所有 Kernel 在同一 Stream 上串行执行 |

---

## 第五~八层：Triton MLIR 编译 Pass → [`triton_passes/`](./triton_passes/)

这是 Triton 编译器内部对第三层 Triton Kernel 源码进行的逐步降级过程。
**真实缓存路径**：`/tmp/torchinductor_charlley/triton/0/WQZWHKDXGTUIK36K53HKSMUGNOKBSNYG4D7VHO6MHHNRK6PAKXGQ/`

### 5. Triton IR (`.ttir`) → [`triton_passes/01_unoptimized.ttir`](./triton_passes/01_unoptimized.ttir)

**对应 `triton/python/triton/compiler/compiler.py` 中的 Stage：`ttir`**

生成入口：`ASTSource.make_ir()` → `code_generator.ast_to_ttir()`（`compiler/code_generator.py`）

特征：
- 使用 `tt.*` 方言（dialect）的 MLIR 操作
- 张量无 GPU 布局信息，只有逻辑形状 `tensor<256xf32>`
- 内存操作：`tt.load / tt.store`，指针由 `tt.addptr` 计算

关键操作映射（Python Triton → TTIR）：

| Triton Python | TTIR 操作 |
|--------------|-----------|
| `tl.program_id(0)` | `tt.get_program_id x : i32` |
| `tl.arange(0, XBLOCK)` | `tt.make_range {end=256, start=0}` |
| `tl.load(ptr, mask)` | `tt.load %ptr, %mask : tensor<256xf32>` |
| `tmp8 * tmp8` | `arith.mulf %tmp8, %tmp8` |
| `1.0 / tmp10` | `arith.divf %cst_0, %tmp10` |

### 6. Triton GPU IR (`.ttgir`) → [`triton_passes/02_optimized_gpu.ttgir`](./triton_passes/02_optimized_gpu.ttgir)

**对应 Stage：`ttgir`**，在 Triton 的 NVIDIA CUDA 后端中由 `backend.add_stages()` 注册

这一步是 Triton 最重要的优化 Pass：**插入 GPU 线程块布局语义**。

```mlir
// 新增的 GPU 块布局属性 (文件第 1 行):
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
```

**含义解析**：
- `sizePerThread = [2]`：每个线程负责 **2 个** 连续 f32 元素
- `threadsPerWarp = [32]`：每个 Warp 有 **32** 个线程（标准 NVIDIA Warp 大小）
- `warpsPerCTA = [4]`：每个 CTA（线程块）有 **4** 个 Warp
- 因此每个 CTA 处理：`2 × 32 × 4 = 256` 个元素（即 `XBLOCK = 256`）

**这个布局保证了内存合并访问（Coalesced Memory Access）**：
- 同一 Warp 内的 32 个线程访问连续的内存地址
- GPU 内存控制器可以将多个访问合并为一次宽 DRAM 事务

所有张量类型从 `tensor<256xf32>` 变为 `tensor<256xf32, #blocked>`：
```mlir
# Before (ttir):  tensor<256xf32>
# After (ttgir):  tensor<256xf32, #blocked>
```

### 7. LLVM IR (`.llir`) → [`triton_passes/03_llvm.llir`](./triton_passes/03_llvm.llir)

**对应 Stage：`llir`**，由 Triton 的 LLVM 转换层产生

特征：
- 抽象的 MLIR 操作被降级为具体的 LLVM 指令
- NVIDIA GPU 特有操作通过 **内联 PTX 汇编**（`asm sideeffect`）表达
- 地址空间显式标注：`ptr addrspace(1)` = 全局 GPU 内存

关键内联汇编的语义：
```llvm
; 向量化 2×f32 加载 (对应 sizePerThread=2):
asm sideeffect "@$3 ld.global.v2.b32 { $0, $1 }, [ $2 + 0 ];"

; L1/L2 缓存策略 (evict_last 的底层实现):
asm sideeffect "createpolicy.fractional.L2::evict_last.b64 $0, 1.0;"
asm sideeffect "@$3 ld.global.L1::evict_last.L2::cache_hint.b32 { $0 }, [ $1 + 0 ], $2;"
```

### 8. PTX Assembly (`.ptx`) → [`triton_passes/04_assembly.ptx`](./triton_passes/04_assembly.ptx)

**对应 Stage：`ptx`**，由 LLVM NVPTX 后端生成，target `sm_120a`（Blackwell 架构）

这是最接近硬件的人类可读代码，能够被 `ptxas` 直接编译为 `.cubin` 二进制。

关键指令解析：

```ptx
; ATan 替代梯度的核心计算（完全展开）:
mul.f32    %r42, %r40, 0f40490FDB;  ; (v-1) * pi  (0x40490FDB = 3.14159274f)
fma.rn.f32 %r44, %r42, %r42, 0f3F800000; ; pi^2*x^2 + 1.0  (FMA 融合乘加)
div.full.f32 %r46, %r37, %r44;      ; 1.0 / (1 + pi^2*x^2)
fma.rn.f32 %r50, %r38, %r1, %r48;  ; grad*(1-spike) + reciprocal*chain_grad

; L2 驱逐策略指令:
createpolicy.fractional.L2::evict_last.b64 %rd8, 1.0;

; 向量化写回 (2 个 f32 同时写):
@%p1 st.global.v2.b32 [ %rd2 + 0 ], { %r21, %r22 };
```

**优化点**：`arith.divf` (TTIR) → `fma.rn.f32` + `div.full.f32` (PTX)：编译器将 `tmp8*tmp8+1.0` 优化为单条 FMA（Fused Multiply-Add）指令，减少了一次浮点舍入。

---

## 各层技术要点对照表

| 层级 | 抽象形式 | 关键组件 | 文件路径 |
|-----|---------|---------|---------|
| L1 用户代码 | Python OOP | SpikingJelly `LIFNode`, `ATan` | `spikingjelly/activation_based/neuron/lif.py:96`, `surrogate.py:792` |
| L2 FX 图 | ATen 算子 DAG | TorchDynamo + AOTAutograd | `torch/_dynamo/`, `torch/_functorch/aot_autograd.py` |
| L3 Triton Kernel | Python DSL (`tl.*`) | TorchInductor codegen + 算子融合 | `/tmp/torchinductor_charlley/fr/*.py` |
| L4 Inductor Wrapper | Python 调度脚本 | `empty_strided_cuda`, `extern_kernels`, `get_raw_stream` | `/tmp/torchinductor_charlley/md/*.py` |
| L5 TTIR | MLIR (tt dialect) | `triton/compiler/code_generator.py` `ast_to_ttir()` | `triton_passes/01_unoptimized.ttir` |
| L6 TTGIR | MLIR (ttg dialect) | Triton GPU Layout Pass | `triton_passes/02_optimized_gpu.ttgir` |
| L7 LLVM IR | LLVM IR + PTX ASM | `triton/compiler/compiler.py` LLVM stage | `triton_passes/03_llvm.llir` |
| L8 PTX | NVIDIA PTX Assembly | LLVM NVPTX Backend, `ptxas` | `triton_passes/04_assembly.ptx` |
| L9 CUBIN | GPU 二进制 | CUDA Driver | `~/.triton/cache/HASH/*.cubin` |

---

## 附加工具：运行时调用栈追踪 (`full_trace.py`)

在父目录中提供了一个额外的脚本 [`../full_trace.py`](../full_trace.py)，用于在真实执行环境中使用 `sys.settrace` 截获 `torch.compile` 到 `Triton` 编译管线触发的每一个相关函数调用。由于 `torch.compile` 会利用 JIT 编译并延迟执行大量优化操作，此脚本能够打印出每一个内部调用的确切源文件和代码行号，有助于进行更底层的调试和编译机制探索。

运行该追踪器并结合 `TORCH_LOGS="output_code"`：
```bash
TORCH_LOGS="output_code" python ../full_trace.py
```
您可以在产生的跟踪日志中观察到本目录下涉及的 TorchDynamo、AOTAutograd 以及 Inductor 的函数级活动。
