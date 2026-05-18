# Triton Pass Analysis & SNN Optimization

本仓库是一个针对 [Triton 编译器 (triton-lang/triton)](https://github.com/triton-lang/triton) 的深度技术解析与定制化优化项目。它不仅提供了对于 Triton 将 Python 代码逐级编译为高效 GPU 机器码的全面剖析，还**实战演示了如何开发一个专门针对脉冲神经网络（SNN）的时空拆分 Triton Pass**。

## 🎯 核心内容

1. **核心优化 Pass 深度剖析 (`Document/Passes/*.md`)**
   对 Triton 源码中位于 `lib/Dialect/TritonGPU/Transforms` 目录下的 19 个核心优化 Pass（C++ 代码）进行了结构化的 Markdown 代码分析。涵盖了从共享内存的内存合并访存（Coalescing）、寄存器层面的 Tensor Core 降级（AccelerateMatmul），到全局的控制流融合和延迟隐藏（Prefetching / Pipelining）等编译器核心原理。
   
2. **SNN 时空拆分自定义 Pass 实践 (`Document/SNN_Pass_Execution_Analysis.md`)**
   基于真实的 VGG16 网络需求，我们在 Triton 编译器后端实现了一个定制化的 C++ MLIR Pass（即 `MyNoOpPass` 的升级版）。该 Pass 能够在编译期对 SNN 的时间步（Time Steps）和空间神经元（SM 寄存器限制）进行智能拆分优化。您可以通过阅读这份执行报告了解它的设计逻辑以及阶段性的 IR（中间表示）演变过程。

3. **编译全流程调用树 (`Document/compiler.md`)**
   分析了 Triton 编译器前端的调用总控点 `python/triton/compiler/compiler.py` 以及后端 `third_party/nvidia/backend/compiler.py`，并构建了**从上层 Python 算子到下层 LLVM IR 生成的完整代码调用关系图**，精确标注了每一个优化 Pass 发挥作用的时机。

4. **完整编译调用栈分析 (`Document/CallStack.md`)**
   通过在真实 CUDA 环境中运行 SNN 示例代码，使用 `sys.settrace` 截获了从 `torch.compile` 经由 TorchDynamo、AOTAutograd、TorchInductor 到 Triton 编译器各阶段的完整函数调用链，并标注了每个核心函数的**精确源文件位置与行号**。

5. **框架到编译器的端到端映射 (SpikingJelly 示例)**
   通过 `examples/` 下提供的基于 SpikingJelly 的 SNN 模型代码（包含最基础的追溯示例与真实的 VGG16 示例），呈现了上层框架的脉冲计算逻辑如何被动态追踪、编译，并最终触发我们自定义的 Triton GPU Kernel Pass 的全过程。

## 🔖 源码版本控制与 Submodule

为了保证所分析的代码块与实际代码的一致性和可复现性，本分析挂载了一个定制过的 Triton 代码库：

- **定制版 Triton**: 本项目包含了一个指向包含 SNN 优化 Pass 的 Git Submodule (`triton/` 目录)。
- **SNN 优化分支**: 我们对 Triton 官方主线 `5d69e1cf4` 进行了切分，所有 C++ 层的改动均保存在 `snn-optimization` 分支中。

### 获取对应的源码

克隆本仓库时，如需一并获取我们所修改的精确源码，请加上 `--recursive` 标志，或者在克隆后运行：

```bash
git submodule update --init --recursive
```

这样 `triton/` 目录下就会包含与文档内容逐行对应的 C++ 和 Python 代码实现（包含自定义 Pass）。

## 📂 目录结构

- `Document/`：包含所有关于 Pass 的 Markdown 分析报告及文档。
  - `compiler.md`：核心阅读起点，包含了总体调用流程树。
  - `CallStack.md`：从 `torch.compile` 到 Triton 优化 Pass 的完整运行时函数调用栈。
  - `SNN_Pass_Execution_Analysis.md`：本轮最新研发的 SNN 时空拆分 Pass 的执行效果详细分析。
  - `Passes/`：包含 `AccelerateMatmul.md`、`Coalesce.md` 等 19 份详细的官方优化 Pass 源码级剖析。
- `examples/`：包含了所有的测试与运行脚本。
  - `spikingjelly_triton/`：展示如何将 SpikingJelly 的计算逻辑通过 `torch.compile` 下发给 Triton 进行编译的溯源示例代码。
  - `vgg16_snn/`：**包含触发我们自定义 SNN 拆分 Pass 的 VGG16 测试套件**（`vgg16_test.py` 和 `test_snn_split.py`）。
- `dev-log/`：开发日志与任务分解计划记录（如 `dev-plan.md`）。
- `triton/`：(Submodule) 包含了自定义 SNN 优化 Pass 的 Triton 源代码。
- `spikingjelly/`：(Submodule) 用于示例和分析的 SpikingJelly 源代码。
