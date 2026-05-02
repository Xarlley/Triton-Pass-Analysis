# Triton Pass Analysis

本仓库是一个针对 [Triton 编译器 (triton-lang/triton)](https://github.com/triton-lang/triton) 核心 GPU 优化代码的深度技术解析项目。它提供了对于 Triton 将 Python 代码逐级编译为高效 GPU 机器码的全面剖析。

## 🎯 分析内容

1. **核心优化 Pass 深度剖析 (`Document/*.md`)**
   对 Triton 源码中位于 `lib/Dialect/TritonGPU/Transforms` 目录下的 19 个核心优化 Pass（C++ 代码）进行了结构化的 Markdown 代码分析。涵盖了从共享内存的内存合并访存（Coalescing）、寄存器层面的 Tensor Core 降级（AccelerateMatmul），到全局的控制流融合和延迟隐藏（Prefetching / Pipelining）等编译器核心原理。
   
2. **编译全流程调用树 (`Document/compiler.md`)**
   分析了 Triton 编译器前端的调用总控点 `python/triton/compiler/compiler.py` 以及后端 `third_party/nvidia/backend/compiler.py`，并构建了**从上层 Python 算子到下层 LLVM IR 生成的完整代码调用关系图**，精确标注了每一个优化 Pass 发挥作用的时机。

## 🔖 源码版本控制

为了保证所分析的代码块与实际代码的一致性和可复现性，本分析**精确锁定**在 Triton 代码库的特定版本（Commit）。

- **目标分析版本 (Commit Hash)**: `5d69e1cf4d99a2bc518d4082ad14eb40d2732597`
- **提交时间**: 2026-05-01
- **Submodule 绑定**: 本仓库已包含一个指向原 `triton` 代码库特定版本的 Git Submodule (`triton/` 目录)。

### 获取对应的源码

克隆本仓库时，如需一并获取我们所分析的精确源码，请加上 `--recursive` 标志，或者在克隆后运行：

```bash
git submodule update --init --recursive
```

这样 `triton/` 目录下就会包含与文档内容逐行对应的 C++ 和 Python 代码实现，您可以随时与本目录下的 `Document/*.md` 进行对照阅读。

## 📂 目录结构

- `Document/`：包含所有关于 Pass 的 Markdown 分析报告。
  - `compiler.md`：核心阅读起点，包含了总体调用流程树。
  - `AccelerateMatmul.md`、`Coalesce.md`、`RemoveLayoutConversions.md` 等 19 份详细的优化 Pass 源码级剖析。
- `triton/`：(Submodule) 所分析的原始 Triton 项目源代码。
