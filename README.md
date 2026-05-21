# Triton Pass Analysis & SNN Optimization

本仓库是一个针对 [Triton 编译器 (triton-lang/triton)](https://github.com/triton-lang/triton) 的深度技术解析与定制化优化项目，包含两条主线：

1. **编译原理剖析** —— 系统梳理 Triton 如何把 Python 算子逐级编译为 GPU 机器码，并对其核心优化 Pass 做源码级分析。
2. **面向 SNN 的自定义 Pass 开发** —— 在 Triton 后端搭建一个针对脉冲神经网络（SNN）推理加速的自定义 MLIR Pass。目前已完成「准备工作」（SNN 标记的端到端打通与 Pass 的条件性插入），真正的时空拆分优化仍在开发中。

## 🎯 核心内容

### 一、Triton 编译器原理剖析

- **核心优化 Pass 源码级剖析 (`Document/Passes/*.md`)**
  对 `lib/Dialect/TritonGPU/Transforms` 目录下的核心优化 Pass 与工具模块做了 19 份结构化分析，涵盖内存合并访存（Coalesce）、寄存器层面的 Tensor Core 降级（AccelerateMatmul）、控制流融合与延迟隐藏（FuseNestedLoops / Prefetch / Pipelining）等编译器核心原理。

- **编译全流程调用树 (`Document/compiler.md`)**
  分析前端总控点 `python/triton/compiler/compiler.py` 与后端 `third_party/nvidia/backend/compiler.py`，构建从上层 Python 算子到下层 LLVM IR 生成的完整调用关系图，并标注每个优化 Pass 发挥作用的时机。

- **完整编译调用栈 (`Document/CallStack.md`)**
  在真实 CUDA 环境运行 SNN 示例，用 `sys.settrace` 截获从 `torch.compile` 经由 TorchDynamo、AOTAutograd、TorchInductor 到 Triton 编译器各阶段的完整函数调用链，标注每个核心函数的精确源文件位置与行号。

- **真实 kernel 的逐 Pass IR 变换跟踪与优化洞察 (`Document/IR-Trace/`)**
  对真实 VGG16-SNN 推理中的 4 个代表性 Triton kernel（卷积、BN+LIF、MaxPool、矩阵乘法），逐 Pass 记录其 IR 从 TTIR → TTGIR → LLVM IR 的每一次变换（104 篇变换文档；捕获采用确定性重放，并与真实运行的 TTGIR / PTX 逐字节对账）。在此之上还有三篇专题分析：
  - `Optimization-Insights.md` —— 时间步结构、寄存器溢出等关键事实核查，及对 §2.1 的启示；
  - `All-Kernels.md` —— 一次真实推理生成的全部 48 个 Triton kernel 的完整代码；
  - `Inductor-Tile-Register-Strategy.md` —— 结合 PyTorch 源码讲解 tile 决策与寄存器分配策略。

### 二、面向 SNN 的自定义 Pass 开发

- **自定义 Pass 骨架与触发链路**
  在定制版 Triton（`triton/` 子模块）中实现了自定义 C++ MLIR Pass `MyNoOpPass`，并打通了 `SNN_FLAG`（`@triton.jit` 的 `constexpr` 参数）与 `ENABLE_SNN_PASS`（环境变量）两种条件性插入方式。**目前 `MyNoOpPass` 仅为占位实现**：它会打印各阶段 IR 并写入两个 Module 属性，真正的时间/空间拆分（见 `dev-log/dev-plan.md` 第 2.1 节）尚未实现。

- **Pass 当前行为的如实记录 (`Document/SNN_Pass_Execution_Analysis.md`)**
  如实记录 `MyNoOpPass` 当前做了什么、没做什么，以及与开发计划 §2.1 的差距。

- **SNN_FLAG 触发测试 (`examples/triton_pass/test_triton.py`)**
  以向量加法 kernel 验证条件性插入逻辑：`SNN_FLAG=True` 时触发 SNN Pass，`False` 或未声明时跳过。

- **可复现的 VGG16-SNN 基准 (`examples/vgg16_snn/`)**
  一个标准结构的 VGG16 脉冲神经网络（13 卷积 + 3 全连接，基于 SpikingJelly）。使用固定保存的随机权重与输入，保证每次推理输出逐位一致；并已配置为整个模型**完整经由 Triton 编译**（无 eager 回退、无 cuDNN/cuBLAS extern kernel）。作为后续 Pass 开发验证 IR 正确性的黄金基准。

### 三、方法与经验沉淀

- **端到端追踪示例 (`examples/spikingjelly_triton/`)**
  展示 SpikingJelly 的脉冲计算逻辑如何被 `torch.compile` 动态追踪、编译，并附带各阶段 IR / PTX 的导出产物（`analysis/`）。

- **技能文档 (`Document/Skill/`)**
  沉淀通用的诊断与调优方法。当前收录 `full-triton-compilation.md`：如何诊断并让一个 SpikingJelly SNN 完整走 Triton 编译——图中断、`recompile_limit`、extern kernel 三类问题的成因（基于 SpikingJelly 源码）与解法。

## 📂 目录结构

```
Document/                            编译原理分析与经验文档
├── compiler.md                      编译全流程调用树（核心阅读起点）
├── CallStack.md                     torch.compile → Triton 运行时调用栈
├── SNN_Pass_Execution_Analysis.md   自定义 SNN Pass 当前行为的如实记录
├── Passes/                          19 份官方优化 Pass / 工具模块的源码级剖析
├── Skill/                           通用诊断与调优经验
│   └── full-triton-compilation.md   让 SpikingJelly SNN 完整走 Triton 编译
└── IR-Trace/                        真实 VGG16-SNN 代表 kernel 的逐 Pass IR 变换跟踪
    ├── README.md                    方法、等价性保证与流水线总览
    ├── Optimization-Insights.md     关键事实核查（时间步结构 / 寄存器溢出）与 §2.1 启示
    ├── All-Kernels.md               一次真实推理生成的全部 48 个 Triton kernel 完整代码
    ├── Inductor-Tile-Register-Strategy.md   tile 决策与寄存器分配策略（TorchInductor 源码级）
    └── {convolution,bn_lif,maxpool,matmul}/   每 kernel：索引 + 逐 Pass 变换文档 + 各阶段 IR

examples/                            测试与示例脚本
├── spikingjelly_triton/             SpikingJelly → torch.compile → Triton 溯源示例
│   └── analysis/                    各阶段 IR / PTX 导出产物
├── triton_pass/
│   └── test_triton.py               SNN_FLAG 条件触发测试
└── vgg16_snn/
    ├── vgg16_test.py                可复现、全 Triton 编译的 VGG16-SNN 基准
    └── test_snn_split.py            触发 SNN Pass 的小 kernel

dev-log/                             开发日志与计划
├── dev-plan.md                      开发目标与任务分解
├── dev-log.md                       定制版 Triton 构建踩坑记录
└── build_triton.sh                  定制版 Triton 的一键清理重建脚本

triton/        (Submodule)           含自定义 SNN Pass 的定制版 Triton
spikingjelly/  (Submodule)           示例与分析所用的 SpikingJelly
pytorch/       (Submodule)           TorchInductor 源码分析所用的 PyTorch
```

## 🚧 SNN Pass 开发进度

详见 `dev-log/dev-plan.md`。

- [x] **准备工作** —— `SNN_FLAG` 从 `@triton.jit` 到编译后端的端到端打通、Pass 的条件性插入（已通过 `test_triton.py` 验证）。
- [ ] **§2.1 时间拆分与空间拆分** —— 开发中。
- [ ] **§2.2 符号重物质化** —— 待开发。

## ⚙️ 构建与运行

定制版 Triton 的完整构建步骤见 `dev-log/build_triton.sh`，相关 LLVM / CUDA 版本踩坑记录见 `dev-log/dev-log.md`。构建并安装到 conda 环境后：

```bash
# SNN_FLAG 条件触发测试（True 触发 / False 或未声明跳过）
TRITON_ALWAYS_COMPILE=1 python examples/triton_pass/test_triton.py

# 可复现的 VGG16-SNN 推理基准（输出与黄金输出逐位比对）
python examples/vgg16_snn/vgg16_test.py
```

## 🔖 源码版本控制与 Submodule

为保证文档分析与实际代码逐行对应、可复现，本仓库以 Git Submodule 形式挂载相关源码：

- **`triton/`** —— 定制版 Triton。基于官方主线 `5d69e1cf4` 切出 `snn-optimization` 分支，所有 C++ / Python 层的改动（含自定义 Pass）均保存在该分支。
- **`spikingjelly/`** —— 示例与分析所用的 SpikingJelly。
- **`pytorch/`** —— TorchInductor 源码分析所用的 PyTorch，固定在已安装运行版本对应的 commit `70d99e9`（torch 2.11.0）。用于 `Document/IR-Trace/Inductor-Tile-Register-Strategy.md` 中对 tile / 寄存器策略的源码级讲解。

克隆本仓库时加上 `--recursive`，或在克隆后执行：

```bash
git submodule update --init --recursive
```
