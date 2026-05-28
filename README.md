# Triton Pass Analysis & SNN Optimization

本仓库是一个针对 [Triton 编译器 (triton-lang/triton)](https://github.com/triton-lang/triton) 的深度技术解析与定制化优化项目，包含三条主线：

1. **编译原理剖析** —— 系统梳理 Triton 如何把 Python 算子逐级编译为 GPU 机器码，并对其核心优化 Pass 做源码级分析。
2. **面向 SNN 的自定义 Pass 开发** —— 在 Triton 后端搭建一个针对脉冲神经网络（SNN）推理加速的自定义 MLIR Pass。目前已完成「准备工作」（SNN 标记的端到端打通与 Pass 的条件性插入），真正的时空拆分优化仍在开发中。
3. **NIR 跨框架表达探索与对照实验** —— 评估 SpikingJelly 在 [NIR](https://neuroir.org/) 协议上的表达能力，用 NIR 重新实现同一个 VGG16-SNN，对比 eager / torch.compile+全 Triton / NIR roundtrip 三条算子后端组合的推理延迟与显存特性。

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
  - `Optimization-Insights.md` —— 时间步结构、寄存器溢出、脉冲稀疏度等关键事实核查，及对 §2.1 / §2.2 各优化思路的实测结论；
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

### 三、NIR 跨框架表达探索与对照实验

[NIR (Neuromorphic Intermediate Representation)](https://neuroir.org/) 是一种跨神经形态框架/硬件的中间表示协议。本仓库系统评估了 SpikingJelly 在 NIR 上的表达能力，并把同一个 VGG16-SNN 用 NIR 重新实现作为对照实验，揭示「`torch.compile` + 全 Triton」与「eager + cuDNN」两条算子后端组合在小批量推理场景下的性能差距。

- **NIR 表达能力评估**
  基于 SpikingJelly 自带的 `nir_exchange` 包，逐算子核查 VGG16-SNN 在 NIR 协议下的可表达性。结论：BatchNorm 必须 fold（NIR 协议无 BN 原语）、MaxPool 必须替换为 AvgPool（NIR 协议无 MaxPool 原语），Conv / Linear / Flatten / IF / LIF 等可无损映射；soft reset 在 NIR 协议下会被强制写成硬复位（v_reset=0）。

- **NIR 版 VGG16-SNN 实现 (`examples/vgg16_snn/vgg16_via_nir.py`)**
  构造 BN + AvgPool 单步网络 → `fuse_conv_bn_eval_modules` 折 BN → `export_to_nir` → `import_from_nir(device='cuda', step_mode='m')` → 在 RTX 5070 Ti 上完成端到端多步推理。

- **三种实现方式对照 (`examples/vgg16_snn/benchmark_compare.py` + [`Implementation-Modes.md`](examples/vgg16_snn/Implementation-Modes.md))**
  在同一台机器、同一份输入下对比 eager / torch.compile + 全 Triton / NIR roundtrip 三条路径，跨 `BATCH=1/32/40/50/56` 测量平均推理延迟与显存上限。三路径的算子后端归宿、`fx.GraphModule.forward` → `libcudnn.so.9` 的六层完整调用栈、按 layer 的库归属都在文档里列清。

- **SpikingJelly 兼容性补丁 ([`SpikingJelly-Triton-Patch.md`](examples/vgg16_snn/SpikingJelly-Triton-Patch.md))**
  定位并修复 SpikingJelly 手写多步 LIF / IF / PLIF Triton kernel 在本仓库 Triton fork 上的 `convert_and_store` 块指针 dtype API 不兼容（双层 `.element_ty` → 单层），修复后 SpikingJelly 的 fused-T-loop LIF Triton kernel 在三条路径上都能正常工作。本仓库已 fork SpikingJelly 仓库，相关补丁固化在 `Xarlley/spikingjelly` 的 `triton-fork-compat` 分支。

### 四、方法与经验沉淀

- **端到端追踪示例 (`examples/spikingjelly_triton/`)**
  展示 SpikingJelly 的脉冲计算逻辑如何被 `torch.compile` 动态追踪、编译，并附带各阶段 IR / PTX 的导出产物（`analysis/`）。

- **技能文档 (`Document/Skill/`)**
  沉淀通用的诊断与调优方法。当前收录：
  - `full-triton-compilation.md` —— 如何诊断并让一个 SpikingJelly SNN 完整走 Triton 编译（图中断、`recompile_limit`、extern kernel 三类问题的成因与解法）。
  - `audit-full-triton-path.md` —— 全 Triton 路径的端到端审计程序（10 个独立可观测指标 + 一段一键 grep + 一份 PASS/FAIL bash 脚本），保证「自定义 Pass 作用于整网」这条假设的可重复验证。
  - `spikingjelly-nir-implementation.md` —— SpikingJelly 里 NIR 编程模型 + 用户代码到 NIR 图的源码级映射 + NIR → fx.GraphModule → ATen → cuDNN/cuBLAS + LIF kernel 五级 IR (TTIR/TTGIR/LLIR/PTX/SASS) 的真机捕获讲解。
  - `nir-call-stack-trace.md` —— NIR 路径的 nirtorch / SpikingJelly / Triton / cuDNN+cuBLAS+ATen 真实运行时调用栈（`sys.settrace` + `torch.profiler` 双向截获），九个子节含：BATCH=56 三路冷启动 10024 样本对照、path B BN 算术在 GPU 上仍跑的 TTIR 实证、NIR-compile vs SJ-direct 的 FX 图同形性逐字段对比。
  - `nir-op-mapping.md` —— NIR v1.0.8 全部 17 个原语 ↔ PyTorch / SpikingJelly / ATen 算子的双向映射详表（含 nirtorch DEFAULT_MAP 覆盖 SJ map_dict 的反直觉行为）+ 协议级强约束清单 + SJ+nirtorch 当前能 round-trip 的 9 类原语交集。

## 📂 目录结构

```
Document/                            编译原理分析与经验文档
├── compiler.md                      编译全流程调用树（核心阅读起点）
├── CallStack.md                     torch.compile → Triton 运行时调用栈
├── SNN_Pass_Execution_Analysis.md   自定义 SNN Pass 当前行为的如实记录
├── Passes/                          19 份官方优化 Pass / 工具模块的源码级剖析
├── Skill/                           通用诊断与调优经验
│   ├── full-triton-compilation.md   让 SpikingJelly SNN 完整走 Triton 编译
│   ├── audit-full-triton-path.md    全 Triton 路径端到端审计（10 项指标 + PASS/FAIL 脚本）
│   ├── spikingjelly-nir-implementation.md  SJ 里 NIR 编程模型 + 到 GPU 的完整 IR 下降链（真实捕获）
│   ├── nir-call-stack-trace.md      NIR 路径的运行时调用栈实测 + 三路冷启动对照 + FX 图同形性证据
│   └── nir-op-mapping.md            NIR ↔ Torch ↔ ATen 算子双向映射详表 + 协议级强约束清单
└── IR-Trace/                        真实 VGG16-SNN 代表 kernel 的逐 Pass IR 变换跟踪
    ├── README.md                    方法、等价性保证与流水线总览
    ├── Optimization-Insights.md     关键事实核查（时间步 / 寄存器 / 脉冲稀疏度）与 §2.1·§2.2 结论
    ├── All-Kernels.md               一次真实推理生成的全部 48 个 Triton kernel 完整代码
    ├── Inductor-Tile-Register-Strategy.md   tile 决策与寄存器分配策略（TorchInductor 源码级）
    ├── {convolution,bn_lif,maxpool,matmul}/   每 kernel：索引 + 逐 Pass 变换文档 + 各阶段 IR
    └── nir_lif_kernel/              NIR 路径真实运行捕获（NIR 图、fx 源码、LIF kernel 五级 IR、调用栈、profiler、冷启动 10024 样本 jsonl、AOT FX 图对比）

examples/                            测试与示例脚本
├── spikingjelly_triton/             SpikingJelly → torch.compile → Triton 溯源示例
│   └── analysis/                    各阶段 IR / PTX 导出产物
├── triton_pass/
│   └── test_triton.py               SNN_FLAG 条件触发测试
└── vgg16_snn/
    ├── vgg16_test.py                可复现、全 Triton 编译的 VGG16-SNN 基准
    ├── vgg16_via_nir.py             NIR roundtrip 版 VGG16-SNN（BN-folded + AvgPool）
    ├── nir_compile_test.py          验证 NIR 返回的 fx.GraphModule 套 torch.compile 也能纯走 Triton
    ├── trace_nir_calls.py           sys.settrace + torch.profiler 抓 NIR 路径运行时调用栈
    ├── cold_start_10k_compare.py    三路径冷启动 10024 样本对照（MODE=B/NIR/SJ + BATCH 参数化）
    ├── benchmark_inference.py       ImageNet val 上 N 张样本的延迟测量（COMPILE=1 切全 Triton）
    ├── benchmark_compare.py         三条路径在同一输入上做 100-iter 平均对比（支持 BATCH 参数）
    ├── Implementation-Modes.md      三条实现路径的代码走读、cuDNN 调用栈、BATCH 调参指南
    ├── SpikingJelly-Triton-Patch.md SpikingJelly 自带 LIF Triton kernel 与本仓库 fork 的兼容性补丁
    └── test_snn_split.py            触发 SNN Pass 的小 kernel

dev-log/                             开发日志与计划
├── dev-plan.md                      开发目标与任务分解
├── dev-log.md                       定制版 Triton 构建踩坑记录
└── build_triton.sh                  定制版 Triton 的一键清理重建脚本

triton/        (Submodule)           含自定义 SNN Pass 的定制版 Triton (Xarlley fork)
spikingjelly/  (Submodule)           SpikingJelly (Xarlley fork, triton-fork-compat 分支)
pytorch/       (Submodule)           TorchInductor 源码分析所用的 PyTorch
nir/           (Submodule)           NIR 协议参考实现 (neuromorphs/NIR)
nirtorch/      (Submodule)           NIRTorch 参考实现 (neuromorphs/NIRTorch)
```

## 🚧 SNN Pass 开发进度

详见 `dev-log/dev-plan.md`。

- [x] **准备工作** —— `SNN_FLAG` 从 `@triton.jit` 到编译后端的端到端打通、Pass 的条件性插入（已通过 `test_triton.py` 验证）。
- [x] **§2.1 / §2.2 优化思路可行性核查** —— 时间拆分、空间拆分、（膜电位的）符号重物质化、块级零值检测四条思路，经源码级核查与真实训练 SNN（top-1≈46.6% 的 T=4 ImageNet VGG16-SNN）的实测级核查，**均判定不值得或不可行**：或处于错误的优化层级（应在 TorchInductor 图层而非 TritonGPU Pass），或收益不足以覆盖代价（脉冲张量虽 80% 为零，可整块跳过的工作量仅 1.6%）。完整论证见 [`Document/IR-Trace/Optimization-Insights.md`](Document/IR-Trace/Optimization-Insights.md)。

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

- **`triton/`** —— 定制版 Triton (Xarlley fork)。基于官方主线 `5d69e1cf4` 切出 `snn-optimization` 分支，所有 C++ / Python 层的改动（含自定义 Pass）均保存在该分支。
- **`spikingjelly/`** —— 定制版 SpikingJelly (Xarlley fork)。`triton-fork-compat` 分支基于上游 master，补上一处针对本仓库 Triton fork 块指针 dtype 新 API 的兼容性修补（双层 `.element_ty` → 单层，详见 [`SpikingJelly-Triton-Patch.md`](examples/vgg16_snn/SpikingJelly-Triton-Patch.md)）。
- **`pytorch/`** —— TorchInductor 源码分析所用的 PyTorch，固定在已安装运行版本对应的 commit `70d99e9`（torch 2.11.0）。用于 `Document/IR-Trace/Inductor-Tile-Register-Strategy.md` 中对 tile / 寄存器策略的源码级讲解。
- **`nir/`** —— [NIR](https://github.com/neuromorphs/NIR) 协议参考实现，供 §三 NIR 表达能力评估直接读源码。
- **`nirtorch/`** —— [NIRTorch](https://github.com/neuromorphs/NIRTorch) 参考实现，SpikingJelly 的 `nir_exchange` 通过它做 fx tracing 与 NIR ↔ PyTorch 双向重建。

克隆本仓库时加上 `--recursive`，或在克隆后执行：

```bash
git submodule update --init --recursive
```
