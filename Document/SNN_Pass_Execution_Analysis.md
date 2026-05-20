# SNN 自定义 Pass（MyNoOpPass）当前行为说明

> 本文档如实记录自定义 SNN Pass 的**当前**状态。`dev-log/dev-plan.md` 第 2.1 节
> 所规划的真正的**时间拆分**与**空间拆分**目前**尚未实现**，本文不对尚未存在的
> 优化效果做任何分析。

## 1. 现状概述

`MyNoOpPass`（定义于 `triton/lib/Dialect/TritonGPU/Transforms/MyNoOpPass.cpp`）
目前是 SNN 优化 Pass 的**开发骨架 / 占位实现**。当前已完成的只有「准备工作」——
即把一个 SNN 标记从 `@triton.jit` 一路传递到编译后端，并据此条件性地插入本 Pass。
Pass 本身**不做任何真正的 SNN 优化**。

## 2. 已打通并验证的部分：SNN Pass 的条件性插入

触发方式有两种（任一满足即插入 `MyNoOpPass`）：

- **`SNN_FLAG`**：在 `@triton.jit` kernel 中声明 `SNN_FLAG: tl.constexpr` 并传入
  `True`。`python/triton/compiler/compiler.py` 会提取该值写入 `metadata["snn_flag"]`。
- **`ENABLE_SNN_PASS=1`**：环境变量开关，用于兼容 PyTorch Inductor 自动生成、
  不含 `SNN_FLAG` 参数的 kernel。

在 `third_party/nvidia/backend/compiler.py` 的 `make_ttgir` 末尾，满足条件时调用
`passes.ttgpuir.add_tritongpu_my_no_op(pm)`，否则跳过。运行
`examples/triton_pass/test_triton.py` 可观察到：`SNN_FLAG=True` 触发本 Pass、
`False` 或未声明时跳过。这条链路已验证可用，属于 dev-plan「准备工作」的范围。

## 3. MyNoOpPass 当前的实际行为

`runOnOperation()` 作用于整个 `ModuleOp`，仅做三件事：

1. 打印进入时的 Module IR；
2. 写入自定义属性 `ttg.snn_time_split = "T0-1, T2-3"`；
3. 写入属性 `ttg.maxnreg = 64`。

对这两个属性需要如实说明：

- **`ttg.snn_time_split`**：这是一个**自定义字符串属性**，**没有任何下游 Pass
  或调度器会读取它**。它只是一个惰性占位标记，不产生任何优化效果，更没有把
  时间步真正切分开。
- **`ttg.maxnreg = 64`**：这是 Triton 的**原生属性**，后端确实会据此限制单线程
  寄存器用量。但此处的 `64` 是**硬编码常量**，并非 dev-plan §2.1 所要求的
  「计算出的、保证不溢出的最大空间分块尺寸」，本 Pass 也不会按 warp 做任何分块。

## 4. 与 dev-plan §2.1 的差距

| §2.1 计划 | 当前实现 |
|---|---|
| **时间拆分**：在数据量/稀疏度显著变化处（如 maxpooling 前后）按用户显式输入切分时间步 | 未实现。仅写入一个无人消费的字符串属性。 |
| **空间拆分**：将寄存器约束硬编码进编译流程，计算出不溢出的最大分块尺寸并分配到各 warp | 未实现。仅硬编码 `ttg.maxnreg=64`，无任何计算，也无按 warp 的分块。 |

## 5. 关于「结果一致性」

由于 `MyNoOpPass` 只写入 Module 级属性、**不改写任何计算 IR**，启用与否都不会
改变推理结果——这是因为它**几乎什么都没做**，而不是因为实现了某种「保持语义
等价的拆分」。`examples/vgg16_snn/vgg16_test.py` 在启用本 Pass 后仍能正常完成
前向、输出形状 `[4, 1, 10]`，这**只能**说明插入链路不会破坏编译，**不能**作为
任何优化效果的证据。

## 6. 后续工作

真正的时间拆分与空间拆分见 `dev-log/dev-plan.md` 第 2.1 节，属于待实现项。
