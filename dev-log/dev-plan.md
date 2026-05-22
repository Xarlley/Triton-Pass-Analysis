# 开发目标

## 1 目的

基于现有的 demo Triton Pass，开发出一个完整的 Triton Pass。这个 Triton Pass 面向脉冲神经网络，加速脉冲神经网络的推理。

## 2 开发计划

准备工作：
- [x] 实现使用@triton.jit 传给 Triton 一个 SNN_Flag，用于标记 SNN 代码区域，然后对 SNN 代码区域执行本优化 PASS。

开发记录：
- 为了实现准备工作要求的 SNN_Flag，已经修改了 `triton/python/triton/compiler`（SNN_FLAG 提取），`triton/third_party/nvidia/backend`（SNN Pass 条件性插入），`examples/triton_pass`（带有 flag 的 SNN 向量加法 kernel 和必要的测试代码）。
- 通过执行 `test_triton.py`，已验证 SNN_Flag 提取和传递逻辑的正确性（SNN_Flag=True时触发，False或未定义时跳过）。准备工作已全部完成。


> **核查结论**：§2.1、§2.2 共四条优化思路经源码级与真实训练 SNN 的实测级核查，
> **均判定不予实现**——逐条标注于下。完整论证见
> [`Document/IR-Trace/Optimization-Insights.md`](../Document/IR-Trace/Optimization-Insights.md)。

### 2.1 时间拆分与空间拆分

时间拆分：拆分位置例如maxpooling前后的数据量可能发生显著变化的地方，以及实测脉冲稀疏度明显变化的地方（由用户显式传入）。

> ❌ **判定不可行——错误的优化层级**：进入 Triton 后时间步循环已被 TorchDynamo 展开、
> `seq_to_ann_forward` 又把时间维与 batch 维合并，IR 中既无循环、也无法精确界定时间步；
> 「对不同时间块单独应用编译策略」要求各时间块是独立 kernel，而 kernel 边界由
> TorchInductor 决定，够不到 TritonGPU Pass 这一层。详见
> [Optimization-Insights.md](../Document/IR-Trace/Optimization-Insights.md) 第一、三部分。

空间拆分：将寄存器的总量约束直接硬编码到编译流程中。计算出不会发生任何寄存器溢出的最大空间分块尺寸，分配到各warp。

> ❌ **判定不可行——错误的优化层级**：tile 尺寸由 Inductor 模板的自动调优决定；寄存器
> 分配与溢出发生在 ptxas（全部 73 个 Pass 之后）。TritonGPU Pass 既改不了 tile、也管
> 不到 ptxas 的寄存器分配。详见
> [Optimization-Insights.md](../Document/IR-Trace/Optimization-Insights.md) 第三部分。

### 2.2 符号重物质化

符号重物质化包括两个部分：
1. 符号重物质化来表示某些中间膜电位，实现中间变量的活跃范围缩减。

> ❌ **判定不必要**：膜电位本身不是寄存器问题——Inductor 的逐层拆分 + LIF 递推的短活跃
> 区间已把它自然控制在几十个寄存器内；真实的寄存器溢出只发生在卷积（矩阵乘模板的
> 寄存器压力 / occupancy 权衡），与膜电位无关。详见
> [Optimization-Insights.md](../Document/IR-Trace/Optimization-Insights.md) 第二、三部分。

2. 引入基于块的零值检测操作。

> ❌ **判定易致负优化**：实测真实训练的 T=4 ImageNet SNN（top-1≈46.6%），脉冲张量虽
> 80% 为零，但可整块跳过的归约块仅 1.6%；零值检测的代价（每块一次「是否全零」判定 +
> 数据依赖分支）摊在 100% 的块上、收益却不足 1.6%，在稠密 GPU 上几乎必然是负优化。
> 详见 [Optimization-Insights.md](../Document/IR-Trace/Optimization-Insights.md) 第四部分。