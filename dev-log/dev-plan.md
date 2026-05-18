# 开发目标

## 1 目的

基于现有的 demo Triton Pass，开发出一个完整的 Triton Pass。这个 Triton Pass 面向脉冲神经网络，加速脉冲神经网络的推理。

## 2 开发计划

准备工作：
- [x] 实现使用@triton.jit 传给 Triton 一个 SNN_Flag，用于标记 SNN 代码区域，然后对 SNN 代码区域执行本优化 PASS。

开发记录：
- 为了实现准备工作要求的 SNN_Flag，已经修改了 `triton/python/triton/compiler`（SNN_FLAG 提取），`triton/third_party/nvidia/backend`（SNN Pass 条件性插入），`examples/triton_pass`（带有 flag 的 SNN 向量加法 kernel 和必要的测试代码）。
- 通过执行 `test_triton.py`，已验证 SNN_Flag 提取和传递逻辑的正确性（SNN_Flag=True时触发，False或未定义时跳过）。准备工作已全部完成。


### 2.1 时间拆分与空间拆分

时间拆分：拆分位置例如maxpooling前后的数据量可能发生显著变化的地方，以及实测脉冲稀疏度明显变化的地方（由用户显式传入）。

空间拆分：将寄存器的总量约束直接硬编码到编译流程中。计算出不会发生任何寄存器溢出的最大空间分块尺寸，分配到各warp。

### 2.2 符号重物质化

符号重物质化包括两个部分：
1. 符号重物质化来表示某些中间膜电位，实现中间变量的活跃范围缩减。
2. 引入基于块的零值检测操作。