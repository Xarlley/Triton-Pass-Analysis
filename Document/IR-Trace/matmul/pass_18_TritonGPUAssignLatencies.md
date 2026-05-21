# Pass 18：TritonGPUAssignLatencies

> kernel：矩阵乘法 / 全连接 (addmm) ｜ CLI：`tritongpu-assign-latencies` ｜ 编译流水线第 18 个 Pass

## 这个 Pass 的作用

TritonGPUAssignLatencies 为循环中的内存访问（`tt.load`、`tt.store`）和 MMA 矩阵乘法指令标注延迟信息（`tt.latency` 属性）。这些延迟信息将被后续的 TritonGPUScheduleLoops 和 TritonGPUPipeline Pass 用于决定哪些操作需要提前发射（软件流水线），以及计算流水线的最大深度（max stage）。对于 Blackwell sm_120 架构，全局内存访问的典型延迟为 4 个 stage（与 `num_stages=5` 配合使用）。

## IR 变化

**关键变化：** 循环体内的两个 `tt.load` 各自被添加了 `{tt.latency = 4 : i32}` 属性。

**变换前（load 无延迟标注）：**

```mlir
%a_41 = tt.load %a_40 : tensor<16x32x!tt.ptr<f32>, #blocked1>
...
%b_45 = tt.load %b_44 : tensor<32x32x!tt.ptr<f32>, #blocked2>
```

**变换后（load 带延迟属性）：**

```mlir
%a_41 = tt.load %a_40 {tt.latency = 4 : i32} : tensor<16x32x!tt.ptr<f32>, #blocked1>
...
%b_45 = tt.load %b_44 {tt.latency = 4 : i32} : tensor<32x32x!tt.ptr<f32>, #blocked2>
```

IR 行数保持 185 不变（仅修改了两行 load 指令的属性，无增删）。

## 说明

`tt.latency = 4` 表示这两个全局内存 load 操作具有 4 个 stage 的延迟，即在 stage S 发出的 load，其数据要到 stage S+4 才可用。这与编译参数 `num_stages=5` 直接对应：软件流水线将循环展开 5 个 stage，其中 stage 0～3 负责预取（prefetch），stage 4 负责消费（dot 运算）。

对于这个全连接 kernel（权重矩阵 `4096×4096`，激活 `1×4096`），两个 load 分别加载：
- 激活矩阵 A 的 `16×32` 分块（`tensor<16x32x!tt.ptr<f32>, #blocked1>`）
- 权重矩阵 B 的 `32×32` 分块（`tensor<32x32x!tt.ptr<f32>, #blocked2>`）

延迟标注确保后续 TritonGPUScheduleLoops 能正确地将这两个 load 分配到 `loop.stage = 0`（最早预取），而 `tt.dot` 分配到 `loop.stage = 4`（数据就绪后执行），从而在 128 次 K 维迭代中实现内存延迟与计算的充分重叠，充分利用 Blackwell SM 的内存子系统带宽。
