# Pass 37：CanonicalizerPass

> kernel：BatchNorm + LIF 脉冲神经元 ｜ CLI：`canonicalize` ｜ 编译流水线第 37 个 Pass

## 这个 Pass 的作用

CanonicalizerPass 是 MLIR 的标准规范化 Pass，它对 IR 中每个操作应用其内置的规范化规则（canonicalization patterns），将 IR 化简为"标准形式"。典型规则包括：死代码消除（未使用的定义）、常量折叠、冗余操作消除、操作化简（如 `x + 0 → x`）等。在此位置，它主要消除 Pass 36 插入的悬空 `ub.poison` 值。

## IR 变化

本次变换行数从 234 行降至 233 行，精确地消除了 Pass 36 新增的那一行：

**变换前（234 行，含 poison 值和偏移编号的操作序列）：**

```mlir
  tt.func public @triton_poi_fused_...(...) {
    %0 = ub.poison : !ttg.async.token
    ...
    %1 = tt.splat %in_out_ptr0 : !tt.ptr<f32> -> tensor<16x64x!tt.ptr<f32>, #blocked1>
    %2 = tt.addptr %1, %tmp0_28 : ...
    tt.store %2, %tmp32, %tmp0_31 : ...
    %3 = arith.muli %xindex_24, %cst : tensor<1x64xi32, #blocked>
    %4 = tt.broadcast %yindex_16 : ...
    %5 = tt.broadcast %3 : ...
    %6 = arith.addi %4, %5 : ...
    %7 = tt.splat %out_ptr0 : ...
    %8 = tt.addptr %7, %6 : ...
    %9 = ttg.convert_layout %tmp43 : tensor<16x64xf32, #blocked1> -> tensor<16x64xf32, #blocked>
    tt.store %8, %9, %tmp0_32 : ...
```

**变换后（233 行，poison 被移除，操作编号重新从 %0 开始）：**

```mlir
  tt.func public @triton_poi_fused_...(...) {
    ...
    %0 = tt.splat %in_out_ptr0 : !tt.ptr<f32> -> tensor<16x64x!tt.ptr<f32>, #blocked1>
    %1 = tt.addptr %0, %tmp0_28 : ...
    tt.store %1, %tmp32, %tmp0_31 : ...
    %2 = arith.muli %xindex_24, %cst : tensor<1x64xi32, #blocked>
    %3 = tt.broadcast %yindex_16 : ...
    %4 = tt.broadcast %2 : ...
    %5 = arith.addi %3, %4 : ...
    %6 = tt.splat %out_ptr0 : ...
    %7 = tt.addptr %6, %5 : ...
    %8 = ttg.convert_layout %tmp43 : tensor<16x64xf32, #blocked1> -> tensor<16x64xf32, #blocked>
    tt.store %7, %8, %tmp0_32 : ...
```

变化精确到 1 行：`%0 = ub.poison : !ttg.async.token` 被删除，后续 SSA 值编号整体减 1（`%1` → `%0`，`%2` → `%1`，以此类推）。

## 说明

这是一次极其精确的单行死代码消除：`ub.poison` 值 `%0` 没有任何使用者，规范化规则直接将其标记为死代码并删除。

值得注意的是，CanonicalizerPass 在整个 73 Pass 流水线中出现多次（Pass 9、13、37、40、54、69），每次都在特定 Pass 修改 IR 后执行清理。在本 Pass 37 的位置，它清理了 NVWS/TMem 相关 Pass（Pass 20-36）留下的最后一个残留——那个悬空的 async token poison 值。

经过 Pass 37，BN+LIF kernel 的 TTGIR 达到了一个稳定的"最终 GPU IR"形式：
- 2 种 blocked 布局（读取路径 `#blocked1` 和写入路径 `#blocked`）
- 4 条 coalesced load（加载 4 个时间步数据）
- 完整的 LIF 计算链（4 个独立的充电-发放-reset 循环）
- 2 条 store（写回膜电位 `in_out_ptr0` 和输出脉冲 `out_ptr0`）

后续从 Pass 38 到 Pass 62 的大多数 Pass 对本 kernel 均为 no-op，直到 Pass 63 的 LLVM 降级。
