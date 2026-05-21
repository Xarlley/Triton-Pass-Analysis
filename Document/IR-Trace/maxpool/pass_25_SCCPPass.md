# Pass 25：SCCPPass

> kernel：MaxPool + BN + LIF 融合 ｜ CLI：`sccp` ｜ 编译流水线第 25 个 Pass

## 这个 Pass 的作用

SCCPPass（Sparse Conditional Constant Propagation，稀疏条件常量传播）是 MLIR 标准优化 Pass，负责传播已知常量值、消除死代码，并在常量折叠后对操作数进行规范化。在本编译流水线中，此 Pass 在 NVWS warp 专化分析之后运行，主要作用是规范化常量声明顺序和变量命名，为后续 CSE（公共子表达式消除）准备更整洁的 IR。

## IR 变化

主要变化是常量声明的**重新规范化**：原先分散命名的常量（带语义名称如 `%tmp5`、`%tmp3`、`%cst`、`%x2`、`%x1` 等）被统一重排为以 `%cst_N` 编号、按值从小到大顺序排列，并移除了与旧 loc 关联的语义命名。

**变化前**（before，带语义名称和原始 loc）：

```mlir
%tmp5 = arith.constant dense<14400> : tensor<512xi32, #blocked> loc(#loc35)
%tmp3 = arith.constant dense<14336> : tensor<512xi32, #blocked> loc(#loc36)
%cst = arith.constant dense<28672> : tensor<512xi32, #blocked> loc(#loc1)
%cst_0 = arith.constant dense<128> : tensor<512xi32, #blocked> loc(#loc1)
%x2 = arith.constant dense<7168> : tensor<512xi32, #blocked> loc(#loc37)
%x1 = arith.constant dense<112> : tensor<512xi32, #blocked> loc(#loc38)
%cst_1 = arith.constant dense<64> : tensor<512xi32, #blocked> loc(#loc1)
%c512_i32 = arith.constant 512 : i32 loc(#loc1)
```

**变化后**（after，统一规范命名、按值排序）：

```mlir
%cst = arith.constant dense<64> : tensor<512xi32, #blocked> loc(#loc1)
%cst_0 = arith.constant dense<112> : tensor<512xi32, #blocked> loc(#loc1)
%cst_1 = arith.constant dense<7168> : tensor<512xi32, #blocked> loc(#loc1)
%cst_2 = arith.constant dense<128> : tensor<512xi32, #blocked> loc(#loc1)
%cst_3 = arith.constant dense<28672> : tensor<512xi32, #blocked> loc(#loc1)
%cst_4 = arith.constant dense<14336> : tensor<512xi32, #blocked> loc(#loc1)
%cst_5 = arith.constant dense<14400> : tensor<512xi32, #blocked> loc(#loc1)
%c512_i32 = arith.constant 512 : i32 loc(#loc1)
```

随之，所有引用这些常量的操作数也更新为新名称（例如 `%x0 = arith.remsi %xindex_4, %cst_1` 变为 `%x0 = arith.remsi %xindex_8, %cst`）。变量编号同步重排（如 `%xindex_4` → `%xindex_8`）。before 和 after 的 IR 行数均为 271（含诊断 dump），但有效逻辑内容行数相同。

## 说明

SCCP 在此 kernel 上未做任何真正的"常量折叠"（因为所有运算依赖运行期的 `tt.get_program_id` 和 `tt.make_range`，无法静态求值）。实际发生的是：常量池的**规范化重排**——SCCP 将所有常量的 loc 属性统一归一（全部指向 `#loc1 = loc(unknown)`），并以数值大小重新排序。这为后续 CSE Pass 提供了便利：CSE 通过值相等性消除重复操作，统一的命名和排序让相同值的常量更容易被识别和合并。

对于本 MaxPool+BN+LIF kernel，7 个整数常量（64、112、128、7168、14336、14400、28672）分别对应 MaxPool 2×2 窗口在展平张量中的步长关系（channel=64，width=112，channel×width=7168，…），SCCP 规范化后这些常量的逻辑映射不变，只是编译器内部表示更整洁。
