# Pass 36：TritonNvidiaGPURemoveTMEMTokensPass

> kernel：矩阵乘法 / 全连接 (addmm) ｜ CLI：`triton-nvidia-gpu-remove-tmem-tokens` ｜ 编译流水线第 36 个 Pass

## 这个 Pass 的作用

TritonNvidiaGPURemoveTMEMTokensPass 负责处理与 Tensor Memory（TMem）相关的 async token 依赖链，将那些没有对应 TMem 写入方的孤立 async token（即未被任何 TMem 操作产生的 token 引用）替换为 `ub.poison`（Undefined Behavior Poison）类型的占位符，以打破依赖链并为后续优化（如 Pass 37 CanonicalizerPass）消除这些值做准备。

对于本 kernel，由于没有真正的 TMem 分配，该 Pass 引入了 `%0 = ub.poison : !ttg.async.token` 作为一个哨兵（sentinel）值，表示"不存在的 TMem token"。同时，变量编号因新增 `%0` 而向后移位（`%0`→`%1`, `%1`→`%2` 等）。IR 行数从 273 行增加到 274 行（净增 1 行：新增 `ub.poison` 声明）。

## IR 变化

**关键变化：新增 `ub.poison` 声明；变量编号重新排列。**

**变换前（无 ub.poison）：**

```mlir
    %0 = arith.cmpi sge, %pid_m_16, %c0_i32 : i32
    llvm.intr.assume %0 : i1
    %1 = arith.cmpi sge, %pid_n_17, %c0_i32 : i32
    llvm.intr.assume %1 : i1
    ...
    %2 = tt.splat %out_ptr0 ...
    %3 = tt.addptr %2, %xindex_100 ...
    %4 = ttg.convert_layout %acc_89#0 ...
    tt.store %3, %4, %mask_96 ...
```

**变换后（插入 `ub.poison` 作为 TMem token 占位符）：**

```mlir
    %0 = ub.poison : !ttg.async.token   ← 新增：TMem token 占位符
    %1 = arith.cmpi sge, %pid_m_16, %c0_i32 : i32   ← 编号从 %0 移至 %1
    llvm.intr.assume %1 : i1
    %2 = arith.cmpi sge, %pid_n_17, %c0_i32 : i32   ← 编号从 %1 移至 %2
    llvm.intr.assume %2 : i1
    ...
    %3 = tt.splat %out_ptr0 ...
    %4 = tt.addptr %3, %xindex_100 ...
    %5 = ttg.convert_layout %acc_89#0 ...
    tt.store %4, %5, %mask_96 ...
```

before 文件 273 行，after 文件 274 行（净增 1 行）。

## 说明

`ub.poison` 是 MLIR 的 undefined behavior 类型操作，表示"该值在语义上未定义，可以被任意优化消除"。在这里，它被用作不存在的 TMem async token 的占位符，以便：
1. 保持 SSA 值的合法性（所有 `!ttg.async.token` 类型的值必须有明确的定义）。
2. 在后续 CanonicalizerPass（Pass 37）中，通过标准化规则（canonicalization pattern）将使用了 `ub.poison` 的操作（如 `ttg.async_wait %0`，其中 `%0` 是 poison）消除或简化。

对于真正使用 TMem 的 kernel（`num_warps≥4`），该 Pass 会将 TMem 的写 token 注入到相应的 async 依赖链中，确保 TMem 写完成后才开始 MMA 计算。对于本 kernel，该 Pass 的实际效果仅为添加了一个哨兵 poison 值，由后续 Canonicalizer 清理。
