# Pass 35：TritonGPUHoistTMEMAlloc

> kernel：矩阵乘法 / 全连接 (addmm) ｜ CLI：`tritongpu-hoist-tmem-alloc` ｜ 编译流水线第 35 个 Pass

## 这个 Pass 的作用

TritonGPUHoistTMEMAlloc 负责将 Tensor Memory（TMem）分配操作提升到循环外，并优化 `scf.for` 的 `iter_args` 参数列表，去除其中不需要作为循环携带值（loop-carried value）的常量 `i32` 参数。

对于本 kernel，虽然没有实际的 TMem 分配（因 `num_warps=2` 不满足 NVWS 要求），但该 Pass 仍然成功地将 `scf.for` 的 `iter_args` 从 **15 个参数**精简为 **11 个参数**：移除了 4 个多余的循环携带 `i32` 常量（`%acc_104 = %c4_i32`, `%acc_105 = %c4_i32`, `%acc_106 = %c4_i32`, `%acc_107 = %c4_i32`），这些值在循环中不发生变化，始终等于常量 `4`（共享内存缓冲深度）。

before 文件 273 行，after 文件 273 行（行数不变，但循环签名和内部变量编号发生变化）。

## IR 变化

**关键变化：`scf.for` 的 `iter_args` 从 15 个减少到 11 个。**

**变换前（iter_args 包含 4 个冗余的 `%c4_i32` 常量）：**

```mlir
%acc_89:15 = scf.for %acc_101 = %c0_i32 to %c128_i32 step %c1_i32
    iter_args(%arg4 = %cst_13,      // dot 累加器
              %acc_102 = %acc,      // 写槽索引
              %acc_103 = %acc_6,    // 读槽索引
              %acc_104 = %c4_i32,   // ← 冗余：缓冲深度常量 4
              %acc_105 = %c4_i32,   // ← 冗余：缓冲深度常量 4
              %acc_106 = %c4_i32,   // ← 冗余：缓冲深度常量 4
              %acc_107 = %c4_i32,   // ← 冗余：缓冲深度常量 4
              %a_108 = %a_40, %a_109 = %a_54, %a_110 = %a_68, %a_111 = %a_82,   // A async tokens
              %b_112 = %b_46, %b_113 = %b_60, %b_114 = %b_74, %b_115 = %b_88)  // B async tokens
    -> (tensor<16x32xf32, #blocked2>, i32, i32, i32, i32, i32, i32,
        !ttg.async.token x4, !ttg.async.token x4)  : i32 {
  // 循环内使用 %acc_104 作为"缓冲深度"比较基准
  %acc_118 = arith.cmpi sge, %acc_117, %acc_104 : i32   ← 引用 %acc_104（= 常量 4）
  ...
  scf.yield %acc_127, %acc_130, %acc_119, %acc_105, %acc_106, %acc_107, %c4_i32, ...
```

**变换后（冗余 `i32` 参数移除，循环体内直接使用 `%c4_i32`）：**

```mlir
%acc_89:11 = scf.for %acc_101 = %c0_i32 to %c128_i32 step %c1_i32
    iter_args(%arg4 = %cst_13,      // dot 累加器
              %acc_102 = %acc,      // 写槽索引
              %acc_103 = %acc_6,    // 读槽索引
              %a_104 = %a_40, %a_105 = %a_54, %a_106 = %a_68, %a_107 = %a_82,   // A async tokens
              %b_108 = %b_46, %b_109 = %b_60, %b_110 = %b_74, %b_111 = %b_88)  // B async tokens
    -> (tensor<16x32xf32, #blocked2>, i32, i32,
        !ttg.async.token x4, !ttg.async.token x4)  : i32 {
  // 循环内直接引用外部常量 %c4_i32（不再通过 iter_args 传递）
  %acc_114 = arith.cmpi sge, %acc_113, %c4_i32 : i32   ← 直接使用外部常量
  ...
  scf.yield %acc_123, %acc_126, %acc_115, %a_105, %a_106, %a_107, %a_139, ...
```

## 说明

Pass 35 分析 `scf.for` 的每个 `iter_args`，发现初始值为 `%c4_i32`（常量 4）且在 yield 时仍然 yield `%c4_i32`（即不更新）的参数，将其识别为循环不变常量并从 `iter_args` 中移除。移除后，循环体内原来引用这些 `iter_args` 的地方改为直接引用外部常量 `%c4_i32`。

具体被移除的 4 个参数（`%acc_104`～`%acc_107`）均是"最大缓冲槽索引"比较时使用的常量 4（共享内存循环缓冲深度为 4），用于 `arith.cmpi sge, slot_idx, 4` 的环形缓冲越界判断。由于这 4 个值在循环的任何迭代中都等于 4，不需要通过 `iter_args` 传递，Pass 35 将其替换为直接引用 `%c4_i32`。

`iter_args` 参数数量从 15 减少到 11，减少了 SSA 值传递开销（在 PTX 级别相当于减少了 4 个寄存器的传递）。这是软件流水线后的一次简单但重要的清理优化。
