# Pass 32：TritonGPUPipeline

> kernel：矩阵乘法 / 全连接 (addmm) ｜ CLI：`tritongpu-pipeline` ｜ 编译流水线第 32 个 Pass

## 这个 Pass 的作用

TritonGPUPipeline 是软件流水线的最终展开 Pass，负责将 Pass 31（TritonGPUScheduleLoops 第二次执行）产生的"带 LowerLoops/ExpandLoops 中间 dump 的多段 IR"规范化为单段、完整的流水线 IR。该 Pass 完成以下工作：

1. **消除多余的 IR dump 段**（before 504 行 → after 273 行）：将 Pass 31 after 文件中的三段 IR 合并为最终单段。
2. **规范化变量命名**：将 LowerLoops 阶段的匿名变量（`%37`, `%38`...）替换为带语义名称的变量（`%acc_89`, `%a_33`, `%b_34`...）。
3. **添加共享内存布局别名**：引入 `#shared`、`#shared1`、`#smem` 别名以简化 IR 中的 swizzled_shared 类型写法。
4. **固化 prologue 常量**：将 prologue 中使用的 K 偏移量（32, 64, 96）直接嵌为常量（`dense<32>`, `dense<64>`, `dense<96>`），并使用整数常量 `%acc=3`, `%acc_2=2` 作为共享内存槽索引。
5. **循环迭代范围调整**：主 loop 从 `%c0_i32 to %c128_i32`（Pass 31 LowerLoops）变为 `%c0_i32 to %c128_i32`（边界条件由 `%acc_116 = arith.cmpi slt, %acc_101, %c124_i32` 控制尾部 mask）。

before 文件 504 行，after 文件 273 行，IR 功能完全等价，仅为格式规范化。

## IR 变化

**变换前（before，Pass 31 的三段 dump 拼接结构）：**

```
// 第一段（1-182 行）：原始带属性的抽象循环
%acc = scf.for ... { tt.load ... tt.dot ... } {tt.scheduled_max_stage = 4}

// 第二段（183-299 行）：SoftwarePipeliner LowerLoops
%35 = ttg.local_alloc : () -> !ttg.memdesc<4x16x32xf32, swizzled_shared, ...>
scf.for { ttg.async_copy_global_to_local ... ttg.async_commit_group ... ttg.async_wait ... ttg.local_load ... tt.dot ... }

// 第三段（300-504 行）：SoftwarePipeliner ExpandLoops（含 prologue 展开）
// prologue：4 次预取（K=0, 1, 2, 3）手动展开
// 主 loop：从 K=4 到 K=127 运行
```

**变换后（after，273 行，完整规范化流水线 IR）：**

```mlir
// 新布局别名
#shared  = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem    = #ttg.shared_memory

// Prologue 常量（固化的 K 偏移：32、64、96）
%b_k_idx_vals   = arith.constant dense<96> : tensor<32x1xi32, #blocked>
%a_k_idx_vals   = arith.constant dense<96> : tensor<1x32xi32, #blocked1>
%acc            = arith.constant 3 : i32    ← 共享内存槽索引（最后一个 prologue 预取槽）
%b_k_idx_vals_0 = arith.constant dense<64> : tensor<32x1xi32, #blocked>
%a_k_idx_vals_1 = arith.constant dense<64> : tensor<1x32xi32, #blocked1>
%acc_2          = arith.constant 2 : i32
%b_k_idx_vals_3 = arith.constant dense<32> : tensor<32x1xi32, #blocked>
%a_k_idx_vals_4 = arith.constant dense<32> : tensor<1x32xi32, #blocked1>

// 共享内存分配
%a_33 = ttg.local_alloc : () -> !ttg.memdesc<4x16x32xf32, #shared,  #smem, mutable>
%b_34 = ttg.local_alloc : () -> !ttg.memdesc<4x32x32xf32, #shared1, #smem, mutable>

// Prologue：4 次预取（K=0, 1, 2, 3）——每次：地址计算 + async_copy + commit_group
%a_39 = ttg.async_copy_global_to_local %a_37, %a_38 mask %cst_5 {contiguity = 4 : i32} : ...
%a_40 = ttg.async_commit_group tokens %a_39
...（K=1, 2, 3 的 prologue 预取省略）...

// 主 loop（128 次迭代，但含 mask 控制尾部）
%acc_89:15 = scf.for %acc_101 = %c0_i32 to %c128_i32 step %c1_i32
    iter_args(%arg4 = %cst_13,      // dot 累加器
              %acc_102 = %acc,      // 写槽索引（循环 mod 4）
              %acc_103 = %acc_6,    // 读槽索引
              %acc_104 = %c4_i32,   // 缓冲深度
              ...                   // 4 组 async token（A）
              %a_108 = %a_40, %a_109 = %a_54, %a_110 = %a_68, %a_111 = %a_82,
              ...                   // 4 组 async token（B）
              %b_112 = %b_46, %b_113 = %b_60, %b_114 = %b_74, %b_115 = %b_88) : i32 {
    // 读等待（num=6 表示等待 flight 中最多 6 个组）
    %a_120 = ttg.async_wait %a_108, %b_112 {num = 6 : i32}
    // 从共享内存读 A、B 到寄存器
    %a_122 = ttg.local_load %a_121 token %a_120 : !ttg.memdesc<16x32xf32, ...> -> tensor<16x32xf32, #blocked1>
    %b_124 = ttg.local_load %b_123 token %a_120 : !ttg.memdesc<32x32xf32, ...> -> tensor<32x32xf32, #blocked>
    // 布局转换 + MMA
    %a_125 = ttg.convert_layout %a_122 : ... -> #ttg.dot_op<{opIdx = 0, parent = #blocked2}>
    %b_126 = ttg.convert_layout %b_124 : ... -> #ttg.dot_op<{opIdx = 1, parent = #blocked2}>
    %acc_127 = tt.dot %a_125, %b_126, %arg4 : ... -> tensor<16x32xf32, #blocked2>
    // 写入下一次预取到共享内存（带 mask 控制是否为最后几次迭代）
    %acc_116 = arith.cmpi slt, %acc_101, %c124_i32    ← 控制尾部 4 次迭代不写入
    %a_142 = ttg.async_copy_global_to_local %a_139, %a_140 mask %acc_141 {contiguity = 4 : i32} : ...
    scf.yield %acc_127, ... (更新所有 iter_args)
}
// Epilogue
%acc_90 = ttg.async_wait {num = 0 : i32}
ttg.local_dealloc %b_34
ttg.local_dealloc %a_33
```

## 说明

Pass 32 生成的 IR 是完整的 5-stage 软件流水线结构，对应 K=4096、step=32、128 次迭代、`num_stages=5`：

- **Prologue（4 次手动展开）**：在循环开始前，依次发射 K=0, 1, 2, 3 的异步 DMA 拷贝，将 A 和 B 的第 0～3 分块分别写入共享内存缓冲的槽 [0], [1], [2], [3]。这 4 次预取在主 loop 开始时已"在途（in-flight）"。
- **主 loop（128 次迭代）**：每次迭代同时：
  1. 等待最早发出的 DMA 完成（`ttg.async_wait ... {num = 6}`），从共享内存读取当前 K 块数据。
  2. 执行 `tt.dot`（MMA），将读到的数据加入累加器 `%arg4`。
  3. 异步发射下一次（K+4）的 DMA 预取，写入下一个空闲槽（mod 4 循环）。
  - 尾部 4 次迭代（`%acc_101 >= 124`）不发射新预取，由 mask 控制（`%acc_116 = cmpi slt, %acc_101, 124`）。
- **Epilogue**：等待所有在途 DMA 完成（`ttg.async_wait {num = 0}`），释放共享内存。
- **循环缓冲深度 4**：共享内存 `memdesc<4x16x32xf32, ...>` 和 `memdesc<4x32x32xf32, ...>` 各有 4 个槽，对应流水线的 4 级预取深度（`tt.scheduled_max_stage = 4`）。
- `iter_args` 有 15 个参数（1 个累加器 + 2 个槽索引 + 4 个循环计数 + 8 个 async token），是该流水线状态的完整快照。

经过 Pass 32，IR 已从高层抽象（`tt.load` + `tt.dot`）完全展开为真实的 PTX 一级异步内存操作序列，为后续的 LLVM lowering 做好准备。
