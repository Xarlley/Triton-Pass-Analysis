# Pass 31：TritonGPUScheduleLoops（第二次）

> kernel：矩阵乘法 / 全连接 (addmm) ｜ CLI：`tritongpu-schedule-loops` ｜ 编译流水线第 31 个 Pass

## 这个 Pass 的作用

这是 TritonGPUScheduleLoops 的第二次执行（第一次在 Pass 19）。在本阶段，它作为软件流水线展开（SoftwarePipeliner）的子 Pass 运行，负责将带有 `loop.stage` / `loop.cluster` 属性的 `scf.for` 循环转化为使用共享内存（shared memory）和异步拷贝（async copy）的流水线循环。

具体而言，Pass 31 内部执行两个子步骤，并在 after 文件中各产生一段内部 IR dump：
1. **LowerLoops**（after 文件 183～299 行）：将 `tt.load` 替换为 `ttg.async_copy_global_to_local` + `ttg.async_commit_group` + `ttg.async_wait` + `ttg.local_load` 的异步加载模式；分配共享内存缓冲区（`ttg.local_alloc`）。
2. **ExpandLoops**（after 文件 300～504 行）：展开循环的 prologue（预热阶段），将 `tt.scheduled_max_stage = 4` 对应的 4 次预取迭代（stage 0～3）手动展开到循环体之前，再保留主 loop 在 `%c4_i32` 到 `%c128_i32` 迭代范围内运行。

before 文件 179 行，after 文件 504 行（3 段 IR dump 拼接：第一段 179 行原始 IR + LowerLoops 117 行 + ExpandLoops 204 行）。

## IR 变化

**变换前（before，179 行，抽象的带属性循环）：**

```mlir
%acc = scf.for %arg3 = %c0_i32 to %c128_i32 step %c1_i32 iter_args(%arg4 = %cst_5) -> (tensor<16x32xf32, #blocked2>) : i32 {
  %a_41 = tt.load %a_40 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<16x32x!tt.ptr<f32>, #blocked>
  %b_45 = tt.load %b_44 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<32x32x!tt.ptr<f32>, #blocked1>
  %a_46 = ttg.convert_layout %a_41 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : ... -> #ttg.dot_op<{opIdx = 0, parent = #blocked2}>
  %b_47 = ttg.convert_layout %b_45 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : ... -> #ttg.dot_op<{opIdx = 1, parent = #blocked2}>
  %acc_48 = tt.dot %a_46, %b_47, %arg4 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : ...
  scf.yield %acc_48
} {tt.scheduled_max_stage = 4 : i32}
```

**变换后 LowerLoops（async copy + shared memory 版本）：**

```mlir
// 循环外：分配共享内存缓冲（深度为 4 的循环缓冲）
%35 = ttg.local_alloc : () -> !ttg.memdesc<4x16x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
%36 = ttg.local_alloc : () -> !ttg.memdesc<4x32x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>, #ttg.shared_memory, mutable>

// 循环内（stage 0 / cluster 4：异步全局→共享内存拷贝）：
%67 = ttg.memdesc_index %35[%55] ... : !ttg.memdesc<4x16x32xf32, ...> -> !ttg.memdesc<16x32xf32, ...>
%68 = ttg.async_copy_global_to_local %66, %67 {contiguity = 4 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<16x32x!tt.ptr<f32>, ...> -> <16x32xf32, ...>
%69 = ttg.async_commit_group tokens %68 {loop.cluster = 4 : i32, loop.stage = 0 : i32}
%70 = ttg.async_wait %69 {loop.cluster = 0 : i32, loop.stage = 4 : i32, num = 0 : i32}
%71 = ttg.memdesc_index %35[%58] ... : !ttg.memdesc<4x16x32xf32, ...> -> !ttg.memdesc<16x32xf32, ...>
%72 = ttg.local_load %71 token %70 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : !ttg.memdesc<16x32xf32, ...> -> tensor<16x32xf32, ...>

// 循环内（stage 4 / cluster 0：从共享内存加载并执行 MMA）：
%82 = ttg.convert_layout %72 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : ... -> #ttg.dot_op<{opIdx = 0, parent = ...}>
%84 = tt.dot %82, %83, %arg4 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : ...

// 循环后：
%38 = ttg.async_wait {num = 0 : i32}
ttg.local_dealloc %35 : !ttg.memdesc<4x16x32xf32, ...>
ttg.local_dealloc %36 : !ttg.memdesc<4x32x32xf32, ...>
```

**变换后 ExpandLoops（含 prologue 展开，部分摘录）：**

```mlir
// Prologue 第 1 次预取（K=0）
%48 = ttg.async_copy_global_to_local %45, %46 mask %47 {contiguity = 4 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32} : ...
%49 = ttg.async_commit_group tokens %48 {loop.cluster = 4 : i32, loop.stage = 0 : i32}
// Prologue 第 2 次预取（K=1）
%69 = ttg.async_copy_global_to_local %66, %67 mask %68 {contiguity = 4 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32} : ...
%70 = ttg.async_commit_group tokens %69 {loop.cluster = 4 : i32, loop.stage = 0 : i32}
// ...（Prologue 第 3、4 次预取省略）...
// 主 loop 从 K=4 开始（%c4_i32 to %c128_i32）
```

## 说明

Pass 31 是 Triton 软件流水线实现的核心阶段，将抽象的"带调度属性的循环"转化为真实的异步内存操作序列：

- **LowerLoops** 将每个 `tt.load`（同步全局内存读）替换为 `ttg.async_copy_global_to_local`（异步 DMA 拷贝）+ `ttg.async_commit_group`（提交 DMA 请求）+ `ttg.async_wait`（等待 DMA 完成）+ `ttg.local_load`（从共享内存读入寄存器）。同时，分配深度为 4 的循环缓冲区（`memdesc<4x16x32xf32, ...>`），对应 `tt.scheduled_max_stage = 4`。
- **ExpandLoops** 展开 prologue：将循环开始前的 4 次预取（K=0, 1, 2, 3）显式展开为 4 组异步拷贝提交，使主 loop 从 K=4 开始时，前 4 次迭代的数据已在飞行中（in-flight），实现完整的 5 stage 流水线（4 次预取 + 1 次使用 = stage 0~4）。
- 共享内存布局 `#ttg.swizzled_shared` 使用 swizzle 模式（`vec=1, perPhase=1, maxPhase=1`），减少 bank conflict。
- `ttg.memdesc_index %buf[%slot]` 从循环缓冲（深度=4）的第 `%slot` 槽取出对应的 shared memory descriptor。
