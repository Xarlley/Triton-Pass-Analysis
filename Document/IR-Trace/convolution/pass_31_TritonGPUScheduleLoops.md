# Pass 31：TritonGPUScheduleLoops

> kernel：卷积 (Convolution) ｜ CLI：`tritongpu-schedule-loops` ｜ 编译流水线第 31 个 Pass

## 这个 Pass 的作用

`TritonGPUScheduleLoops`（第二次执行，Pass 31）是在 Warp Specialization 完整建立（Pass 20~28）之后的第二轮循环调度。此次执行将单个逻辑 `scf.for` 循环展开为包含 prologue（预热）、steady-state（稳态）和 epilogue（排空）三个部分的多缓冲流水线 IR。与 Pass 19 不同，此时 IR 已包含完整的 shared memory 分配（`ttg.local_alloc`）和异步拷贝（`ttg.async_copy_global_to_local`）原语，因此调度结果直接产出可执行的多缓冲 IR。IR 行数从 292 增至 815。

## IR 变化

Pass 31 引入了 `#shared` 和 `#shared1` 两个新的 shared memory layout 属性，并分配了三重缓冲（depth=3，因 num_stages=4，中间状态数为 3）的 shared memory 缓冲区：

```mlir
// 变换后新增 layout 定义
#shared  = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory

// 变换后新增 shared memory 分配（3 槽缓冲）
%50 = ttg.local_alloc : () -> !ttg.memdesc<3x128x16xf32, #ttg.swizzled_shared<{...order = [1, 0]}>, #ttg.shared_memory, mutable>
%51 = ttg.local_alloc : () -> !ttg.memdesc<3x16x64xf32, #ttg.swizzled_shared<{...order = [0, 1]}>, #ttg.shared_memory, mutable>
```

循环内的 `tt.load` 被替换为异步拷贝链：

```mlir
// 变换后（producer cluster=3，stage=0）
%120 = ttg.memdesc_index %50[%86] {loop.cluster = 3 : i32, loop.stage = 0 : i32}
      : !ttg.memdesc<3x128x16xf32, ...> -> !ttg.memdesc<128x16xf32, ...>
%121 = ttg.async_copy_global_to_local %105, %120 mask %119 other %cst_16 {loop.cluster = 3 : i32, loop.stage = 0 : i32}
      : tensor<128x16x!tt.ptr<f32>, #blocked1> -> <128x16xf32, #shared, ...>
%122 = ttg.async_commit_group tokens %121 {loop.cluster = 3 : i32, loop.stage = 0 : i32}

// consumer cluster=0，stage=3：等待并从 shared memory 加载
%123 = ttg.async_wait %122 {loop.cluster = 0 : i32, loop.stage = 3 : i32, num = 0 : i32}
%124 = ttg.memdesc_index %50[%89] {loop.cluster = 0 : i32, loop.stage = 3 : i32}
      : !ttg.memdesc<3x128x16xf32, ...> -> !ttg.memdesc<128x16xf32, ...>
%125 = ttg.local_load %124 token %123 {loop.cluster = 0 : i32, loop.stage = 3 : i32}
      : !ttg.memdesc<128x16xf32, ...> -> tensor<128x16xf32, #blocked1>
```

循环结束后插入最终 dealloc：

```mlir
%53 = ttg.async_wait {num = 0 : i32}
ttg.local_dealloc %51 : !ttg.memdesc<3x16x64xf32, ...>
ttg.local_dealloc %50 : !ttg.memdesc<3x128x16xf32, ...>
```

## 说明

Pass 31 是整个流水线构建的顶点。三重缓冲（depth=3）的 shared memory 分配：
- `%50`：激活矩阵 X 的三重缓冲，`3×128×16×4 = 24576` 字节
- `%51`：权重矩阵 W 的三重缓冲，`3×16×64×4 = 12288` 字节
- 合计约 36KB shared memory 用于流水线缓冲

`ttg.memdesc_index` 操作通过对 3 取模的索引（`%86 = i mod 3`）在三个缓冲槽中轮转，producer 始终写入当前槽（`%86`），consumer 读取 3 个槽之前的数据（`%89 = (i-3) mod 3`），实现真正的 latency=3 流水线。`#ttg.swizzled_shared` 的 `perPhase=1, maxPhase=1` 表示此处的激活和权重数据不使用 bank-conflict 避免的 swizzle 模式（因为数据块足够小，L1 cache 可以很好地覆盖）。`ttg.async_copy_global_to_local` 将在最终 PTX 中对应 `cp.async.ca.shared.global` 指令，实现真正的异步 DMA，将全局内存拷贝与计算完全解耦。
