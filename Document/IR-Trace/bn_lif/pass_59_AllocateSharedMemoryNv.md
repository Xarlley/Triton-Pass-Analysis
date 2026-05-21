# Pass 59：AllocateSharedMemoryNv

> kernel：BatchNorm + LIF 脉冲神经元 ｜ CLI：`allocate-shared-memory-nv` ｜ 编译流水线第 59 个 Pass

## 这个 Pass 的作用

AllocateSharedMemoryNv 分析 kernel 中所有需要共享内存（shared memory / SMEM）的操作（主要是 `ttg.convert_layout` 跨布局转换需要 SMEM 作为中间缓冲），计算所需的 SMEM 总量，并在模块属性中记录为 `ttg.shared`（字节数）。同时，为每个需要 SMEM 的操作分配具体的偏移量（通过 `allocation.offset` 属性标注）。

## IR 变化

本次变换对模块属性和一处操作进行了修改：

**模块属性新增 `ttg.shared`：**

```mlir
// 变换前
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32}

// 变换后
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 4096 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32}
```

**唯一的 `ttg.convert_layout` 操作获得偏移量标注：**

```mlir
// 变换前
%8 = ttg.convert_layout %tmp43 : tensor<16x64xf32, #blocked> -> tensor<16x64xf32, #blocked1>

// 变换后
%8 = ttg.convert_layout %tmp43 {allocation.offset = 0 : i32} : tensor<16x64xf32, #blocked> -> tensor<16x64xf32, #blocked1>
```

## 说明

分配了 **4096 字节**（4 KB）的共享内存，这完全来自最后一条 `ttg.convert_layout` 操作：将 LIF 第 4 个时间步的最终输出 `%tmp43`（类型 `tensor<16x64xf32, #blocked>`）转换为写入 `out_ptr0` 所需的 `#blocked1` 布局。

计算验证：
- `tensor<16x64xf32>` = 1024 个 f32 元素
- 每个 f32 = 4 字节
- 1024 × 4 = 4096 字节 = 4 KB

这次布局转换之所以需要 SMEM，是因为源布局 `#blocked = {sizePerThread=[1,4], threadsPerWarp=[2,16], warpsPerCTA=[4,1]}` 和目标布局 `#blocked1 = {sizePerThread=[4,1], threadsPerWarp=[4,8], warpsPerCTA=[1,4]}` 的线程数据分配方式不同，无法通过寄存器对寄存器操作完成转换，必须先写入 SMEM 再以新布局读取。

`allocation.offset = 0` 表示这个 SMEM 分配从地址偏移 0 开始——因为整个 kernel 中只有这一处需要 SMEM，无需多个操作共享内存区域，所以偏移为 0，且总量等于这一个操作的需求量。

对应到 PTX，这 4 KB 共享内存会被声明为 `.shared .b8 global_smem[4096]`，在 LLVM IR 中表现为 `llvm.mlir.global external @global_smem() {addr_space = 3} : !llvm.array<0 x i8>`（实际大小由模块属性决定）。
