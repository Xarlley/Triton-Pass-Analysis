# Pass 27：NVWSLowerAref

> kernel：卷积 (Convolution) ｜ CLI：`nvws-lower-aref` ｜ 编译流水线第 27 个 Pass

## 这个 Pass 的作用

`NVWSLowerAref`（NVIDIA Warp Specialization Lower Async Reference）将高层的 `nvws.aref` 异步引用原语降低为 Triton GPU IR 中的具体内存同步操作，例如 shared memory 分配、async-copy 发起指令、信号量/屏障等。经过 Pass 24（InsertTmemAref）和 Pass 25/26 清理后，IR 中的 aref 描述了 producer 和 consumer 之间的数据流合同；Pass 27 将其展开为实际的 GPU 内存操作序列。IR 行数从 292 扩展到 587，说明每个 aref 被展开为多条具体操作。

## IR 变化

Pass 27 将 292 行的抽象 IR 展开为 587 行，展开模式类似于 Pass 20（WarpSpecialization）——主 IR 加上验证副本。新增的部分是对 `nvws.aref` 的具体化展开，包括：
- shared memory 缓冲区的实际分配操作
- producer warp group 使用 `ttg.async_copy_global_to_local` 发起异步拷贝
- consumer warp group 使用 `ttg.wait_barrier` 等待数据就绪

```mlir
// 变换后（验证副本中的常量，包含具体的卷积参数）
%cst    = arith.constant dense<64>      : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%cst_0  = arith.constant dense<3211264> : tensor<128x1xi32, #blocked>  // 64×224×224，输出总元素
%cst_1  = arith.constant dense<64>      : tensor<128x1xi32, #blocked>
%cst_2  = arith.constant dense<14336>   : tensor<128x1xi32, #blocked>  // 64×224，行步长
%cst_4  = arith.constant dense<3>       : tensor<16x1xi32, #blocked1>  // 卷积核大小
%cst_11 = arith.constant dense<27>      : tensor<64xi32, ...>           // 3×3×3，权重组大小
%cst_12 = arith.constant dense<150528>  : tensor<128xi32, ...>          // 3×224×224，输入步长
%cst_15 = arith.constant dense<0.000000e+00> : tensor<16x64xf32, #blocked1>   // 权重零填充
%cst_16 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #blocked2>  // 激活零填充
%cst_17 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked3>  // 累加器初值
```

`#blocked` layout 重新编号为按分区功能排列：`#blocked`（输出存储）→ `#blocked1`（权重 W 视图）→ `#blocked2`（激活 X 视图）→ `#blocked3`（点积累加器视图）。

## 说明

Pass 27 是 Warp Specialization 流水线中将"描述"变为"实现"的关键步骤。在 Pass 24 中插入的 `nvws.aref` 类似于一个接口契约：producer 承诺在某个 shared memory 位置写入数据，consumer 承诺在 aref 信号后读取。Pass 27 将这个契约实例化为：
1. 为每个流水线阶段的每种数据（激活 X 的 `num_stages=4` 个缓冲 + 权重 W 的 4 个缓冲）分配 shared memory 槽位
2. 在 producer 路径中插入异步全局内存到 shared memory 拷贝指令
3. 在 consumer 路径中插入屏障等待指令，确保在 `tt.dot` 执行前 producer 已经完成当前阶段的写入

对于本卷积 kernel，4 级流水线意味着需要 4 × (128×16 + 16×64) × sizeof(float) = 4 × (2048 + 1024) × 4 = 49152 字节的 shared memory 缓冲区，这正是 Blackwell sm_120 上 192KB shared memory 的合理分配。
