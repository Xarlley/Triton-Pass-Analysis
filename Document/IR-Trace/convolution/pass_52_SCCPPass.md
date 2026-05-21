# Pass 52：SCCPPass

> kernel：卷积 (Convolution) ｜ CLI：`sccp` ｜ 编译流水线第 52 个 Pass

## 这个 Pass 的作用

`SCCPPass`（Sparse Conditional Constant Propagation，稀疏条件常量传播，第二次执行）在 Pass 51（MMALowering）之后再次运行常量传播。对于本卷积 kernel，此时所有可折叠的常量（如 `cst_16 = dense<224>`、`cst_12 = dense<672>`）在 Pass 25 的 SCCP 阶段已全部折叠完毕，Pass 52 没有发现新的可传播常量。IR 行数保持 403 不变，唯一可见的变化是匿名 `#blocked` 属性的编号顺序被调整（由序列化时的属性排序顺序变化引起），以及调试位置表（`#loc` 表）的编号重排。功能等价的 IR 不受影响。

## IR 变化

**`#blocked` 属性编号重排**（功能等价，仅序号变化）：

```mlir
// 变换前（#blocked 为 16x2 warpsPerCTA = [1,4] 布局）
#blocked  = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

// 变换后（相同的 4 种布局，但编号重排：原 #blocked3 成为 #blocked）
#blocked  = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
```

由于所有操作中的 `#blocked` 引用也随之对应更新，实际计算语义不变。调试位置表（`#loc1`..`#loc118`）也发生对应重排，同样不影响功能。

## 说明

SCCP 的工作原理是在格（lattice）上迭代求解每个 SSA 值的常量状态：若一个值能被静态确定为常量，则将其内联到所有使用处。Pass 25（第一次 SCCP）已处理了所有从程序参数（`tt.get_program_id`）和 kernel 元数据中可推导的常量，包括 VGG16 卷积的各维度大小（3×224×224 = 150528 等），因此 Pass 52 运行时所有值均已达到常量格的不动点，没有新的折叠机会。

`#blocked` 编号重排是 MLIR IR 序列化的正常现象：MLIR 在每次 Pass 后重新序列化 IR 时，会按某种内部遍历顺序（通常是首次出现顺序）重新为匿名属性分配编号。Pass 51（MMALowering）可能修改了某些操作使用 `#blocked` 属性的顺序，导致序列化时属性的首次出现顺序发生变化，进而引起编号重排。这一行为在 MLIR 编译流水线中是预期且无害的。
