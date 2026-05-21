# Pass 52：SCCPPass（第二次）

> kernel：矩阵乘法 / 全连接 (addmm) ｜ CLI：`sccp` ｜ 编译流水线第 52 个 Pass

## 这个 Pass 的作用

这是 SCCPPass（稀疏条件常量传播）的第二次执行（第一次在 Pass 25）。在本阶段，SCCP 对经过软件流水线展开、控制流转换（Pass 57 之前）等多次变换后的 IR 再次执行常量折叠和死代码消除。

对于本 kernel，Pass 52 的主要效果是将模块级的布局别名顺序重新规范化：`#blocked`（B-load 布局，`sizePerThread=[4,1]`）和 `#blocked2`（dot accumulator 布局，`sizePerThread=[2,2]`）互换名称，使 `#blocked` 重新对应 dot accumulator 布局，`#blocked2` 对应 B-load 布局。IR 行数保持 271 行不变，仅有布局别名名称和 location annotation 编号发生变化。

## IR 变化

**关键变化：`#blocked` 和 `#blocked2` 布局别名互换，loc annotation 编号重新排列。**

**变换前（before）：**

```mlir
#blocked  = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 2], order = [0, 1]}>   ← B-load
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [2, 1], order = [1, 0]}>   ← A-load
#blocked2 = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>  ← dot accumulator
```

**变换后（after）：**

```mlir
#blocked  = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>  ← dot accumulator（恢复）
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [2, 1], order = [1, 0]}>  ← A-load（不变）
#blocked2 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 2], order = [0, 1]}>  ← B-load（恢复）
```

## 说明

这种布局别名重排是 MLIR 在 SCCP pass 内部对模块属性别名顺序进行规范化（de Bruijn 规范化）时产生的副产品，不影响 IR 的语义。SCCP 在本轮没有找到可以继续折叠的常量（软件流水线展开后的常量在 Pass 32 已完全物化），因此唯一可见的变化是布局名称的重排。

经过 Pass 52 规范化后，`#blocked` 再次对应 dot accumulator 布局，`#blocked2` 对应 B-load 布局，与 Pass 26 之后的状态相比布局名称有所来回，但物理布局（`sizePerThread`, `threadsPerWarp`, `warpsPerCTA`）始终不变。
