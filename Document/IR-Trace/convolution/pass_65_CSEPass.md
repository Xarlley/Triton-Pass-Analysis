# Pass 65：CSEPass

> kernel：卷积 (Convolution) ｜ CLI：`cse` ｜ 编译流水线第 65 个 Pass

## 这个 Pass 的作用

`CSEPass`（公共子表达式消除，第三次执行，在 LLVM IR 层面）对 Pass 64 产出的 6373 行 LLVM IR 进行 CSE，消除在不同计算路径中重复读取的特殊寄存器值（如 `nvvm.read.ptx.sreg.tid.x`、`nvvm.read.ptx.sreg.ctaid.x`、`ttg.warp_id`）以及重复的常量加载。IR 行数从 6373 降至 5752（减少 621 行，约 10%）。参见 [`CSE.md`](../../Passes/CSE.md)。

## IR 变化

**消除重复的 `nvvm.read.ptx.sreg.tid.x` 读取**：

```mlir
// 变换前（prologue 和 epilogue 各自独立读取 threadIdx.x，形成多余的重复）
%nhw_1 = nvvm.read.ptx.sreg.tid.x : i32    // prologue 第一次 load 的 tid
// ...（中间有其他操作）
%nhw_45 = nvvm.read.ptx.sreg.tid.x : i32   // prologue 第二次 load 的 tid（重复）
%idx_y_c_174 = nvvm.read.ptx.sreg.tid.x : i32  // 输出 store 的 tid（又一次重复）
%idx_x_c = nvvm.read.ptx.sreg.tid.x : i32      // 权重指针计算的 tid（另一次重复）
// ...（共约 20+ 次重复读取）

// 变换后（CSE 合并所有同类读取，后续均引用同一个值）
%nhw_1 = nvvm.read.ptx.sreg.tid.x : i32   // 只保留一次读取
// （原有 %nhw_45, %idx_y_c_174, %idx_x_c 等全部被替换为对 %nhw_1 的引用）
```

**消除重复的 `ttg.warp_id` 读取**：

```mlir
// 变换前
%nhw_4 = ttg.warp_id {omitUniformHint} : i32
%nhw_48 = ttg.warp_id {omitUniformHint} : i32   // 重复
%idx_y_c_177 = ttg.warp_id {omitUniformHint} : i32  // 再次重复
// ...（共约 10+ 次）

// 变换后（所有引用统一指向第一次出现的 %nhw_4）
```

**消除重复的 `nvvm.read.ptx.sreg.ctaid.x/y` 读取**：

```mlir
// 变换前（idx_y_c 块调用了多次 ctaid.y 读取）
%nhw = nvvm.read.ptx.sreg.ctaid.x : i32
%idx_y_c = nvvm.read.ptx.sreg.ctaid.y : i32
// ...（这些在 prologue 和 loop 体中各读一次）

// 变换后（合并为单次读取，仅在函数入口处读取一次）
```

## 说明

在 Pass 63（ConvertTritonGPUToLLVM）展开后，软件流水线的 prologue、steady-state 循环体和 epilogue 三段式结构中，每段都独立生成了计算线程索引所需的特殊寄存器读取（`tid.x`、`warp_id`、`ctaid.x/y`）。这些读取在 GPU 硬件上是读取不可变的寄存器（threadIdx 和 blockIdx 在 kernel 执行期间不变），因此是纯函数（没有副作用），完全可以被 CSE 合并。

621 行的减少主要来自消除重复的特殊寄存器读取（约 20 个 `tid.x`、10 个 `warp_id`、若干个 `ctaid`），以及随之消除的依赖于这些值的相同计算链（对同一 `tid.x` 值执行 `and`、`rem`、`shl` 等计算得到线程内的 lane/warp 偏移量）。

CSE 在 LLVM IR 层面相比在高层 IR 层面效果更强，因为向量操作已经被展开为标量，使得相同的标量子表达式在不同向量通道中的重复更加明显。
