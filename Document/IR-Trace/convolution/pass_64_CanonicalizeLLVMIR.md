# Pass 64：CanonicalizeLLVMIR

> kernel：卷积 (Convolution) ｜ CLI：`canonicalize-llvm-ir` ｜ 编译流水线第 64 个 Pass

## 这个 Pass 的作用

`CanonicalizeLLVMIR`（LLVM IR 规范化）在 Pass 63 生成的 22043 行原始 LLVM IR 上执行 LLVM 方言特有的规范化清理，包括：删除冗余的 `builtin.unrealized_conversion_cast`（Triton 到 LLVM 类型的临时桥接转型）、消除 `llvm.bitcast f32 to f32`（同类型 bitcast，恒等操作）、删除 `#blocked` 等现在已无用的 TritonGPU 属性定义，以及压缩调试位置表中不再引用的条目。IR 行数从 22043 大幅降至 6373（减少 15670 行，约 71% 的行数消除），这是因为大量"转换桥接"代码被移除。

## IR 变化

**删除 `builtin.unrealized_conversion_cast` 桥接转型**（共 82 处）：

```mlir
// 变换前（Pass 63 为 Triton 类型到 LLVM 类型的桥接插入的转型）
%out_ptr0_0 = builtin.unrealized_conversion_cast %out_ptr0 : !llvm.ptr<1> to !tt.ptr<f32>
%arg_W_1 = builtin.unrealized_conversion_cast %arg_W : !llvm.ptr<1> to !tt.ptr<f32>
%arg_X_2 = builtin.unrealized_conversion_cast %arg_X : !llvm.ptr<1> to !tt.ptr<f32>

// 变换后（删除，后续操作直接使用 !llvm.ptr<1>）
（已删除）
```

**删除 `llvm.bitcast f32 to f32` 恒等操作**：

```mlir
// 变换前（对 f32 常量进行多余的 bitcast）
%0 = llvm.mlir.constant(0.000000e+00 : f32) : f32
%1 = llvm.bitcast %0 : f32 to f32   // 无意义，f32 → f32
%2 = llvm.mlir.undef : !llvm.struct<(f32 × 64)>
%3 = llvm.insertvalue %1, %2[0] : ...

// 变换后（bitcast 被折叠，直接使用常量）
%108 = llvm.mlir.constant(0.000000e+00 : f32) : f32
%107 = llvm.mlir.undef : !llvm.struct<(f32 × 64)>
%109 = llvm.insertvalue %108, %107[0] : ...
```

**删除 `#blocked` 等 TritonGPU 属性定义**（IR 文件头部）：

```mlir
// 变换前（仍保留 #blocked = #ttg.blocked<...>）
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#loc = loc(...)

// 变换后（#blocked 定义已删除，因为 LLVM IR 中不再引用它）
#loc = loc(...)
```

## 说明

`CanonicalizeLLVMIR` 的 71% 行数消除（22043→6373）主要来自两个机制：

1. **`builtin.unrealized_conversion_cast` 删除**：Pass 63 在将 TritonGPU 方言转换到 LLVM 方言时，为了保持 SSA 正确性，会临时插入 `builtin.unrealized_conversion_cast` 作为不同 dialect 类型之间的"桥接"。这些桥接在所有使用处都已被正确替换后，就可以被规范化消除。本 kernel 共有 82 处这样的桥接（3 个函数参数 × 多处使用 + 其他类型转换场景）。

2. **`llvm.bitcast f32 to f32` 折叠**：在 Pass 63 中，累加器初始化时先通过 `llvm.bitcast` 将常量 `0.0f32` 转为 `f32`（本质上是 no-op），再用于 64 次 `insertvalue`。规范化 Pass 将 64 处对该 bitcast 结果的引用都替换为对原始常量的直接引用，然后删除 bitcast 本身。

3. **调试信息压缩**：删除 `#blocked` 属性定义后，大量原本引用该属性的 `#loc` 条目也随之失效并被清理。`#loc` 表从 Pass 63 中的 ~119 条压缩至 Pass 64 中的更少条目。

经过此 Pass，IR 已完全处于 LLVM 方言 + NVVM 方言中，没有任何 TritonGPU 方言的痕迹，可以直接交给 LLVM 后端（或继续经过下游 Pass）处理。
