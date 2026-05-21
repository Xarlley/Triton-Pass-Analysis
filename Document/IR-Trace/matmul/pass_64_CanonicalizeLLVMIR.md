# Pass 64：CanonicalizeLLVMIR

> kernel：矩阵乘法 / 全连接 (addmm) ｜ CLI：`canonicalize-llvm-ir` ｜ 编译流水线第 64 个 Pass

## 这个 Pass 的作用

CanonicalizeLLVMIR 对 Pass 63 生成的 LLVM 方言 IR 执行 LLVM 专项规范化，主要完成以下工作：

1. **消除 `builtin.unrealized_conversion_cast`**：Pass 63 为了桥接 TTG 类型和 LLVM 类型，大量插入了 `builtin.unrealized_conversion_cast`（如 `tensor<16x32xf32, #blocked> ↔ !llvm.struct<...>`），本 Pass 将这些桥接操作全部折叠消除，使 IR 成为纯 LLVM 方言。
2. **删除 `#blocked` 布局别名**：因 `builtin.unrealized_conversion_cast` 消除后不再有任何 TTG 类型引用，`#blocked` 别名声明被移除（文件头部减少 1 行）。
3. **常量合并与位移优化**：合并重复常量，将冗余的 `llvm.bitcast x : T to T`（同类型 bitcast，即 no-op）折叠消除，大量简化初始化序列中的 `undef + insertvalue` 链式构建。
4. **loc annotation 重排**：移除了若干不再被引用的 loc 条目（`#loc31→#loc30`、`#loc41→#loc40`、`#loc45→#loc44` 等），编号紧缩。

IR 行数从 7710 行骤降至 2319 行（减少约 5391 行，69% 的代码被消除），是全流水线中单次 Pass 减少行数最多的一步。

## IR 变化

**变化 1：移除 `#blocked` 布局别名（文件第 1 行）**

```mlir
// 变换前（before，第 1 行）：
#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
#loc = loc(...)

// 变换后（after）：
#loc = loc(...)    ← #blocked 完全删除
```

**变化 2：消除 `builtin.unrealized_conversion_cast` 桥接**

```mlir
// 变换前（pass 63 中的典型桥接模式）：
%out_ptr0_0 = builtin.unrealized_conversion_cast %out_ptr0 : !llvm.ptr<1> to !tt.ptr<f32>
%arg_B_1 = builtin.unrealized_conversion_cast %arg_B : !llvm.ptr<1> to !tt.ptr<f32>
%arg_A_2 = builtin.unrealized_conversion_cast %arg_A : !llvm.ptr<1> to !tt.ptr<f32>
// ... 函数体内数百处 builtin.unrealized_conversion_cast ...
%11 = builtin.unrealized_conversion_cast %10 : !llvm.struct<(f32,...,f32)> to tensor<16x32xf32, #blocked>

// 变换后：所有桥接被折叠消除，指针/struct 直接以 LLVM 类型使用
// （函数参数 %arg_A 直接作为 !llvm.ptr<1> 使用，无需 cast）
```

**变化 3：冗余常量和 undef-insertvalue 链简化**

```mlir
// 变换前（pass 63 初始化 f32 struct，含 bitcast no-op）：
%0 = llvm.mlir.constant(0.000000e+00 : f32) : f32
%1 = llvm.bitcast %0 : f32 to f32          ← no-op，被消除
%2 = llvm.mlir.undef : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
%3 = llvm.insertvalue %1, %2[0] : ...       ← 使用 %1（bitcast 后），折叠为直接用 %0
// ... (共 8 个 insertvalue)

// 变换后（直接用 %55，无中间 bitcast）：
%54 = llvm.mlir.undef : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
%55 = llvm.mlir.constant(0.000000e+00 : f32) : f32
%56 = llvm.insertvalue %55, %54[0] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
%57 = llvm.insertvalue %55, %56[1] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
...
%63 = llvm.insertvalue %55, %62[7] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
```

**变化 4：函数开头新增 shared memory 全局地址（hoisted 到函数顶部）**

```mlir
// 变换后（pass 64 after，常量提升到函数入口块顶部）：
%33 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>   ← 提前到函数开头
```

## 说明

行数从 7710 降至 2319 的主要来源：
- **`builtin.unrealized_conversion_cast` 消除**：Pass 63 每处类型转换会生成 1 行 cast + 上下游各引用修改，成批删除约 2000+ 行。
- **初始化 struct 的 bitcast no-op 消除**：Pass 63 为每个常量生成 `mlir.constant` + `bitcast` 再 `insertvalue`（3 步），折叠后只需 1 步 `mlir.constant` + 1 步 `insertvalue`，对于大型 struct（16 元素、32 元素）可消除数十行。
- **常量合并**：Pass 63 为每个使用点各自生成独立的 `llvm.mlir.constant`（因为那时还在进行 lowering），CanonicalizeLLVMIR 将相同值的常量合并为一个，大幅减少常量声明行数。
- **loc annotation 紧缩**：unused loc 条目删除，编号重新排列（`#loc31→#loc30` 等，偏移量减少 1~2）。

经过 Pass 64，IR 变为纯 LLVM 方言（不含任何 `tt.*`、`ttg.*`、`builtin.*` 操作），但仍保留少量 `ttg.warp_id`（尚未降低，将在 Pass 68 处理）和少量 Triton 特有属性（`tt.divisibility`）。
