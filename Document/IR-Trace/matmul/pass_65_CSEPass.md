# Pass 65：CSEPass（第三次）

> kernel：矩阵乘法 / 全连接 (addmm) ｜ CLI：`cse` ｜ 编译流水线第 65 个 Pass

## 这个 Pass 的作用

这是 CSEPass（公共子表达式消除）的第三次执行（前两次分别在 Pass 26 和 Pass 28）。本次 CSE 在 CanonicalizeLLVMIR（Pass 64）之后对 LLVM 方言 IR 执行，主要消除 Pass 64 规范化后残留的重复计算，以及 loc annotation 的进一步紧缩。

IR 行数从 2319 行降至 2073 行（减少 246 行）。主要变化为：
1. **重复子表达式消除**：消除 LLVM IR 中多次计算相同值的冗余指令（主要集中在常量和共享内存地址计算）。
2. **loc annotation 重新编号**：由于部分 loc 条目不再被引用，编号紧缩（`#loc31→#loc30`、`#loc41→#loc40`、`#loc43→#loc42`、`#loc44→#loc43`、`#loc45→#loc44` 等）。
3. **循环头块参数重编号**：`^bb1` 的块参数 `%acc_344/%acc_345` 重编号为 `%acc_171/%acc_172`（反映 CSE 后指令数量减少导致的 SSA 编号重新分配）。

## IR 变化

**变化 1：loc annotation 编号紧缩（文件头部）**

```mlir
// 变换前（pass 64 after）：
#loc31 = loc(".../69:5")
#loc41 = loc("arg_A"(#loc))
#loc42 = loc("arg_B"(#loc))
#loc43 = loc("out_ptr0"(#loc))
#loc69 = loc("acc"(#loc31))

// 变换后（pass 65 after）：
#loc30 = loc(".../69:5")
#loc40 = loc("arg_A"(#loc))
#loc41 = loc("arg_B"(#loc))
#loc42 = loc("out_ptr0"(#loc))
#loc67 = loc("acc"(#loc30))
```

**变化 2：`%rn` 常量 loc 编号更新**

```mlir
// 变换前：
%rn = llvm.mlir.constant(24 : i32) : i32 loc(#loc44)

// 变换后：
%rn = llvm.mlir.constant(24 : i32) : i32 loc(#loc43)
```

**变化 3：循环头 `^bb1` 块参数编号变化（反映 CSE 消除了约 173 个指令）**

```mlir
// 变换前（pass 64 after）：
^bb1(%acc: i32 loc("acc"(#loc31)), %66: !llvm.struct<(f32,f32,f32,f32,f32,f32,f32,f32)>,
     %acc_344: i32 loc("acc"(#loc31)), %acc_345: i32 loc("acc"(#loc31))):

// 变换后（pass 65 after）：
^bb1(%acc: i32 loc("acc"(#loc30)), %66: !llvm.struct<(f32,f32,f32,f32,f32,f32,f32,f32)>,
     %acc_171: i32 loc("acc"(#loc30)), %acc_172: i32 loc("acc"(#loc30))):
```

## 说明

CSE 在 LLVM IR 级别的消除来源主要是：
- **重复常量折叠残留**：Pass 64 的 canonicalization 将大量冗余常量声明 `llvm.mlir.constant(X : i32)` 合并，但同一值在不同作用域仍可能存在多份。CSE 进一步消除这些跨块重复定义，减少约 246 行。
- **共享内存地址重复计算**：`llvm.mlir.addressof @global_smem` + `llvm.getelementptr` 在 prologue 的多个 stage（K=0,1,2,3 预取）中分别出现，CSE 识别并消除相同偏移量的重复地址计算。
- **线程 ID 读取去重**：`nvvm.read.ptx.sreg.tid.x` 在 Pass 64 after 中仍有多处，CSE 将同一函数内多次读取 tid.x 的操作折叠为一次（但因 `has_side_effects` 属性，实际行为取决于 MLIR CSE 对 intrinsic 的处理策略）。

经过 Pass 65，IR 中的 `%acc_*` 编号从数百跳跃到 171/172 附近，说明约有 170+ 条指令被识别为公共子表达式并消除。剩余 IR 仍保留 `ttg.warp_id` 操作（将在 Pass 68 中替换为 NVVM 等价序列），其他部分已是纯 LLVM 方言。
