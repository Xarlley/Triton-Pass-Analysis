# Pass 37：CanonicalizerPass

> kernel：矩阵乘法 / 全连接 (addmm) ｜ CLI：`canonicalize` ｜ 编译流水线第 37 个 Pass

## 这个 Pass 的作用

CanonicalizerPass 执行 MLIR 标准规范化变换（canonicalization），应用各个操作（op）注册的规范化规则（canonicalization patterns）来简化 IR。在本阶段，该 Pass 完成两件关键工作：

1. **消除 `ub.poison` token**（Pass 36 插入的占位符）：将 `%0 = ub.poison : !ttg.async.token` 声明移除，相关变量编号恢复。
2. **融合 `ttg.local_load` + `ttg.convert_layout` 为直接输出 `dot_op` 布局的 load**：将循环内的两步操作（先 load 到 `#blocked` 布局，再 `ttg.convert_layout` 到 `#ttg.dot_op`）合并为一步，`ttg.local_load` 直接输出 `#ttg.dot_op` 布局，从而消除中间转换。

before 文件 274 行，after 文件 271 行（减少 3 行）。

## IR 变化

**变化 1：移除 `ub.poison` 声明（减少 1 行），变量编号恢复：**

```mlir
// before：
%0 = ub.poison : !ttg.async.token    ← 移除
%1 = arith.cmpi sge, %pid_m_16, %c0_i32 : i32
llvm.intr.assume %1 : i1

// after：
%0 = arith.cmpi sge, %pid_m_16, %c0_i32 : i32    ← 编号恢复为 %0
llvm.intr.assume %0 : i1
```

**变化 2：循环内 `ttg.local_load` 直接输出 `dot_op` 布局，消除后续 `ttg.convert_layout`（减少 2 行）：**

```mlir
// before（先 load 到 blocked，再 convert_layout 到 dot_op）：
%a_118 = ttg.local_load %a_117 token %a_116 : !ttg.memdesc<16x32xf32, #shared, #smem, mutable>
         -> tensor<16x32xf32, #blocked1>                              ← load 到 blocked
%b_120 = ttg.local_load %b_119 token %a_116 : !ttg.memdesc<32x32xf32, #shared1, #smem, mutable>
         -> tensor<32x32xf32, #blocked>                               ← load 到 blocked
%a_121 = ttg.convert_layout %a_118 : tensor<16x32xf32, #blocked1>
         -> tensor<16x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>   ← 转换
%b_122 = ttg.convert_layout %b_120 : tensor<32x32xf32, #blocked>
         -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>>   ← 转换
%acc_123 = tt.dot %a_121, %b_122, %arg4 : ...

// after（load 直接输出 dot_op 布局，无需 convert_layout）：
%a_118 = ttg.local_load %a_117 token %a_116 : !ttg.memdesc<16x32xf32, #shared, #smem, mutable>
         -> tensor<16x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>   ← 直接输出 dot_op
%b_120 = ttg.local_load %b_119 token %a_116 : !ttg.memdesc<32x32xf32, #shared1, #smem, mutable>
         -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>>   ← 直接输出 dot_op
%acc_121 = tt.dot %a_118, %b_120, %arg4 : ...                         ← 变量号减少
```

## 说明

这两个规范化变换都是 Triton/MLIR 中的标准 canonicalization 规则：

- **`ub.poison` 消除**：`ub.poison` 值在任何使用点都可以被任意值替换（因为它表示 undefined behavior），Canonicalizer 通过折叠使用了 poison 值的操作来清理 IR。

- **`ttg.local_load` → `dot_op` 融合**：`ttg.local_load` 操作的输出布局可以直接指定为目标布局（包括 `dot_op`），而无需先 load 到中间的 `blocked` 布局再做转换。Canonicalizer 识别了"local_load → convert_layout → dot"这一模式，并将其合并为"local_load（直接输出 dot_op）→ dot"，消除了两个 `ttg.convert_layout` 操作（A 和 B 各一个），共节省 2 行 IR、2 条运行时指令。

经过 Pass 37，循环体内 MMA 计算路径从"load → convert → dot"简化为"load（直接 dot_op 格式）→ dot"，是在进入 LLVM lowering 之前的最后一次高层 IR 简化。这对于高效生成 PTX wgmma/mma 指令至关重要，因为 wgmma 操作要求操作数以特定的寄存器布局排列，直接 load 为 `dot_op` 格式可以避免额外的寄存器重排操作。
