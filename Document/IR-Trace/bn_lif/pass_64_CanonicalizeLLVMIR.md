# Pass 64：CanonicalizeLLVMIR

> kernel：BatchNorm + LIF 脉冲神经元 ｜ CLI：`canonicalize-llvm-ir` ｜ 编译流水线第 64 个 Pass

## 这个 Pass 的作用

CanonicalizeLLVMIR 是针对 LLVM IR 方言的专用规范化 Pass，它对 Pass 63 生成的庞大 LLVM IR（3125 行）进行大规模化简，主要工作包括：消除冗余的 `llvm.bitcast`（自身类型到自身类型的无效转换）、折叠常量 struct 的 `insertvalue`/`extractvalue` 链、消除多余的 `llvm.mlir.undef` 中间值、合并重复的常量定义，以及化简线程索引计算中的零值加法和常量乘法。结果是行数从 3125 行缩减至 870 行（减少约 72%）。

## IR 变化

**常量池从分散化简为集中标量定义：**

```mlir
// 变换前（3125 行版本：每个常量通过 struct 展开，含大量重复）
%0 = llvm.mlir.constant(5.000000e-01 : f32) : f32
%1 = llvm.bitcast %0 : f32 to f32        // 冗余 bitcast
%2 = llvm.mlir.undef : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
%3 = llvm.insertvalue %1, %2[0] : ...
...（8 个 insertvalue）
// 同样的 1.0 常量在多个位置重复定义

// 变换后（870 行版本：常量直接定义为标量，去除无用 bitcast 和 undef）
%0 = llvm.mlir.constant(268 : i32) : i32   // SMEM 偏移
%1 = llvm.mlir.constant(264 : i32) : i32
...
%35 = llvm.mlir.constant(5.000000e-01 : f32) : f32
%34 = llvm.mlir.constant(1.000000e+00 : f32) : f32
%33 = llvm.mlir.constant(0.000000e+00 : f32) : f32
%27 = llvm.mlir.constant(50176 : i32) : i32
%30 = llvm.mlir.constant(3211264 : i32) : i32
%29 = llvm.mlir.constant(6422528 : i32) : i32
%28 = llvm.mlir.constant(9633792 : i32) : i32
```

**线程索引计算的零值操作被折叠：**

```mlir
// 变换前（含大量常量 0 的多余操作）
%yindex_10 = llvm.mlir.constant(0 : i32) : i32
%yindex_11 = llvm.mlir.constant(0 : i32) : i32
%yindex_16 = llvm.shl %yindex_8, %yindex_15 : i32    // shl by 0 → identity
%yindex_17 = llvm.or %yindex_14, %yindex_16 : i32    // or with 0 → identity

// 变换后（零值操作被折叠）
%yindex_5 = llvm.shl %yindex_3, %23 : i32      // %23 = 0，保留以备后续 CSE
%yindex_6 = llvm.or %23, %yindex_5 : i32       // → 简化为 %yindex_5
%yindex_7 = llvm.shl %yindex_4, %22 : i32      // warp_id << 5
%yindex_8 = llvm.or %yindex_6, %yindex_7 : i32
```

**warp_id 和 lane_id 计算路径被简化：**

```mlir
// 变换前（重复读取 tid.x 多次）
%yindex_4 = nvvm.read.ptx.sreg.tid.x : i32
%yindex_35 = nvvm.read.ptx.sreg.tid.x : i32
%yindex_39 = nvvm.read.ptx.sreg.tid.x : i32  // 多次重复读取

// 变换后（单次读取，共享）
%yindex = nvvm.read.ptx.sreg.tid.x : i32
%yindex_2 = llvm.and %yindex, %25 : i32    // tid & 127
%yindex_3 = llvm.urem %yindex_2, %24 : i32 // lane_id = (tid & 127) % 32
%yindex_4 = ttg.warp_id {omitUniformHint}  // warp_id（保留为 ttg 方言）
```

## 说明

这次规范化的核心效果是将 Pass 63 生成的"展开式"（每个 tensor 操作展开为 8 路标量 struct）规范化为更紧凑的"标量共享"形式：

1. **消除冗余 bitcast**：`llvm.bitcast %x : f32 to f32` 是自身类型转换，完全无意义，直接替换为 `%x`。Pass 63 生成这类 bitcast 是因为其模板化的 tensor→struct 展开框架总是插入 bitcast，而规范化 Pass 负责清理。

2. **SMEM 地址常量**：after.mlir 开头出现的新常量（268、264、3684、2596 等）是 Pass 63 根据 `allocation.offset=0` 和布局几何形状计算出的 SMEM 字节偏移量，用于 `ttg.convert_layout` 展开为 SMEM load/store 的地址计算。这些常量在 before.mlir 中是通过复杂表达式计算的，规范化后变为直接常量。

3. **行数 3125→870**：约 72% 的 IR 是 Pass 63 插入的展开样板（struct 构建/解构、无用 bitcast、重复常量），这些在规范化后全部合并或消除。最终的 870 行 LLVM IR 是 BN+LIF kernel 的真实计算量——大约对应 PTX 中的 826 行（Pass 65 后）。
