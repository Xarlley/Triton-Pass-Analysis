# Pass 37：CanonicalizerPass

> kernel：MaxPool + BN + LIF 融合 ｜ CLI：`canonicalize` ｜ 编译流水线第 37 个 Pass

## 这个 Pass 的作用

CanonicalizerPass 是 MLIR 的通用规范化 Pass，应用所有注册的 canonicalization patterns：消除死代码、化简代数恒等式、折叠常量表达式、删除无用操作等。在编译流水线中，它通常紧跟某个引入冗余代码的 Pass 之后运行，起到"清理整理"的作用。

## IR 变化

此 Pass 删除了 Pass 36 插入的 `ub.poison` 死代码，将 IR 从 135 行压缩回 134 行，并恢复了 store 操作的 SSA 编号：

**变化前（before，含 ub.poison，编号偏移）**：
```mlir
%0 = ub.poison : !ttg.async.token loc(#loc)
...
%1 = tt.splat %out_ptr0 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked> loc(#loc30)
%2 = tt.addptr %1, %xindex_8 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked> loc(#loc30)
tt.store %2, %tmp6 : tensor<512x!tt.ptr<f32>, #blocked> loc(#loc31)
```

**变化后（after，ub.poison 被删除，编号恢复）**：
```mlir
%0 = tt.splat %out_ptr0 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked> loc(#loc30)
%1 = tt.addptr %0, %xindex_8 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked> loc(#loc30)
tt.store %1, %tmp6 : tensor<512x!tt.ptr<f32>, #blocked> loc(#loc31)
```

`ub.poison` 的结果值 `%0` 从未被任何操作使用，Canonicalizer 识别为纯死代码并删除。其余函数体完全不变，函数逻辑内容保持 Pass 25（SCCP）规范化后的形态。

## 说明

`ub.poison : !ttg.async.token` 是一种"定义但不使用"的 dead value（死值），满足 MLIR 死代码消除的标准条件（操作无副作用，结果从未被引用）。Canonicalizer 的 dead code elimination pattern 将其删除。

这次清理与 Pass 36 配合形成一个典型的 Triton 编译器"两步操作"：
1. Pass 36（RemoveTMEMTokensPass）插入 poison token，完成 TMem token 框架的规范化；
2. Pass 37（CanonicalizerPass）删除无用的 poison 值，恢复 IR 的最小形态。

经过此 Pass，IR 进入一个稳定的"最终 TTGIR 状态"，后续 Pass（38–62）主要处理最终内存分配和 lowering 前的收尾工作：OptimizeDotOperands、CoalesceAsyncCopy、TMem 布局优化、TMA lowering 等。对于本 MaxPool+BN+LIF kernel，这些 Pass 大多是空操作，直到 Pass 56（AllocateWarpGroups）才有下一次属性变化。
