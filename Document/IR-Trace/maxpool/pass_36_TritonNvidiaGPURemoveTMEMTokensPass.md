# Pass 36：TritonNvidiaGPURemoveTMEMTokensPass

> kernel：MaxPool + BN + LIF 融合 ｜ CLI：`triton-nvidia-gpu-remove-tmem-tokens` ｜ 编译流水线第 36 个 Pass

## 这个 Pass 的作用

TritonNvidiaGPURemoveTMEMTokensPass 负责清理 Blackwell（sm_120）Tensor Memory（TMem）相关的异步 token 和同步屏障操作。在含 WGMMA 或 TMA 操作的 kernel 中，此 Pass 移除已不再需要的 `ttg.async.token` 类型的 wait token；对于不含 TMem 操作的 kernel，它会插入一个 `ub.poison` 占位来标记 token 槽为未定义值，随后由 Canonicalizer（Pass 37）清理。

## IR 变化

此 Pass 在函数体顶部**插入了一行 `ub.poison` 操作**，IR 行数从 134 增至 135：

**变化前（before）**：
```mlir
tt.func public @triton_poi_fused_...(%in_ptr0: ..., %out_ptr0: ..., %xnumel: ...) {
    %c512_i32 = arith.constant 512 : i32 loc(#loc1)
    %cst = arith.constant dense<64> : tensor<512xi32, #blocked> loc(#loc1)
    ...
    %0 = tt.splat %out_ptr0 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
    %1 = tt.addptr %0, %xindex_8 : ...
    tt.store %1, %tmp6 : tensor<512x!tt.ptr<f32>, #blocked>
```

**变化后（after）**：
```mlir
tt.func public @triton_poi_fused_...(%in_ptr0: ..., %out_ptr0: ..., %xnumel: ...) {
    %0 = ub.poison : !ttg.async.token loc(#loc)
    %c512_i32 = arith.constant 512 : i32 loc(#loc1)
    ...
    %1 = tt.splat %out_ptr0 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
    %2 = tt.addptr %1, %xindex_8 : ...
    tt.store %2, %tmp6 : tensor<512x!tt.ptr<f32>, #blocked>
```

新增的 `%0 = ub.poison : !ttg.async.token` 导致后续所有以 `%N` 编号的 SSA 值编号整体偏移 +1（如 `%0 = tt.splat` → `%1 = tt.splat`，`%1 = tt.addptr` → `%2 = tt.addptr`）。

## 说明

`ub.poison`（undefined behavior poison value）是 MLIR 中表示"未定义值"的标准操作，用于占据一个 SSA 槽位但不产生有意义的结果。此处 RemoveTMEMTokensPass 插入 `ub.poison : !ttg.async.token` 是一种"安全占位"策略：

在有 TMem 的 kernel 中，token 用于追踪异步操作的完成状态；在无 TMem 操作的本 kernel 中，token 槽为空，Pass 需要为框架完整性插入一个占位 token（值为 poison 表示"此 token 从未被创建，应被优化掉"）。

Pass 37（CanonicalizerPass）随即会识别这个 `ub.poison` 值未被任何操作使用，将其作为死代码删除，使 IR 恢复到 134 行。这种"insert then canonicalize"的两步模式是 Triton 编译器中常见的清理策略，确保每个 Pass 只做自己职责范围内的转换，不额外引入清理逻辑。
