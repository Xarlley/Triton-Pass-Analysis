# Pass 36：TritonNvidiaGPURemoveTMEMTokensPass

> kernel：BatchNorm + LIF 脉冲神经元 ｜ CLI：`triton-nvidia-gpu-remove-tmem-tokens` ｜ 编译流水线第 36 个 Pass

## 这个 Pass 的作用

TritonNvidiaGPURemoveTMEMTokensPass 清理 Tensor Memory（TMem）操作中不再需要的 token 链。在 Blackwell 架构中，TMem 读写需要通过 token 进行同步（类似 async token），完成流水线化或 warp 专业化后，部分 token 已失去同步意义，可以被移除或替换为 `ub.poison`（undefined behavior poison value）。对于没有 TMem 操作的 kernel，Pass 仅插入一个占位 poison 值。

## IR 变化

本次变换非常微小，行数从 233 行增至 234 行，仅在函数体开头新增一行：

**变换前（第 10 行后无 poison）：**

```mlir
  llvm.mlir.global external @global_smem() ...
  tt.func public @triton_poi_fused_...(...) {
    %cst = arith.constant dense<50176> : tensor<1x64xi32, #blocked>
```

**变换后（新增 ub.poison 行）：**

```mlir
  tt.func public @triton_poi_fused_...(...) {
    %0 = ub.poison : !ttg.async.token
    %cst = arith.constant dense<50176> : tensor<1x64xi32, #blocked>
```

同时，后续使用 token 的操作（若有）会改为引用 `%0`，但本 kernel 中 `%0` 未被任何后续操作引用（它只是一个悬空定义），后续的 `CanonicalizerPass`（Pass 37）会将其清除。

## 说明

`ub.poison : !ttg.async.token` 是一个未定义值（类似 LLVM 的 `undef`），表示"此处本应有一个 async token，但已被确认不需要同步"。对于本 BN+LIF kernel：

- kernel 没有使用任何 TMem（Tensor Memory 是 Blackwell 上 MMA accumulator 的专用存储区），也没有异步 token 链，因此 Pass 只是插入了一个占位 poison 值以满足框架的统一处理逻辑。
- `!ttg.async.token` 类型代表异步操作完成信号，在有 TMem 的 GEMM kernel 中会由 `ttng.tmem_load` 等操作产生，并被后续的 barrier 等待操作消费。
- 实际上这个 `%0 = ub.poison : !ttg.async.token` 值在 BN+LIF kernel 中没有任何消费者，它将在 Pass 37 的规范化中被识别为死代码并删除，恢复到 233 行。

这个变换揭示了 Triton 编译框架的一个设计模式：为所有 kernel（无论是否使用相关特性）统一通过相同的 Pass 序列，通过 poison 值占位 + 后续 canonicalize 删除来处理"本 Pass 无事可做"的情况，而不是在 Pass 内部做条件判断跳过。
