# Pass 27：NVWSLowerAref

> kernel：BatchNorm + LIF 脉冲神经元 ｜ CLI：`nvws-lower-aref` ｜ 编译流水线第 27 个 Pass

## 这个 Pass 的作用

NVWSLowerAref 将 Warp 专业化抽象中的 `nvws.aref` 节点（抽象引用）具体化为 Blackwell 架构的同步原语（如 barrier、token、tmem 相关指令）。如果 kernel 未经 warp 专业化（无 `nvws.aref`），Pass 会输出一个带有验证用 IR 副本的文件，用于后续分析。对于本 kernel，行数从 233 行增至 469 行，原因与 Pass 20 相同：Pass 输出了两份 IR（原始 + 验证副本）。

## IR 变化

本 Pass 对功能 IR **未作任何修改**，只是再次输出了两份相同的 IR（追加了一份验证副本）。两份 IR 内容完全一致，第二份带有注释头：

```
// -----// IR Dump Before (anonymous namespace)::VerifyWarpSpecializationPartitions () ('builtin.module' operation) //----- //
```

两份 IR 均使用相同的布局（`#blocked1` 是读取路径布局，`#blocked` 是写入路径布局）：

```mlir
#blocked  = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 8],  warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
```

常量池结构与 Pass 26 after 完全相同：

```mlir
%cst   = arith.constant dense<50176>   : tensor<1x64xi32, #blocked>
%cst_0 = arith.constant dense<9633792> : tensor<1x64xi32, #blocked1>
%cst_1 = arith.constant dense<6422528> : tensor<1x64xi32, #blocked1>
%cst_2 = arith.constant dense<3211264> : tensor<1x64xi32, #blocked1>
%cst_3 = arith.constant dense<64>      : tensor<16x1xi32, #blocked1>
%cst_4 = arith.constant dense<64>      : tensor<1x64xi32, #blocked>
%cst_5 = arith.constant dense<64>      : tensor<1x64xi32, #blocked1>
%cst_6 = arith.constant dense<0.000000e+00> : tensor<16x64xf32, #blocked1>
%cst_7 = arith.constant dense<1.000000e+00> : tensor<16x64xf32, #blocked1>
%cst_8 = arith.constant dense<5.000000e-01> : tensor<16x64xf32, #blocked1>
```

## 说明

对于本 BN+LIF pointwise kernel，NVWSLowerAref 是完全的 no-op（从功能角度），因为没有任何 `nvws.aref` 节点需要降级。Pass 仍然输出双份 IR，这是 Triton 编译框架对 NVWS 系列 Pass 的统一处理策略——无论是否发生实质变换，都会在 NVWS 的验证链上输出快照。

真正的 NVWSLowerAref 降级在 GEMM kernel 中发生时，会将 `nvws.aref` 转换为 `nvvm.mbarrier.*` 系列指令（Blackwell 的硬件 barrier），或为 tensor memory 的读写插入 token 握手机制。这些都是 Blackwell 架构（sm_120）上实现 warp 专业化所必需的同步基础设施，但对逐元素 kernel 无意义。

后续的 NVWSAssignStagePhase（Pass 28）将再次将双份 IR 压缩回单份。
