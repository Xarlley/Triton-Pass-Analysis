# 卷积 (Convolution) —— IR 逐 Pass 跟踪索引

## kernel 信息

- **函数名**：`triton_tem_fused_convolution_view_2`
- **说明**：VGG16 第 1 个卷积层（3→64 通道，3×3，224×224）。由 Inductor 的 Triton 卷积模板（max_autotune）生成，含 K 维归约循环。
- **编译选项**：`num_warps=4, num_stages=4`，target `cuda:120`（sm_120, Blackwell）
- **来源**：真实 VGG16-SNN 推理（`examples/vgg16_snn/vgg16_test.py`），经 `TRITON_KERNEL_DUMP` 抓取真实 IR 后逐 Pass 重放捕获；产出的 TTGIR / PTX 已与真实运行逐字节对账一致。

## 阶段产物（真实运行的各阶段 IR）

- [`stage_0_entry.ttir`](./stage_0_entry.ttir) —— 进入优化流水线时的 Triton IR
- [`stage_1_final.ttgir`](./stage_1_final.ttgir) —— make_ttgir 结束时的 Triton GPU IR
- [`stage_2.llir`](./stage_2.llir) —— 转换得到的 LLVM IR
- [`stage_3.ptx`](./stage_3.ptx) —— LLVM NVPTX 后端生成的 PTX 汇编

## 完整 Pass 流水线（73 个 Pass，其中 30 个改变了 IR）

> 流水线对所有 kernel 相同。下表「改变 IR」为是的 Pass 各有一篇变换文档；
> 其余 Pass 在本 kernel 上为 no-op（未改变 IR）。

| # | Pass | CLI 名 | 改变 IR | 变换文档 |
|---|------|--------|:---:|------|
| 0 | ConvertTritonToTritonGPU | `convert-triton-to-tritongpu` | ✅ | [pass_00_ConvertTritonToTritonGPU.md](./pass_00_ConvertTritonToTritonGPU.md) |
| 1 | TritonGPUCoalesce | `tritongpu-coalesce` | ✅ | [pass_01_TritonGPUCoalesce.md](./pass_01_TritonGPUCoalesce.md) |
| 2 | TritonGPUF32DotTC | `tritongpu-F32DotTC` |  | — |
| 3 | TritonGPUPlanCTAPass | `triton-nvidia-gpu-plan-cta` |  | — |
| 4 | TritonGPURemoveLayoutConversions | `tritongpu-remove-layout-conversions` | ✅ | [pass_04_TritonGPURemoveLayoutConversions.md](./pass_04_TritonGPURemoveLayoutConversions.md) |
| 5 | TritonGPUOptimizeThreadLocality | `tritongpu-optimize-thread-locality` |  | — |
| 6 | TritonGPUAccelerateMatmul | `tritongpu-accelerate-matmul` |  | — |
| 7 | TritonGPURemoveLayoutConversions | `tritongpu-remove-layout-conversions` |  | — |
| 8 | TritonGPUOptimizeDotOperands | `tritongpu-optimize-dot-operands` |  | — |
| 9 | CanonicalizerPass | `canonicalize` |  | — |
| 10 | TritonNvidiaGPUOptimizeDescriptorEncodingPass | `triton-nvidia-optimize-descriptor-encoding` |  | — |
| 11 | TritonLoopAwareCSE | `triton-loop-aware-cse` | ✅ | [pass_11_TritonLoopAwareCSE.md](./pass_11_TritonLoopAwareCSE.md) |
| 12 | TritonGPUFuseNestedLoops | `tritongpu-fuse-nested-loops` |  | — |
| 13 | CanonicalizerPass | `canonicalize` |  | — |
| 14 | TritonLoopInvariantCodeMotion | `triton-licm` | ✅ | [pass_14_TritonLoopInvariantCodeMotion.md](./pass_14_TritonLoopInvariantCodeMotion.md) |
| 15 | TritonGPUOptimizeAccumulatorInit | `tritongpu-optimize-accumulator-init` |  | — |
| 16 | TritonGPUHoistTMEMAlloc | `tritongpu-hoist-tmem-alloc` |  | — |
| 17 | TritonNvidiaGPUPromoteLHSToTMemPass | `tritongpu-promote-lhs-to-tmem` |  | — |
| 18 | TritonGPUAssignLatencies | `tritongpu-assign-latencies` | ✅ | [pass_18_TritonGPUAssignLatencies.md](./pass_18_TritonGPUAssignLatencies.md) |
| 19 | TritonGPUScheduleLoops | `tritongpu-schedule-loops` | ✅ | [pass_19_TritonGPUScheduleLoops.md](./pass_19_TritonGPUScheduleLoops.md) |
| 20 | TritonGPUAutomaticWarpSpecialization | `tritongpu-automatic-warp-specialization` | ✅ | [pass_20_TritonGPUAutomaticWarpSpecialization.md](./pass_20_TritonGPUAutomaticWarpSpecialization.md) |
| 21 | TritonGPUPartitionScheduling | `tritongpu-partition-scheduling` |  | — |
| 22 | NVWSHoistTmemStore | `nvws-hoist-tmem-store` |  | — |
| 23 | NVWSInsertAref | `nvws-insert-aref` |  | — |
| 24 | NVWSInsertTmemAref | `nvws-insert-tmem-aref` | ✅ | [pass_24_NVWSInsertTmemAref.md](./pass_24_NVWSInsertTmemAref.md) |
| 25 | SCCPPass | `sccp` | ✅ | [pass_25_SCCPPass.md](./pass_25_SCCPPass.md) |
| 26 | CSEPass | `cse` | ✅ | [pass_26_CSEPass.md](./pass_26_CSEPass.md) |
| 27 | NVWSLowerAref | `nvws-lower-aref` | ✅ | [pass_27_NVWSLowerAref.md](./pass_27_NVWSLowerAref.md) |
| 28 | NVWSAssignStagePhase | `nvws-assign-stage-phase` | ✅ | [pass_28_NVWSAssignStagePhase.md](./pass_28_NVWSAssignStagePhase.md) |
| 29 | TritonGPUPartitionLoops | `tritongpu-partition-loops` |  | — |
| 30 | NVWSLowerWarpGroup | `nvws-lower-warp-group` |  | — |
| 31 | TritonGPUScheduleLoops | `tritongpu-schedule-loops` | ✅ | [pass_31_TritonGPUScheduleLoops.md](./pass_31_TritonGPUScheduleLoops.md) |
| 32 | TritonGPUPipeline | `tritongpu-pipeline` | ✅ | [pass_32_TritonGPUPipeline.md](./pass_32_TritonGPUPipeline.md) |
| 33 | TritonGPUOptimizePartitionWarps | `tritongpu-optimize-partition-warps` |  | — |
| 34 | TritonGPUCombineTensorSelectAndIf | `tritongpu-combine-tensor-select-and-if` |  | — |
| 35 | TritonGPUHoistTMEMAlloc | `tritongpu-hoist-tmem-alloc` | ✅ | [pass_35_TritonGPUHoistTMEMAlloc.md](./pass_35_TritonGPUHoistTMEMAlloc.md) |
| 36 | TritonNvidiaGPURemoveTMEMTokensPass | `triton-nvidia-gpu-remove-tmem-tokens` | ✅ | [pass_36_TritonNvidiaGPURemoveTMEMTokensPass.md](./pass_36_TritonNvidiaGPURemoveTMEMTokensPass.md) |
| 37 | CanonicalizerPass | `canonicalize` | ✅ | [pass_37_CanonicalizerPass.md](./pass_37_CanonicalizerPass.md) |
| 38 | TritonLoopAwareCSE | `triton-loop-aware-cse` | ✅ | [pass_38_TritonLoopAwareCSE.md](./pass_38_TritonLoopAwareCSE.md) |
| 39 | TritonGPUOptimizeDotOperands | `tritongpu-optimize-dot-operands` |  | — |
| 40 | CanonicalizerPass | `canonicalize` |  | — |
| 41 | TritonGPUCoalesceAsyncCopy | `tritongpu-coalesce-async-copy` |  | — |
| 42 | TritonNvidiaGPUOptimizeTMemLayoutsPass | `triton-nvidia-optimize-tmem-layouts` |  | — |
| 43 | TritonNvidiaGPUTMALoweringPass | `triton-nvidia-tma-lowering` |  | — |
| 44 | TritonGPURemoveLayoutConversions | `tritongpu-remove-layout-conversions` |  | — |
| 45 | TritonNvidiaGPUInterleaveTMemPass | `triton-nvidia-interleave-tmem` |  | — |
| 46 | TritonGPUReduceDataDuplication | `tritongpu-reduce-data-duplication` |  | — |
| 47 | TritonGPUReorderInstructions | `tritongpu-reorder-instructions` |  | — |
| 48 | TritonLoopAwareCSE | `triton-loop-aware-cse` |  | — |
| 49 | SymbolDCEPass | `symbol-dce` |  | — |
| 50 | TritonGPUFenceInsertion | `triton-nvidia-gpu-fence-insertion` |  | — |
| 51 | TritonNvidiaGPUMMALoweringPass | `triton-nvidia-mma-lowering` |  | — |
| 52 | SCCPPass | `sccp` | ✅ | [pass_52_SCCPPass.md](./pass_52_SCCPPass.md) |
| 53 | CSEPass | `cse` |  | — |
| 54 | CanonicalizerPass | `canonicalize` |  | — |
| 55 | TritonGPUCombineTensorSelectAndIf | `tritongpu-combine-tensor-select-and-if` |  | — |
| 56 | TritonGPUAllocateWarpGroups | `tritongpu-allocate-warp-groups` | ✅ | [pass_56_TritonGPUAllocateWarpGroups.md](./pass_56_TritonGPUAllocateWarpGroups.md) |
| 57 | SCFToControlFlowPass | `convert-scf-to-cf` | ✅ | [pass_57_SCFToControlFlowPass.md](./pass_57_SCFToControlFlowPass.md) |
| 58 | GluonInline | `gluon-inline` |  | — |
| 59 | AllocateSharedMemoryNv | `allocate-shared-memory-nv` | ✅ | [pass_59_AllocateSharedMemoryNv.md](./pass_59_AllocateSharedMemoryNv.md) |
| 60 | TritonTensorMemoryAllocationPass | `triton-tensor-memory-allocation` | ✅ | [pass_60_TritonTensorMemoryAllocationPass.md](./pass_60_TritonTensorMemoryAllocationPass.md) |
| 61 | TritonNvidiaGPUCheckMatmulTwoCTAPass | `triton-nvidia-check-matmul-two-cta` | ✅ | [pass_61_TritonNvidiaGPUCheckMatmulTwoCTAPass.md](./pass_61_TritonNvidiaGPUCheckMatmulTwoCTAPass.md) |
| 62 | TritonGPUProxyFenceInsertion | `triton-nvidia-gpu-proxy-fence-insertion` |  | — |
| 63 | ConvertTritonGPUToLLVM | `convert-triton-gpu-to-llvm` | ✅ | [pass_63_ConvertTritonGPUToLLVM.md](./pass_63_ConvertTritonGPUToLLVM.md) |
| 64 | CanonicalizeLLVMIR | `canonicalize-llvm-ir` | ✅ | [pass_64_CanonicalizeLLVMIR.md](./pass_64_CanonicalizeLLVMIR.md) |
| 65 | CSEPass | `cse` | ✅ | [pass_65_CSEPass.md](./pass_65_CSEPass.md) |
| 66 | ConvertWarpSpecializeToLLVM | `convert-warp-specialize-to-llvm` |  | — |
| 67 | ReconcileUnrealizedCastsPass | `reconcile-unrealized-casts` |  | — |
| 68 | ConvertNVGPUToLLVM | `convert-nv-gpu-to-llvm` | ✅ | [pass_68_ConvertNVGPUToLLVM.md](./pass_68_ConvertNVGPUToLLVM.md) |
| 69 | CanonicalizerPass | `canonicalize` |  | — |
| 70 | CSEPass | `cse` | ✅ | [pass_70_CSEPass.md](./pass_70_CSEPass.md) |
| 71 | SymbolDCEPass | `symbol-dce` |  | — |
| 72 | ConvertNVVMToLLVMPass | `convert-nvvm-to-llvm` |  | — |
