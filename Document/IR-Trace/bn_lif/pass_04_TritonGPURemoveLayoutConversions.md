# Pass 04：TritonGPURemoveLayoutConversions

> kernel：BatchNorm + LIF 脉冲神经元 ｜ CLI：`tritongpu-remove-layout-conversions` ｜ 编译流水线第 4 个 Pass

## 这个 Pass 的作用

TritonGPURemoveLayoutConversions 通过全局布局传播来消除不必要的 `ttg.convert_layout` 操作。Pass 分析整个函数中所有 tensor 的布局需求，选取一个"主导布局"（dominant layout），将所有操作统一到该布局下，从而大规模消除 convert_layout。当两种布局不能合并时，Pass 会保留必要的转换，但会尽量减少转换次数。参见 [`RemoveLayoutConversions.md`](../../Passes/RemoveLayoutConversions.md)。

## IR 变化

这次变换是一次彻底的布局重构，行数从 250 行降至 233 行（消除了 17 行冗余操作）。

**布局别名从 7 个收缩为 2 个：**

```mlir
// 变换前（7 种布局）
#blocked  = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32],  warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1],  warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1],    threadsPerWarp = [32],     warpsPerCTA = [4],    order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1],  warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32],  warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16],  warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 8],   warpsPerCTA = [1, 4], order = [0, 1]}>

// 变换后（2 种布局，即原来的 #blocked5 和 #blocked6）
#blocked  = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16],  warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 8],   warpsPerCTA = [1, 4], order = [0, 1]}>
```

**load 指令周围的冗余 convert_layout 链被消除：**

```mlir
// 变换前（4 个操作）
%tmp0_24 = ttg.convert_layout %tmp0_22 : tensor<16x64x!tt.ptr<f32>, #blocked> -> tensor<16x64x!tt.ptr<f32>, #blocked5>
%tmp0_25 = ttg.convert_layout %tmp0_23 : tensor<16x64xi1, #blocked> -> tensor<16x64xi1, #blocked5>
%tmp0_26 = tt.load %tmp0_24, %tmp0_25 evictionPolicy = evict_last : tensor<16x64x!tt.ptr<f32>, #blocked5>
%tmp0_27 = ttg.convert_layout %tmp0_26 : tensor<16x64xf32, #blocked5> -> tensor<16x64xf32, #blocked>

// 变换后（1 个操作，直接在最优布局执行）
%tmp0_30 = tt.load %tmp0_27, %tmp0_28 evictionPolicy = evict_last : tensor<16x64x!tt.ptr<f32>, #blocked>
```

**索引计算路径也被统一布局，消除了 convert_layout：**

```mlir
// 变换前：yindex 需要经过 #blocked2 → slice → #blocked3 → #blocked1 三步转换
%yindex = tt.make_range ... : tensor<16xi32, #blocked2>
%yindex_5 = ttg.convert_layout %yindex : tensor<16xi32, #blocked2> -> tensor<16xi32, #ttg.slice<{dim=1, parent=#blocked3}>>
%yindex_6 = tt.expand_dims %yindex_5 ... -> tensor<16x1xi32, #blocked3>
%yindex_7 = ttg.convert_layout %yindex_6 : tensor<16x1xi32, #blocked3> -> tensor<16x1xi32, #blocked1>

// 变换后：直接在目标布局生成 range，无需转换
%yindex = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%yindex_7 = tt.expand_dims %yindex {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
```

## 说明

这次变换的本质是：将 Coalesce Pass 插入的"转换成最优布局→操作→转换回去"三明治结构，替换为"所有操作都直接在最优布局下执行"的扁平结构。

具体到本 BN+LIF kernel：
- **主导布局确定为原 `#blocked5`**（即新的 `#blocked`），这是 4 次 load 和 store 都需要的 coalesced 布局。Pass 判断这个布局可以被全局采用，因此将整个计算图（包括 LIF 发放的 `arith.cmpf`、`arith.mulf`、`arith.addf` 等算术操作）都统一到这个布局下运行。
- **写入路径保留了 `#blocked1`**（原 `#blocked6`），因为 `out_ptr0` 的 store 涉及不同的索引计算（`xindex * 50176 + yindex`），需要不同的线程分配方式。
- 这次变换直接决定了 BN+LIF 所有计算指令在 GPU 上的线程分配方案：每线程处理 4 个连续的通道（x 方向），4 个 warp 各负责 4 行（y 方向），实现了计算与访存的布局一致性，避免了运行时的数据重排开销。
