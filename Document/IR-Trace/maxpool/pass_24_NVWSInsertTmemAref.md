# Pass 24：NVWSInsertTmemAref

> kernel：MaxPool + BN + LIF 融合 ｜ CLI：`nvws-insert-tmem-aref` ｜ 编译流水线第 24 个 Pass

## 这个 Pass 的作用

NVWSInsertTmemAref 是 Blackwell（sm_120）Nvidia Warp Specialization（NVWS）流水线中的一环。其职责是：在 warp 专化场景下，为需要通过 Tensor Memory（TMem）传递数据的操作插入 `nvws.aref`（asynchronous reference）标记，以便后续 Pass（NVWSLowerAref）将其降级为具体的同步/异步屏障操作。本 Pass 针对含 TMem 的 warp 专化 kernel 工作。

## IR 变化

本 kernel 是纯 pointwise 逐元素操作，不含 TMem 访问，因此 NVWSInsertTmemAref 对函数体逻辑没有任何修改。after.mlir 的行数仍为 271（与 before 相同），其中多余的 137 行来自 Pass 20（AutomaticWarpSpecialization）遗留的诊断 dump，并非此 Pass 引入。

实际差异发生在诊断 dump 内部的常量顺序重排——这是此 Pass 内部 SCCP 预处理的副作用：

**before 的诊断区常量（行 145–151）**：
```mlir
%cst = arith.constant dense<28672> : tensor<512xi32, #blocked>
%cst_0 = arith.constant dense<128> : tensor<512xi32, #blocked>
...
%cst_1 = arith.constant dense<64> : tensor<512xi32, #blocked>
```

**after 的诊断区常量（行 145–151）**：
```mlir
%cst = arith.constant dense<64> : tensor<512xi32, #blocked>
%cst_0 = arith.constant dense<112> : tensor<512xi32, #blocked>
%cst_1 = arith.constant dense<7168> : tensor<512xi32, #blocked>
%cst_2 = arith.constant dense<128> : tensor<512xi32, #blocked>
%cst_3 = arith.constant dense<28672> : tensor<512xi32, #blocked>
%cst_4 = arith.constant dense<14336> : tensor<512xi32, #blocked>
%cst_5 = arith.constant dense<14400> : tensor<512xi32, #blocked>
```

after 的诊断 dump 已经是常量规范化后的形态（所有常量统一声明、按值从小到大排列）。

## 说明

本 kernel 不会触发 TMem aref 插入，原因是：

1. 没有 warp 专化分区（AutomaticWarpSpecialization 已判定无需专化）；
2. 没有 Tensor Memory 读写（无 WGMMA/TMA 类操作）；
3. 4 次全局内存 load 和 1 次 store 均为普通 `ld.global` / `st.global`。

NVWSInsertTmemAref 在本 kernel 上是完全空操作（no-op）。after 文件中 constants 顺序的变化是诊断 dump 的格式差异，不影响实际编译结果。该 Pass 是 Blackwell sm_120 编译路径的强制节点，对所有 kernel 都会过一遍；对于非矩阵 kernel 其效果等同于 no-op。
