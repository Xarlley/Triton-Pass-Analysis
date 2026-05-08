#!/bin/bash
# 自动捕获 Triton 编译期间的所有 MLIR Passes

cd "$(dirname "$0")/.." || exit

echo "清理 Triton 缓存..."
rm -rf ~/.triton/cache/*

echo "运行示例并捕获所有 MLIR Dumps..."
# 使用 MLIR_ENABLE_DUMP=1 和 TRITON_ALWAYS_COMPILE=1
TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=1 bash run_example.sh > analysis/triton_passes/mlir_dump.log 2>&1

echo "捕获完成！结果保存在 analysis/triton_passes/mlir_dump.log 中。"
