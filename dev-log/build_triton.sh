#!/usr/bin/env bash
# Full clean rebuild of triton into the triton-dev-cuda131 conda env.
# Follows dev-log/dev-log.md, with one deviation: the _C/ clean removes only
# stale .so files and preserves the tracked linear_layout.pyi type stub.
# NOTE: no `set -u` — conda's cuda-nvcc activation hook references unset vars.
set -eo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate triton-dev-cuda131
cd /home/charlley/Code/Triton-Pass-Analysis/triton

echo "=== [1/3] Cleaning previous build artifacts ==="
rm -rf build/ python/build/ "$HOME/.triton/cache"
rm -f python/triton/_C/*.so python/triton/_C/*.so.*
echo "Cleaned. linear_layout.pyi preserved:"
ls -la python/triton/_C/libtriton/ 2>/dev/null || true

echo "=== [2/3] Exporting build environment ==="
export LLVM_SYSPATH=/home/charlley/Code/Triton-Pass-Analysis/triton/llvm-project/build
export PATH=$LLVM_SYSPATH/bin:$PATH
export CC=$LLVM_SYSPATH/bin/clang
export CXX=$LLVM_SYSPATH/bin/clang++
export TRITON_BUILD_WITH_CLANG_LLD=true
export LIBRARY_PATH=$CONDA_PREFIX/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export C_INCLUDE_PATH=$CONDA_PREFIX/include:${C_INCLUDE_PATH:-}
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include:${CPLUS_INCLUDE_PATH:-}
export TRITON_HOME=$HOME
export TRITON_CACHE_PATH=$HOME/.triton/cache
export TRITON_CODEGEN_BACKENDS=nvidia
export LDFLAGS="-L$CONDA_PREFIX/lib"
export CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DCMAKE_SHARED_LINKER_FLAGS=-L$CONDA_PREFIX/lib -DCMAKE_EXE_LINKER_FLAGS=-L$CONDA_PREFIX/lib"
echo "LLVM_SYSPATH=$LLVM_SYSPATH"
echo "CC=$CC"
echo "CONDA_PREFIX=$CONDA_PREFIX"

echo "=== [3/3] Building + installing (MAX_JOBS=16) ==="
MAX_JOBS=16 python3 -m pip install -e . --no-build-isolation -v

echo "=== BUILD SCRIPT FINISHED OK ==="
