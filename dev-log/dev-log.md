# 开发笔记

在 32GB 内存的机器上构建 Triton，16 会占用 20GB 左右内存，如果不加限制会导致机器 OOM 直接卡死。
```bash
MAX_JOBS=16 make dev-install
```

运行实例代码：
```bash
TRITON_ALWAYS_COMPILE=1 python test_triton.py
```

使用 cuda-12.8 时，需要在`triton/third_party/nvidia/backend/compiler.py`中的`get_ptx_version_from_options`函数把`return`强行设定为 87，在 Nvidia 官方的 PTX 指令集规范（PTX ISA）中，关于 Blackwell 架构的特有寄存器、指令和 Target 声明，是在 PTX 8.7 版本才被首次引入的，导致低于 87 就不适配 GPU。而高于 87 又由于 GPU 570.169 驱动对应的底层 API 版本是 CUDA 12.8。在 Nvidia 的发版时间线里，CUDA 12.8 驱动内置的 JIT 编译器，最高只能理解到 PTX 8.7 的语法。

使用cuda-13.1时，自动下载的 LLVM 预构建不能用，导致 triton 不能构建，是预构建 LLVM 与 Conda GCC/glibc 不兼容 导致的链接错误。
Triton 构建时会自动从 `~/.triton/llvm/` 下载一个预编译的 LLVM/MLIR（commit 如 87717bf9），这个包是用较老的 GCC（通常 11~13）+ 较老 glibc 编译的。而你当前 triton-dev-cuda131 环境的 GCC 15.2.0（conda 默认）会生成引用 __libc_single_threaded（glibc 2.32+ 引入的符号）的代码，导致链接时在老 LLVM static lib 中找不到定义。
CUDA 12.8 环境能成功，是因为那个环境的 GCC 版本较老，生成的代码没有这些新符号。
直接执行`make dev-install-llvm`，拉取的`llvm-project`最新源代码无法自动构建（暂未明原因，需要手动构建LLVM）。
Conda 环境里的编译器不是普通的 gcc / g++，而是带完整 triplet 前缀的 x86_64-conda-linux-gnu-gcc / g++（这是 conda-forge 安装 gcc 时的标准方式）。需要指定 gcc 和 g++。在 triton 根目录的 llvm-project 目录（triton构建自动拉取到这里）下对 LLVM 进行构建。

```bash
cd llvm-project/build

cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="llvm;mlir;lld;clang" \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PIC=ON \
  -DCMAKE_C_COMPILER=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ \
  -DCMAKE_AR=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-ar \
  -DCMAKE_RANLIB=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-ranlib \
  -DCMAKE_ASM_COMPILER=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc \
  ../llvm

ninja -j12
```

在 32GB 内存的机器上构建 LLVM，使用`ninja -j$(nproc)`会导致 OOM 机器卡死，指定 j12 时内存峰值可达 20GB+，可以正常构建。

再构建并安装 triton（在triton根目录中执行）：

```bash
cd /home/charlley/Code/Triton-Pass-Analysis/triton

# 1. 物理消灭之前的构建缓存，重要！
rm -rf build/ python/build/ python/triton/_C/ ~/.triton/cache

# 2. 导出 LLVM 路径，并把它加入系统的 PATH 中
export LLVM_SYSPATH=/home/charlley/Code/Triton-Pass-Analysis/triton/llvm-project/build
export PATH=$LLVM_SYSPATH/bin:$PATH

# 3. 强制指定编译器为我们刚编译的 clang/clang++
export CC=$LLVM_SYSPATH/bin/clang
export CXX=$LLVM_SYSPATH/bin/clang++
export TRITON_BUILD_WITH_CLANG_LLD=true

# 4. ★ 核心修复 1：告诉 Clang 和 LLD 链接器去 Conda 里找 -lz 和头文件
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=$CONDA_PREFIX/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include:$CPLUS_INCLUDE_PATH

# 5. ★ 核心修复 2：把缓存路径提升为全局环境变量，防止 CMake 重启时丢失
export TRITON_HOME=$HOME
export TRITON_CACHE_PATH=$HOME/.triton/cache

# 6. 只编译 NVIDIA 后端，省时省力
export TRITON_CODEGEN_BACKENDS=nvidia
export CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$CONDA_PREFIX"

# 7. ★ 核心破局点：强行将 Conda 的 lib 目录塞入 CMake 的底层链接器指令中
export LDFLAGS="-L$CONDA_PREFIX/lib"
export CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DCMAKE_SHARED_LINKER_FLAGS=-L$CONDA_PREFIX/lib -DCMAKE_EXE_LINKER_FLAGS=-L$CONDA_PREFIX/lib"


# 8. 终极执行
MAX_JOBS=16 python3 -m pip install -e . --no-build-isolation -v
```

NVIDIA DRIVER 570.169 不支持新的代码:
```
RuntimeError: The NVIDIA driver on your system is too old (found version 12080). Please update your GPU driver by downloading and installing a new version from the URL:
```
于是换用了 NVIDIA DRIVER 595.71.05。旧驱动运行不了新CUDA的代码。CUDA从12.8升级到13.1就出发了这个问题。

## 安装 nir / nirtorch / spikingjelly（editable，从仓库 submodule）

仓库根目录已把 `nir`（neuromorphs/NIR）、`nirtorch`（neuromorphs/NIRTorch）、`spikingjelly`（PKU/MLG）三者都纳入 submodule，便于分析源码与本地修改。装成 editable：

```bash
conda activate triton-dev-cuda131
# 务必加 --no-deps：见下方 ⚠️
pip install --no-deps -e /home/charlley/Code/Triton-Pass-Analysis/nir \
                     -e /home/charlley/Code/Triton-Pass-Analysis/nirtorch \
                     -e /home/charlley/Code/Triton-Pass-Analysis/spikingjelly
```

⚠️ **大坑：本 env 里几乎任何 `pip install`（不带 `--no-deps`）都会把本地 editable 的 `triton 3.7.0+gitXXXX`（我们的 fork）卸掉，换成 PyPI 上的 `triton 3.6.0`。**

实际踩过两次：
1. `pip install -e ./nir -e ./nirtorch` → 卸掉 triton fork
2. `pip install -e ./spikingjelly`（哪怕被装的包本身和 triton 无关）→ 又卸掉 triton fork

根因：本地 `triton-3.7.0+gitef02d646` 含 `+local` 版本后缀。**只要 pip 解析到的依赖链里出现 `triton`（torch 2.11 把它声明为 runtime dep），pip resolver 就会判定本地版"不符合常规约束"，强行 downgrade 到 PyPI 上能直接拉到的 `3.6.0`**，并装上对应 wheel；pip 输出里会显式打印：

```
Attempting uninstall: triton
  Found existing installation: triton 3.7.0+gitef02d646
  Uninstalling triton-3.7.0+gitef02d646
Successfully installed ... triton-3.6.0 ...
```

被替换后，本仓库基于自定义 Triton Pass 的所有工作流（`TRITON_ALWAYS_COMPILE=1`、SNN Pass、autotune=TRITON 等）全部失效。

**预防：** 在本 env 里装任何 editable / 第三方包时都加 `--no-deps`，让 pip 只动当前包不要碰其他东西（包括 triton）。需要的 runtime 依赖（torch、numpy、h5py、tqdm、tensorboard、pydantic 等）单独 `pip install` 一遍即可，那次允许 pip 拉依赖但显式把 triton 排除掉，例如：

```bash
pip install h5py tqdm tensorboard pydantic SciencePlots
```

**事后回滚 triton fork：**

```bash
# 1. 先卸掉 PyPI 那个 3.6.0
pip uninstall -y triton

# 2. 走完整重建脚本（fast-path 不可用：跳过 clean 时 cmake 报
#    "TRITON_CACHE_PATH must be set or derivable" —— build 目录内的旧
#    CMakeCache.txt 会和新一轮 -D 参数冲突，目前必须全量重建）
bash /home/charlley/Code/Triton-Pass-Analysis/dev-log/build_triton.sh
```

完事用 `python -c "import triton, sys; print(triton.__version__ if hasattr(triton, '__version__') else 'no attr'); import pkg_resources; print(pkg_resources.get_distribution('triton').version)"` 验，应该看到 `3.7.0+git<hash>` 而不是 `3.6.0`。或直接 `pip show triton | grep -i version`。