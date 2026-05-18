"""
test_triton.py — 验证 SNN_FLAG 条件性触发 SNN Pass 的准备工作阶段测试。

运行方式（需在 triton-dev-cuda131 conda 环境中执行）：
    TRITON_ALWAYS_COMPILE=1 python test_triton.py

预期输出：
  - 第一个 kernel（普通 kernel，无 SNN_FLAG）：打印 "跳过 SNN Pass"
  - 第二个 kernel（SNN kernel，SNN_FLAG=True）：打印 "SNN Pass 插入到编译流水线" 并执行 MyNoOpPass
"""

import torch
import triton
print("当前加载的 Triton 路径:", triton.__file__)
import triton.language as tl


# =============================================================================
# Kernel 1：普通向量加法（不含 SNN_FLAG）
# 期望行为：SNN Pass 不被触发
# =============================================================================
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """普通向量加法 kernel，不含 SNN_FLAG，SNN Pass 不应被触发。"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


# =============================================================================
# Kernel 2：SNN 向量加法（含 SNN_FLAG: tl.constexpr）
# 期望行为：当 SNN_FLAG=True 时，SNN Pass 被触发
# =============================================================================
@triton.jit
def snn_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    SNN_FLAG: tl.constexpr,   # <-- SNN 标记参数，constexpr 类型，编译时常量
):
    """含 SNN_FLAG 的向量加法 kernel。SNN_FLAG=True 时，SNN 优化 Pass 被触发。"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def run_kernel(kernel_fn, x, y, output, size, extra_kwargs=None):
    """辅助函数：启动 kernel 并返回执行时间信息。"""
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']), )
    kwargs = dict(BLOCK_SIZE=256)
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    kernel_fn[grid](x, y, output, size, **kwargs)


def verify(output, expected, kernel_name):
    """验证 kernel 计算结果的正确性。"""
    max_diff = torch.max(torch.abs(output - expected)).item()
    status = "✅ 正确" if max_diff < 1e-5 else f"❌ 误差过大: {max_diff}"
    print(f"  [{kernel_name}] 与 PyTorch 原生结果最大误差: {max_diff:.2e}  {status}")


def main():
    size = 1024
    torch.manual_seed(42)
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    expected = x + y

    print("\n" + "=" * 70)
    print("测试 1：普通 kernel（无 SNN_FLAG）—— SNN Pass 应跳过")
    print("=" * 70)
    output_normal = torch.empty_like(x)
    run_kernel(add_kernel, x, y, output_normal, size)
    verify(output_normal, expected, "add_kernel")

    print("\n" + "=" * 70)
    print("测试 2：SNN kernel（SNN_FLAG=True）—— SNN Pass 应被触发")
    print("=" * 70)
    output_snn = torch.empty_like(x)
    run_kernel(snn_add_kernel, x, y, output_snn, size, extra_kwargs={"SNN_FLAG": True})
    verify(output_snn, expected, "snn_add_kernel (SNN_FLAG=True)")

    print("\n" + "=" * 70)
    print("测试 3：SNN kernel（SNN_FLAG=False）—— SNN Pass 应跳过")
    print("=" * 70)
    output_snn_off = torch.empty_like(x)
    run_kernel(snn_add_kernel, x, y, output_snn_off, size, extra_kwargs={"SNN_FLAG": False})
    verify(output_snn_off, expected, "snn_add_kernel (SNN_FLAG=False)")

    print("\n" + "=" * 70)
    print("所有测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()