#!/usr/bin/env python3
"""
full_trace.py
=============
使用 sys.settrace 进行完整的函数调用追踪，捕获从 torch.compile 到 Triton
编译器优化 Pass 的全部关键函数调用栈。

重点追踪以下调用链：
  torch.compile()
    → TorchDynamo (convert_frame, _compile_inner)
    → AOTAutograd (aot_module_simplified, aot_dispatch_autograd)
    → TorchInductor (compile_fx, compile_fx_inner, codegen)
    → triton.compiler.compile (ASTSource.make_ir, ast_to_ttir)
    → make_ttir → make_ttgir → make_llir → make_ptx → make_cubin
"""

import sys
import os
import io
import time
import traceback
import inspect
import linecache

# ═══════════════════════════════════════════════════════════════════════════════
# 配置：关键函数白名单（只有这些函数会被详细记录）
# ═══════════════════════════════════════════════════════════════════════════════

# 需要完整记录的核心函数名
CRITICAL_FUNCS = {
    # --- torch.compile 层 ---
    "compile",
    "optimize",
    "_dynamo_inner_compile",
    # --- TorchDynamo 层 ---
    "convert_frame",
    "convert_frame_assert",
    "_compile_inner",
    "_optimize_catch_errors",
    "run_node",
    "create_graph",
    "compile_subgraph",
    "output",
    # --- AOTAutograd 层 ---
    "aot_module_simplified",
    "aot_module",
    "aot_function",
    "aot_dispatch_autograd",
    "aot_dispatch_base",
    "create_joint",
    "create_aot_dispatcher_function",
    "make_boxed_func",
    # --- TorchInductor 层 ---
    "compile_fx",
    "compile_fx_inner",
    "fx_codegen_and_compile",
    "codegen_and_compile",
    "schedule",
    "generate",
    # --- Triton Compiler 层 ---
    "make_ir",
    "ast_to_ttir",
    "make_ttir",
    "make_ttgir",
    "make_llir",
    "make_ptx",
    "make_cubin",
    "add_stages",
    # --- Triton Pass 注册函数 ---
    "add_coalesce",
    "add_coalesce_async_copy",
    "add_accelerate_matmul",
    "add_f32_dot_tc",
    "add_remove_layout_conversions",
    "add_optimize_thread_locality",
    "add_optimize_dot_operands",
    "add_prefetch",
    "add_pipeline",
    "add_schedule_loops",
    "add_fuse_nested_loops",
    "add_combine_tensor_select_and_if",
    "add_optimize_accumulator_init",
    "add_hoist_tmem_alloc",
    "add_reduce_data_duplication",
    "add_reorder_instructions",
    "add_convert_to_ttgpuir",
    "add_inliner",
    "add_combine",
    "add_canonicalizer",
    "add_cse",
    "add_symbol_dce",
    "add_loop_unroll",
    "add_allocate_shared_memory_nv",
    "add_to_llvmir",
}

# 需要监控的模块路径关键词（包含这些字符串的文件才被追踪）
TRACKED_PATHS = [
    "_dynamo/",
    "_functorch/",
    "_inductor/",
    "torch/fx/",
    "triton/compiler/",
    "triton/runtime/",
    "triton/backends/",
    "nvidia/backend/",
]

# ═══════════════════════════════════════════════════════════════════════════════
# 输出格式
# ═══════════════════════════════════════════════════════════════════════════════

output_lines = []

def emit(line, also_print=True):
    output_lines.append(line)
    if also_print:
        print(line)


def shorten_path(path):
    """截短路径：保留从 site-packages 或 triton/ 开始的部分"""
    for marker in ["site-packages/", "triton/python/triton/", "third_party/"]:
        idx = path.find(marker)
        if idx != -1:
            return path[idx + len(marker):]
    # fallback: 取最后几段
    parts = path.split("/")
    return "/".join(parts[-4:]) if len(parts) > 4 else path


# ═══════════════════════════════════════════════════════════════════════════════
# 追踪器
# ═══════════════════════════════════════════════════════════════════════════════

class CallTracer:
    def __init__(self):
        self.call_stack = []  # [(func_name, filename, lineno)]
        self.depth = 0
        self.events = []      # 顺序记录的调用事件

    def _is_tracked(self, filename):
        if not filename:
            return False
        for kw in TRACKED_PATHS:
            if kw in filename:
                return True
        return False

    def _get_short_loc(self, filename, lineno):
        return f"{shorten_path(filename)}:{lineno}"

    def trace_func(self, frame, event, arg):
        filename  = frame.f_code.co_filename
        func_name = frame.f_code.co_name
        lineno    = frame.f_lineno

        if not self._is_tracked(filename):
            return self.trace_func

        if event == "call":
            self.depth += 1
            loc = self._get_short_loc(filename, lineno)

            # 所有被追踪模块的调用都记录到事件列表
            evt = {
                "type": "call",
                "func": func_name,
                "loc": loc,
                "depth": self.depth,
                "filename": filename,
                "lineno": lineno,
            }
            self.events.append(evt)

            # 关键函数：立即打印
            if func_name in CRITICAL_FUNCS:
                indent = "  " * min(self.depth - 1, 20)
                line = f"{indent}>>> [{func_name}]  @ {loc}"
                emit(line)

        elif event == "return":
            if self.depth > 0:
                self.depth -= 1

        return self.trace_func


# ═══════════════════════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    emit("=" * 78)
    emit("  Triton-Pass-Analysis: 完整函数调用链追踪")
    emit(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    emit("=" * 78)
    emit("")

    # 1. 导入依赖
    emit("[1/6] 导入 PyTorch 和 SpikingJelly ...")
    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import neuron, layer, surrogate

    emit(f"      PyTorch  = {torch.__version__}")
    emit(f"      CUDA     = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        emit(f"      GPU      = {torch.cuda.get_device_name(0)}")
    emit("")

    # 2. 定位关键源文件
    emit("[2/6] 定位 torch.compile 相关源文件 ...")
    try:
        import torch._dynamo.eval_frame as ef
        emit(f"      torch._dynamo.eval_frame : {inspect.getfile(ef)}")
    except:
        pass
    try:
        import torch._dynamo.convert_frame as cf
        emit(f"      torch._dynamo.convert_frame: {inspect.getfile(cf)}")
        _compile_inner_fn = getattr(cf, "_compile_inner", None) or getattr(cf, "convert_frame", None)
        if _compile_inner_fn:
            emit(f"        -> _compile_inner @ line {inspect.getsourcelines(_compile_inner_fn)[1]}")
    except Exception as e:
        emit(f"      [WARN] {e}")
    try:
        import torch._functorch.aot_autograd as aa
        emit(f"      torch._functorch.aot_autograd: {inspect.getfile(aa)}")
        for fn_name in ["aot_module_simplified", "aot_dispatch_autograd"]:
            fn = getattr(aa, fn_name, None)
            if fn:
                emit(f"        -> {fn_name} @ line {inspect.getsourcelines(fn)[1]}")
    except Exception as e:
        emit(f"      [WARN] {e}")
    try:
        import torch._inductor.compile_fx as cfx
        emit(f"      torch._inductor.compile_fx: {inspect.getfile(cfx)}")
        for fn_name in ["compile_fx", "compile_fx_inner"]:
            fn = getattr(cfx, fn_name, None)
            if fn:
                emit(f"        -> {fn_name} @ line {inspect.getsourcelines(fn)[1]}")
    except Exception as e:
        emit(f"      [WARN] {e}")
    try:
        import triton.compiler.compiler as tcc
        emit(f"      triton.compiler.compiler: {inspect.getfile(tcc)}")
        fn = getattr(tcc, "compile", None)
        if fn:
            emit(f"        -> compile @ line {inspect.getsourcelines(fn)[1]}")
    except Exception as e:
        emit(f"      [WARN] {e}")
    try:
        import triton.backends.nvidia.compiler as ncc
        emit(f"      triton.backends.nvidia.compiler: {inspect.getfile(ncc)}")
        for fn_name in ["make_ttir", "make_ttgir", "make_llir", "make_ptx", "make_cubin"]:
            fn = getattr(ncc, fn_name, None)
            if fn:
                emit(f"        -> {fn_name} @ line {inspect.getsourcelines(fn)[1]}")
        # 尝试查找 pass 注册函数
        for fn_name in ["add_coalesce", "add_accelerate_matmul", "add_remove_layout_conversions",
                        "add_optimize_thread_locality", "add_prefetch"]:
            fn = getattr(ncc, fn_name, None)
            if fn:
                emit(f"        -> {fn_name} @ line {inspect.getsourcelines(fn)[1]}")
    except Exception as e:
        emit(f"      [WARN] {e}")
    emit("")

    # 3. 构建模型
    emit("[3/6] 构建 SimpleSNN 模型 ...")

    class SimpleSNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv    = layer.Conv2d(1, 16, kernel_size=3, padding=1)
            self.bn      = layer.BatchNorm2d(16)
            self.lif     = neuron.LIFNode(surrogate_function=surrogate.ATan())
            self.pool    = layer.MaxPool2d(2, 2)
            self.flatten = layer.Flatten()
            self.fc      = layer.Linear(16 * 14 * 14, 10)
            self.lif2    = neuron.LIFNode(surrogate_function=surrogate.ATan())

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.lif(x)
            x = self.pool(x)
            x = self.flatten(x)
            x = self.fc(x)
            x = self.lif2(x)
            return x

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = SimpleSNN().to(device)
    emit(f"      模型构建完成, device = {device}")
    emit("")

    # 4. torch.compile
    emit("[4/6] >>> torch.compile(model) ...")
    emit(f"      torch.compile 源文件: {inspect.getfile(torch.compile)}")
    emit(f"      torch.compile 行号  : {inspect.getsourcelines(torch.compile)[1]}")
    emit("")

    tracer = CallTracer()
    sys.settrace(tracer.trace_func)

    compiled_model = torch.compile(model, backend="inductor")

    sys.settrace(None)
    emit("")
    emit("    [torch.compile() 调用完成 - 仅注册编译器，尚未触发实际编译]")
    emit("")

    # 5. 第一次前向传播（触发实际编译）
    x = torch.randn(4, 1, 28, 28, device=device)

    emit("[5/6] >>> compiled_model(x)  [触发完整编译链] ...")
    emit("")

    tracer2 = CallTracer()
    sys.settrace(tracer2.trace_func)

    out = compiled_model(x)

    sys.settrace(None)
    emit("")
    emit(f"    前向传播完成. Output shape: {out.shape}")
    emit("")

    # 6. 反向传播
    emit("[6/6] >>> loss.backward() ...")
    tracer3 = CallTracer()
    sys.settrace(tracer3.trace_func)

    loss = out.sum()
    loss.backward()

    sys.settrace(None)
    emit(f"    反向传播完成.")
    emit("")

    # ─── 汇总输出 ──────────────────────────────────────────────────────────
    emit("=" * 78)
    emit("  关键函数调用汇总 (按实际调用顺序，前向传播阶段)")
    emit("=" * 78)
    emit("")

    all_events = tracer2.events
    critical_events = [e for e in all_events if e["func"] in CRITICAL_FUNCS]

    for i, evt in enumerate(critical_events, 1):
        indent = "  " * min(evt["depth"] // 2, 15)
        emit(f"{i:3d}. {indent}[{evt['func']}]")
        emit(f"       {evt['loc']}")
        emit("")

    emit("")
    emit("=" * 78)
    emit(f"  统计: 共追踪到 {len(all_events)} 个函数调用事件 (所有受监控模块)")
    emit(f"        其中关键函数调用: {len(critical_events)} 个")
    emit("=" * 78)

    # 保存完整日志
    out_dir  = os.path.dirname(os.path.abspath(__file__))
    out_file = os.path.join(out_dir, "full_callstack_output.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    emit(f"\n  完整日志已保存至: {out_file}")


if __name__ == "__main__":
    main()
