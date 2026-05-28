"""真实运行 vgg16_via_nir.py 配方，捕获 NIR 路径上的函数调用栈。

产物落到 Document/IR-Trace/nir_lif_kernel/：
  - call_trace_build.txt    : export_to_nir + import_from_nir 阶段的 Python 调用栈（过滤后）
  - call_trace_forward.txt  : 一次 gm(x) forward 的 Python 调用栈（过滤后）
  - aten_ops.txt            : torch.profiler 抓到的 ATen 操作（含 cuDNN/cuBLAS 内核）
"""
import os, sys, pathlib

REPO = pathlib.Path("/home/charlley/Code/Triton-Pass-Analysis")
sys.path.insert(0, str(REPO / "examples/vgg16_snn"))
OUT = REPO / "Document/IR-Trace/nir_lif_kernel"
OUT.mkdir(parents=True, exist_ok=True)

# 过滤：仅记录这些库下的 Python 调用
TRACE_KEYWORDS = (
    "spikingjelly/spikingjelly/",
    "nirtorch/nirtorch/",
    "/nir/nir/",
    "/triton/python/triton/",
    "examples/vgg16_snn/",
)
# 强制忽略以下噪声路径（避免大量 dispatcher / autograd 帧）
TRACE_EXCLUDE = (
    "torch/_dynamo",
    "torch/fx/_symbolic_trace.py",  # fx tracing 内部走访每个节点会刷屏
    "torch/_decomp",
    "torch/_inductor",
    "torch/_higher_order_ops",
)
# 同名 helper 帧（list comp / dict comp 等）忽略
HELPER_NAMES = {"<listcomp>", "<dictcomp>", "<setcomp>", "<genexpr>", "<lambda>"}


class CallTracer:
    """记录每次进入 Python 函数的事件，按调用深度缩进输出。"""

    def __init__(self, output_path, max_lines=None):
        self.output_path = output_path
        self.max_lines = max_lines
        self.lines = []
        self._stop = False

    def _shorten(self, filename):
        s = filename
        for prefix in (
            "/home/charlley/Code/Triton-Pass-Analysis/",
            "/home/charlley/miniconda3/envs/triton-dev-cuda131/lib/python3.12/site-packages/",
        ):
            if s.startswith(prefix):
                s = s.replace(prefix, "", 1)
                break
        return s

    def _trace(self, frame, event, arg):
        if self._stop or event != "call":
            return self._trace
        filename = frame.f_code.co_filename
        funcname = frame.f_code.co_name
        # exclude helpers
        if funcname in HELPER_NAMES:
            return self._trace
        # exclude noise
        if any(k in filename for k in TRACE_EXCLUDE):
            return self._trace
        # include only relevant libs
        if not any(k in filename for k in TRACE_KEYWORDS):
            return self._trace
        # compute depth among already-traced frames
        depth = 0
        f = frame.f_back
        while f:
            fn = f.f_code.co_filename
            if (any(k in fn for k in TRACE_KEYWORDS)
                    and not any(k in fn for k in TRACE_EXCLUDE)
                    and f.f_code.co_name not in HELPER_NAMES):
                depth += 1
            f = f.f_back
        self.lines.append((depth, self._shorten(filename), funcname, frame.f_lineno))
        if self.max_lines and len(self.lines) >= self.max_lines:
            self._stop = True
        return self._trace

    def __enter__(self):
        sys.settrace(self._trace)
        return self

    def __exit__(self, *a):
        sys.settrace(None)
        with open(self.output_path, "w") as f:
            f.write(f"# 自动捕获的 Python 调用栈（按调用深度缩进）\n")
            f.write(f"# 过滤包含: {TRACE_KEYWORDS}\n")
            f.write(f"# 共 {len(self.lines)} 条调用事件\n\n")
            for indent, fn, name, lineno in self.lines:
                f.write(f"{'  ' * indent}{fn}:{lineno}  {name}()\n")


# === 复用 vgg16_via_nir.py 的模型构造逻辑 ===
import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, neuron, nir_exchange
from spikingjelly.activation_based.functional.conv_bn_fusion import (
    fuse_conv_bn_eval_modules,
)

torch.manual_seed(42)
VGG16_CFG = [64, 64, "P", 128, 128, "P", 256, 256, 256, "P",
             512, 512, 512, "P", 512, 512, 512, "P"]
feats, in_ch = [], 3
for v in VGG16_CFG:
    if v == "P":
        feats.append(nn.AvgPool2d(2, 2))
    else:
        feats.extend([nn.Conv2d(in_ch, v, 3, padding=1), nn.BatchNorm2d(v),
                      neuron.LIFNode(step_mode="s")])
        in_ch = v
model = nn.Sequential(
    nn.Sequential(*feats),
    nn.Sequential(
        nn.Flatten(),
        nn.Linear(512 * 7 * 7, 4096), neuron.LIFNode(step_mode="s"),
        nn.Linear(4096, 4096), neuron.LIFNode(step_mode="s"),
        nn.Linear(4096, 1000),
    ),
).eval()
folded = fuse_conv_bn_eval_modules(model)
example_input = torch.rand(1, 3, 224, 224)

# === Phase 1: 构造阶段调用栈 ===
print("[1/3] tracing build phase (export + import)...")
with CallTracer(OUT / "call_trace_build.txt", max_lines=20000) as t:
    graph = nir_exchange.export_to_nir(folded, example_input=example_input, dt=1e-4)
    gm = nir_exchange.import_from_nir(graph, dt=1e-4, device="cuda", step_mode="m")
gm.eval()
print(f"  wrote {OUT / 'call_trace_build.txt'}  ({len(t.lines)} call events)")

# === Phase 2: forward 阶段调用栈 ===
# 用小输入 + 单步避免 LIF kernel autotune 触发大量编译帧，让 forward 调用栈聚焦
print("[2/3] tracing forward phase (gm(x))...")
x = torch.randn(4, 1, 3, 224, 224, device="cuda")
# 预热一次让 Triton kernel 编译完成（编译过程本身的调用栈不是我们关注的）
with torch.no_grad():
    gm(x)
torch.cuda.synchronize()

# 真正采样
with torch.no_grad(), CallTracer(OUT / "call_trace_forward.txt", max_lines=2500) as t2:
    out = gm(x)
torch.cuda.synchronize()
print(f"  wrote {OUT / 'call_trace_forward.txt'}  ({len(t2.lines)} call events)")
print(f"  forward output shape: {tuple(out[0].shape) if isinstance(out, tuple) else tuple(out.shape)}")

# === Phase 3: ATen / CUDA kernel profile ===
print("[3/3] tracing ATen + CUDA kernels (torch.profiler)...")
from torch.profiler import profile, ProfilerActivity, record_function
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=False,
    with_stack=False,
) as prof:
    with record_function("gm_forward"), torch.no_grad():
        out = gm(x)
    torch.cuda.synchronize()

# 按 self_cuda_time_total 排序，输出前 50 个 op + kernel
table = prof.key_averages().table(
    sort_by="self_cuda_time_total", row_limit=60, header="Top ATen ops + CUDA kernels by self_cuda_time_total"
)
with open(OUT / "aten_ops.txt", "w") as f:
    f.write("# torch.profiler 抓到的 ATen ops + CUDA kernels（含 cuDNN/cuBLAS 内核）\n")
    f.write(f"# forward: gm({tuple(x.shape)}), eval, no_grad, BATCH=1, T=4\n\n")
    f.write(table)
print(f"  wrote {OUT / 'aten_ops.txt'}")

# 同时 chrome-trace 一份（更友好的可视化，体积小）
prof.export_chrome_trace(str(OUT / "chrome_trace.json"))
print(f"  wrote {OUT / 'chrome_trace.json'}")
print("done.")
