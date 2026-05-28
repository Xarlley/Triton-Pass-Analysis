"""验证: 把 NIR 路径返回的 fx.GraphModule 也包 torch.compile + 全 Triton 配置，
即可达到与 vgg16_test.py (path B) 完全一致的「dynamo graph_break=0 + 无 extern_kernels」
全 Triton 路径。

用法：
    # 基本运行
    python examples/vgg16_snn/nir_compile_test.py

    # 抓 Inductor output_code 做完整审计（推荐）
    TORCH_LOGS=output_code python examples/vgg16_snn/nir_compile_test.py \\
        > /tmp/nir_compile_audit.log 2>&1
    bash Document/Skill/audit-full-triton-path.md 里的 grep 命令对 /tmp/nir_compile_audit.log 跑一遍

实测结论（RTX 5070 Ti, BATCH=1, T=4）：首次 49.4s 编译，10 项全 Triton 审计全过。
"""
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch._dynamo
import torch._inductor.config as inductor_cfg
from spikingjelly.activation_based import functional, neuron, nir_exchange
from spikingjelly.activation_based.functional.conv_bn_fusion import (
    fuse_conv_bn_eval_modules,
)

# ============ 全 Triton 编译配置（复用 vgg16_test.py 的配方）============
# 与 path B (vgg16_test.py) 的 configure_full_triton_compilation() 等价
torch._dynamo.config.recompile_limit = 256
torch._dynamo.config.cache_size_limit = 256
inductor_cfg.max_autotune = True
inductor_cfg.max_autotune_gemm_backends = "TRITON"
inductor_cfg.max_autotune_conv_backends = "TRITON"
inductor_cfg.force_disable_caches = True

# 注意：NIR 路径不需要 patch_spikingjelly_for_full_graph()
# 该 patch 是给 layer.BatchNorm2d.seq_to_ann_forward 用的，但 NIR 路径已经 fold-BN，
# 网络里根本没有 BN 层 —— 无 isinstance 判定，无 graph_break 触发点。

print("[config] full-Triton compile config applied")

# ============ 构造与 vgg16_via_nir.py 等价的网络 ============
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
    nn.Sequential(nn.Flatten(),
                  nn.Linear(512 * 7 * 7, 4096), neuron.LIFNode(step_mode="s"),
                  nn.Linear(4096, 4096), neuron.LIFNode(step_mode="s"),
                  nn.Linear(4096, 1000))).eval()
folded = fuse_conv_bn_eval_modules(model)
graph = nir_exchange.export_to_nir(folded, example_input=torch.rand(1, 3, 224, 224), dt=1e-4)
gm = nir_exchange.import_from_nir(graph, dt=1e-4, device="cuda", step_mode="m")
gm.eval()
print("[build] NIR roundtrip OK")

# ============ 关键 1 行：把 NIR 返回的 fx.GraphModule 喂给 torch.compile ============
compiled = torch.compile(gm)

# ============ 跑一次 forward，触发编译 + autotune ============
x = torch.randn(4, 1, 3, 224, 224, device="cuda")
print("[run] first forward (will compile + autotune, ~50-120s)...")
t0 = time.perf_counter()
with torch.no_grad():
    out = compiled(x)
    if isinstance(out, tuple):
        out = out[0]
torch.cuda.synchronize()
print(f"[run] first forward done in {time.perf_counter()-t0:.1f}s, out shape={tuple(out.shape)}")

# ============ 审计 dynamo graph_break ============
from torch._dynamo.utils import counters
n_break = sum(counters.get("graph_break", {}).values())
print(f"[result] dynamo graph_break count = {n_break}  (0 = single graph)")
print(f"[result] dynamo counters keys: {list(counters.keys())[:10]}")
if "graph_break" in counters:
    print(f"[result] graph_break detail: {dict(counters['graph_break'])}")
