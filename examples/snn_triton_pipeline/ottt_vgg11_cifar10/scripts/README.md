# OTTT VGG-11-WS — 脚本说明

A100 真源路径：`~/charlley/snn_infer_triton/`（路径为 A100 绝对路径，仅供在 A100 复跑）。
环境 `triton-src`（torch 2.12 + 源码 triton 3.7 + spikingjelly 源码版）。权重 `lsf/checkpoints/ottt_vgg11_cifar10/cifar10_ottta.pth`。
CIFAR-10 测试集（无损）：`~/charlley/snn_infer/data/cifar10/cifar10_test.pt`（从 HF parquet 解码，10000 张 uint8）。

| 文件 | 作用 |
|------|------|
| `ottt_vgg_triton.py` | 把 OTTT 在线单步 `OnlineLIFNode` VGG-11-WS **重写**为多步 spikingjelly LIF(triton)；含 `ScaledWSConv2d`（权重标准化）与检查点按位置重映射（`features.{0\|N.op}.{weight,bias,gain}`、`classifier.0.op`）。 |
| `run_ottt.py` | 评测入口：`--n --bs --compile --profile --amp {none,bf16,fp16} --triton-conv`。神经元后端由 `SJ_NEURON_BACKEND` 控制（torch/triton）。 |
| `_common/snn2_lib.py` | CIFAR-10 无损 eval（`_PtCifar`）+ 模型无关 kernel profiler + `setup_inductor_triton`（关 cudnn、GEMM=TRITON、conv_1x1_as_mm、compile_threads=1）。 |
| `_common/verify_equiv.py` | 用同一权重对比"原 OnlineLIFNode 循环 T 次" vs 重写多步模型（argmax 全一致）。`SJ_NEURON_BACKEND=torch` 运行。 |
| `_common/capture_ir2.py` | 截获 IR：`python capture_ir2.py ottt` + `TRITON_CACHE_DIR/TORCHINDUCTOR_CACHE_DIR/TORCH_COMPILE_DEBUG_DIR`。 |
| `_common/{sj_compat,snn_eval_lib_triton,snn_eval_lib,timm_compat,capture_ir}.py` | 与其它模型共享（内容一致）。 |

复跑：
```bash
conda activate triton-src; cd ~/charlley/snn_infer_triton
CIFAR_DIR=~/charlley/snn_infer/data/cifar10 SJ_NEURON_BACKEND=triton python run_ottt.py --n 10000 --bs 200            # 神经元 triton, 93.60%
CIFAR_DIR=~/charlley/snn_infer/data/cifar10 SJ_NEURON_BACKEND=triton python run_ottt.py --n 2000 --bs 200 --compile --profile --amp bf16 --triton-conv  # 整网 triton, 100%
```
