# MS-ResNet-104 — 脚本说明

A100 真源路径：`~/charlley/snn_infer_triton/`（A100 绝对路径，仅供在 A100 复跑）。
环境 `triton-src`。权重 `lsf/checkpoints/ms_resnet104_imagenet/resnet104.pth`。数据集：复用 `~/charlley/snn_infer/data/val_flat`（ImageNet val）。

| 文件 | 作用 |
|------|------|
| `ms_resnet_triton.py` | 把原 MS-ResNet（纯 torch 时间循环 `mem_update`）**重写**为 spikingjelly LIF(triton)：monkeypatch `mem_update→LIFNode(tau=4/3,decay_input=False,vth=0.5,vreset=0,m,triton)`、`Snn_Conv2d→批量卷积`；TDBN(`batch_norm_2d`) 原样保留。检查点 0 missing/0 unexpected。导入原仓库 `repos/MS-ResNet/models/MS_ResNet.py`。 |
| `run_msresnet.py` | 评测入口：`--n --bs --compile --profile --amp --triton-conv`；ImageNet eval 复用 `_common/snn_eval_lib_triton.py`，inductor 配置用 `_common/snn2_lib.py`。 |
| `_common/snn2_lib.py` | `setup_inductor_triton` + 模型无关 profiler（CIFAR 部分对本模型不用）。 |
| `_common/verify_equiv.py` | 神经元等价（原 `mem_update` vs LIF triton，**bit-exact**）+ 全模型加载校验。`SJ_NEURON_BACKEND=torch`。 |
| `_common/capture_ir2.py` | 截获 IR：`python capture_ir2.py msresnet` + cache/debug 环境变量。 |

复跑：
```bash
conda activate triton-src; cd ~/charlley/snn_infer_triton
SJ_NEURON_BACKEND=triton python run_msresnet.py --n 2000 --bs 25                                         # 神经元 triton, 74.20%
SJ_NEURON_BACKEND=triton python run_msresnet.py --n 500 --bs 25 --compile --profile --amp bf16 --triton-conv  # 整网 triton, 98.7%
```
