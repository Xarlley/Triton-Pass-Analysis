#!/bin/bash
# run_example.sh
# 激活预置的 conda 环境 spiking_env 并执行 snn_example.py
# 导出 TORCH_LOGS="output_code" 以观察 Triton Kernel 源码生成

source /home/charlley/miniconda3/etc/profile.d/conda.sh
conda activate spiking_env

export TORCH_LOGS="output_code"
cd "$(dirname "$0")" || exit
python snn_example.py
