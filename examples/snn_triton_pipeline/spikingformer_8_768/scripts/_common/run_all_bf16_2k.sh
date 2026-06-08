#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh; conda activate triton-src
cd ~/charlley/snn_infer_triton
export SJ_NEURON_BACKEND=triton
echo "######## SEW n=2000 bf16 compile ########"
python run_sew_triton.py          --n 2000 --bs 50 --compile --amp bf16 --triton-conv 2>&1 | grep -E "RESULT|compile_threads"
echo "######## Spikingformer n=2000 bf16 compile ########"
python run_spikingformer_triton.py --n 2000 --bs 50 --compile --amp bf16 --triton-conv 2>&1 | grep -E "RESULT|compile_threads"
echo "######## SDT-V2 n=2000 bf16 compile ########"
python run_sdtv2_triton.py         --n 2000 --bs 50 --compile --amp bf16 --triton-conv 2>&1 | grep -E "RESULT|compile_threads"
echo "ALL_DONE"
