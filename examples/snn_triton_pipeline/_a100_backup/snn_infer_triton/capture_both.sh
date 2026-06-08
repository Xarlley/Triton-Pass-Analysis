#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh; conda activate triton-src
cd ~/charlley/snn_infer_triton
for M in sf sdtv2; do
  D=capture/$M; rm -rf $D; mkdir -p $D/triton_cache $D/inductor_cache $D/debug
  echo "######## capturing $M ########"
  env SJ_NEURON_BACKEND=triton TRITON_ALWAYS_COMPILE=1 \
    TRITON_CACHE_DIR=$PWD/$D/triton_cache \
    TORCHINDUCTOR_CACHE_DIR=$PWD/$D/inductor_cache \
    TORCH_COMPILE_DEBUG=1 TORCH_COMPILE_DEBUG_DIR=$PWD/$D/debug \
    python capture_ir.py $M --bs 8 > $D/capture.log 2>&1
  echo "$M exit=$? : $(grep -c CAPTURE_DONE $D/capture.log) done-marker"
done
echo "CAPTURE_BOTH_DONE"
