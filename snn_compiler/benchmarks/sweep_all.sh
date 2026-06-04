#!/bin/bash
# Sweep all neuron / reset / dtype / layout combinations.
# Each runs about 30-60s; total ~10 minutes.

set -e
HERE="$(cd "$(dirname "$0")"; pwd)"
RESULTS="$HERE/sweep_results.jsonl"
rm -f "$RESULTS"

for NEURON in if lif; do
  for RESET in soft hard; do
    for MODE in fp32 bf16; do
      for LAYOUT in NCHW NHWC; do
        echo
        echo "=== NEURON=$NEURON RESET=$RESET MODE=$MODE LAYOUT=$LAYOUT ==="
        BATCH=32 T=4 TOTAL=500 WARMUP=3 \
          MODE=$MODE LAYOUT=$LAYOUT NEURON=$NEURON RESET=$RESET \
          python "$HERE/bench_vgg16.py" 2>&1 | tail -8
      done
    done
  done
done

mv "$HERE/results.jsonl" "$RESULTS"
echo
echo "[sweep complete] results at $RESULTS"
