#!/bin/bash
# Push the local snn_compiler package to the A100 workspace (A100 has no rsync -> tar over ssh).
# Keeps snn_compiler standalone: only the package + explore scripts go over; nothing in triton/ is touched.
set -e
REPO=/home/charlley/Code/Triton-Pass-Analysis
cd "$REPO"
tar czf - --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' snn_compiler \
  | ssh -F ~/.ssh/config.a100 a100 'mkdir -p ~/charlley/snn_compiler_attn && tar xzf - -C ~/charlley/snn_compiler_attn'
echo "[push] snn_compiler -> a100:~/charlley/snn_compiler_attn/snn_compiler"
