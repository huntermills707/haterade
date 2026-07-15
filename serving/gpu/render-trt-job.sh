#!/usr/bin/env bash
# Render the standalone TensorRT build Job manifest to stdout.
# Regenerate serving/gpu/trt-build-job.yaml with:
#   ./serving/gpu/render-trt-job.sh > serving/gpu/trt-build-job.yaml
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/trt-job.sh"

cat <<'HEADER'
# Build the TensorRT plan inside the same Triton image that serves it.
#
# Why not host trtexec? The repo originally targeted Triton 23.05 (TensorRT 8.6),
# but Ubuntu 24.04 workstations often end up with TensorRT 10/11 host packages,
# whose engine format is not backward-compatible with Triton 23.05's TensorRT
# backend. Building inside the serving container guarantees the plan and runtime
# use the exact same TensorRT version.
#
# Usage:
#   1. Export ONNX to the PVC first (see serving/gpu/export_onnx.py).
#   2. kubectl apply -f serving/gpu/trt-build-job.yaml
#   3. kubectl logs -f job/trt-build-distilbert-toxicity
#
# NOTE: This file is generated from serving/gpu/lib/trt-job.sh via
#       ./serving/gpu/render-trt-job.sh. Do not edit manually.
HEADER

render_trt_build_job \
  "trt-build-distilbert-toxicity" \
  "triton-model-repo" \
  "distilbert-toxicity" \
  128 \
  32
