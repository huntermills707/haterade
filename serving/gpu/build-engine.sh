#!/usr/bin/env bash
# Build a TensorRT engine plan from a DistilBERT ONNX export and assemble the
# Triton model repository layout. Run on the GPU workstation (Turing sm_75).
#
# Output layout (what Triton expects):
#   <OUT_DIR>/<MODEL_NAME>/config.pbtxt
#   <OUT_DIR>/<MODEL_NAME>/1/model.plan
#
# The engine baked here is sm_75-bound. It will NOT run on the CPU kind cluster
# or on other GPU archs. See docs/adr/0001-use-kind-for-cpu-and-k3s-for-gpu.md
# (Risks section) and the planned CI matrix in a future ADR.
set -euo pipefail

MODEL_NAME="distilbert-toxicity"
SEQ_LEN="${SEQ_LEN:-128}"
MAX_BATCH="${MAX_BATCH:-32}"
# Dev default; override with a resolved MLflow artifact directory for production.
MODEL_URI="${MODEL_URI:-distilbert-base-uncased-finetuned-sst-2-english}"
OUT_DIR="${OUT_DIR:-./model-repo}"
WORK_DIR="${WORK_DIR:-$(mktemp -d)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for tool in trtexec python3; do
  command -v "$tool" >/dev/null 2>&1 || { echo "missing required tool: $tool"; exit 1; }
done
python3 -c "import torch, transformers" 2>/dev/null || {
  echo "install python deps: pip install torch transformers"
  exit 1
}

echo "==> GPU detected: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n1)"

echo "==> Exporting ONNX (dynamic batch axis, seq_len=$SEQ_LEN, source=$MODEL_URI)"
python3 "$SCRIPT_DIR/export_onnx.py" \
  --model-uri "$MODEL_URI" \
  --seq-len "$SEQ_LEN" \
  --out "$WORK_DIR/model.onnx"

echo "==> Baking TensorRT plan (fp16, batch 1..${MAX_BATCH} x $SEQ_LEN)"
trtexec \
  --onnx="$WORK_DIR/model.onnx" \
  --saveEngine="$WORK_DIR/model.plan" \
  --fp16 \
  --minShapes="input_ids:1x${SEQ_LEN},attention_mask:1x${SEQ_LEN}" \
  --optShapes="input_ids:${MAX_BATCH}x${SEQ_LEN},attention_mask:${MAX_BATCH}x${SEQ_LEN}" \
  --maxShapes="input_ids:${MAX_BATCH}x${SEQ_LEN},attention_mask:${MAX_BATCH}x${SEQ_LEN}"

echo "==> Assembling Triton model repository at $OUT_DIR"
mkdir -p "$OUT_DIR/$MODEL_NAME/1"
cp "$WORK_DIR/model.plan"                                  "$OUT_DIR/$MODEL_NAME/1/model.plan"
cp "$SCRIPT_DIR/model-repository/$MODEL_NAME/config.pbtxt" "$OUT_DIR/$MODEL_NAME/config.pbtxt"

echo "==> Layout:"
find "$OUT_DIR/$MODEL_NAME" -type f | sed 's/^/    /'

cat <<EOF

Engine baked on sm_75. Copy the OUT_DIR contents to the triton-model-repo PVC:
    kubectl cp $OUT_DIR/$MODEL_NAME default/triton-model-repo-pod:/mnt/models/

Or rebuild with OUT_DIR pointing at a mounted PVC path.
EOF
