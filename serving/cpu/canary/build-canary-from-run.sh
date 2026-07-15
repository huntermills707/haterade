#!/usr/bin/env bash
# Build a Triton canary model repository (version 2) from an MLflow run.
#
# Downloads the HF model and tokenizer, exports to ONNX, places the model
# under Triton version "2" in the canary PVC, and pins the config to serve
# only version 2. This is the M5 replacement for the M4 placeholder script.
#
# Prerequisites:
#   - MLflow port-forward: kubectl -n mlflow port-forward svc/mlflow 5000:5000
#   - training/.venv with torch, transformers, mlflow, onnx
#   - kubectl
#
# Usage:
#   ./serving/cpu/canary/build-canary-from-run.sh <run-id>
#   MLFLOW_RUN_ID=<run-id> ./serving/cpu/canary/build-canary-from-run.sh
set -euo pipefail

if [ "${1:-}" ]; then
  MLFLOW_RUN_ID="$1"
fi

if [ -z "${MLFLOW_RUN_ID:-}" ]; then
  echo "Usage: $0 <mlflow-run-id>"
  echo "   or: MLFLOW_RUN_ID=<run-id> $0"
  exit 1
fi

MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
SEQ_LEN="${SEQ_LEN:-128}"
NUM_LABELS="${NUM_LABELS:-6}"
MODEL_NAME="distilbert-toxicity"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORK_DIR="$(mktemp -d)"
EXPORT_SCRIPT="$REPO_ROOT/serving/gpu/export_onnx.py"
VENV="$REPO_ROOT/training/.venv/bin/python"
CANARY_PVC="triton-cpu-canary-model-repo"
HELPER_POD="triton-cpu-canary-cp"

export MLFLOW_TRACKING_URI

echo "==> Downloading HF model from MLflow run $MLFLOW_RUN_ID"
"$VENV" -m mlflow artifacts download \
  --artifact-uri "runs:/${MLFLOW_RUN_ID}/model" \
  --dst-path "$WORK_DIR/mlflow-model"

# Older M2 runs logged a pyfunc artifact whose HF weights live under
# model/artifacts/model. Newer M5 runs log the HF directory directly under
# the artifact path "model". Support both layouts.
if [ -d "$WORK_DIR/mlflow-model/model/artifacts/model" ]; then
  HF_MODEL_DIR="$WORK_DIR/mlflow-model/model/artifacts/model"
elif [ -d "$WORK_DIR/mlflow-model/model" ]; then
  HF_MODEL_DIR="$WORK_DIR/mlflow-model/model"
else
  echo "ERROR: could not locate downloaded model artifacts in $WORK_DIR/mlflow-model"
  find "$WORK_DIR/mlflow-model" -type f | sed 's/^/  /'
  exit 1
fi

# The tokenizer is logged as a separate artifact.
echo "==> Downloading tokenizer from MLflow"
"$VENV" -m mlflow artifacts download \
  --artifact-uri "runs:/${MLFLOW_RUN_ID}/tokenizer" \
  --dst-path "$WORK_DIR/mlflow-tokenizer"

cp "$WORK_DIR"/mlflow-tokenizer/tokenizer/* "$HF_MODEL_DIR/"

echo "==> Exporting ONNX (seq_len=$SEQ_LEN, labels=$NUM_LABELS)"
"$VENV" "$EXPORT_SCRIPT" \
  --model-uri "$HF_MODEL_DIR" \
  --seq-len "$SEQ_LEN" \
  --num-labels "$NUM_LABELS" \
  --out "$WORK_DIR/model.onnx"

echo "==> Assembling canary Triton model repository (version 2)"
mkdir -p "$WORK_DIR/repo/$MODEL_NAME/2"
cp "$WORK_DIR/model.onnx"                                  "$WORK_DIR/repo/$MODEL_NAME/2/model.onnx"
cp "$SCRIPT_DIR/../model-repository/$MODEL_NAME/config.pbtxt" "$WORK_DIR/repo/$MODEL_NAME/config.pbtxt"

# Pin the canary to serve only version 2 so the AnalysisTemplate can filter
# Prometheus metrics on version="2".
if ! grep -q "version_policy" "$WORK_DIR/repo/$MODEL_NAME/config.pbtxt"; then
  echo "" >> "$WORK_DIR/repo/$MODEL_NAME/config.pbtxt"
  echo "version_policy: { specific: { versions: [2] } }" >> "$WORK_DIR/repo/$MODEL_NAME/config.pbtxt"
fi

echo "==> Layout:"
find "$WORK_DIR/repo/$MODEL_NAME" -type f | sed 's/^/    /'

echo ""
echo "==> Creating canary PVC (if not exists)"
kubectl apply -f "$SCRIPT_DIR/model-pvc.yaml"

echo "==> Launching helper pod to trigger PVC binding + receive copy"
kubectl -n default delete pod "$HELPER_POD" --ignore-not-found 2>/dev/null || true
kubectl -n default run "$HELPER_POD" --image=busybox --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"'"$HELPER_POD"'","image":"busybox","command":["sleep","300"],"volumeMounts":[{"name":"model-repo","mountPath":"/mnt/models"}]}],"volumes":[{"name":"model-repo","persistentVolumeClaim":{"claimName":"'"$CANARY_PVC"'"}}]}}'
kubectl -n default wait pod/"$HELPER_POD" --for=condition=Ready --timeout=120s

# Clean previous canary repo on the PVC
kubectl -n default exec "$HELPER_POD" -- sh -c 'rm -rf /mnt/models/*' 2>/dev/null || true
kubectl -n default cp "$WORK_DIR/repo/$MODEL_NAME" "$HELPER_POD:/mnt/models/$MODEL_NAME"

echo "==> Verifying canary PVC contents"
kubectl -n default exec "$HELPER_POD" -- find /mnt/models -type f

echo "==> Cleaning up helper pod"
kubectl -n default delete pod "$HELPER_POD" --ignore-not-found

echo ""
echo "Done. Canary v2 model repository is on PVC $CANARY_PVC."
echo "Start canary: kubectl apply -f $SCRIPT_DIR/rollout-v2.yaml"
