#!/usr/bin/env bash
# Build a Triton model repository for the CPU cluster: download the HF model
# from MLflow, export to ONNX, assemble the repo, and copy to the PVC.
#
# Mirrors serving/gpu/build-engine.sh but skips the trtexec step — the ONNX
# file IS the deployable model on CPU.
#
# Prerequisites:
#   - MLflow port-forward: kubectl -n mlflow port-forward svc/mlflow 5000:5000
#   - training/.venv with torch, transformers, mlflow, onnx
#   - kubectl
#
# Usage:
#   MLFLOW_RUN_ID=4927d59563184da6a5861765de043394 ./serving/cpu/build-model-repo.sh
set -euo pipefail

MLFLOW_RUN_ID="${MLFLOW_RUN_ID:-4927d59563184da6a5861765de043394}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
SEQ_LEN="${SEQ_LEN:-128}"
NUM_LABELS="${NUM_LABELS:-6}"
MODEL_NAME="distilbert-toxicity"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORK_DIR="$(mktemp -d)"
EXPORT_SCRIPT="$REPO_ROOT/serving/gpu/export_onnx.py"
VENV="$REPO_ROOT/training/.venv/bin/python"

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

echo "==> Assembling Triton model repository"
mkdir -p "$WORK_DIR/repo/$MODEL_NAME/1"
cp "$WORK_DIR/model.onnx"                                  "$WORK_DIR/repo/$MODEL_NAME/1/model.onnx"
cp "$SCRIPT_DIR/model-repository/$MODEL_NAME/config.pbtxt" "$WORK_DIR/repo/$MODEL_NAME/config.pbtxt"

echo "==> Layout:"
find "$WORK_DIR/repo/$MODEL_NAME" -type f | sed 's/^/    /'

echo ""
echo "==> Creating PVC (if not exists)"
kubectl apply -f "$SCRIPT_DIR/model-pvc.yaml"
# local-path uses WaitForFirstConsumer — the PVC won't bind until a pod
# references it. The helper pod below triggers binding.

echo "==> Launching helper pod to trigger PVC binding + receive copy"
kubectl -n default delete pod triton-cpu-cp --ignore-not-found 2>/dev/null
kubectl -n default run triton-cpu-cp --image=busybox --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"triton-cpu-cp","image":"busybox","command":["sleep","300"],"volumeMounts":[{"name":"model-repo","mountPath":"/mnt/models"}]}],"volumes":[{"name":"model-repo","persistentVolumeClaim":{"claimName":"triton-cpu-model-repo"}}]}}'
kubectl -n default wait pod/triton-cpu-cp --for=condition=Ready --timeout=120s

# Clean previous model repo on the PVC
kubectl -n default exec triton-cpu-cp -- sh -c 'rm -rf /mnt/models/*' 2>/dev/null || true
kubectl -n default cp "$WORK_DIR/repo/$MODEL_NAME" "triton-cpu-cp:/mnt/models/$MODEL_NAME"

echo "==> Verifying PVC contents"
kubectl -n default exec triton-cpu-cp -- find /mnt/models -type f

echo "==> Cleaning up helper pod"
kubectl -n default delete pod triton-cpu-cp --ignore-not-found

echo ""
echo "Done. Model repository is on PVC triton-cpu-model-repo."
echo "Deploy with: kubectl apply -f $SCRIPT_DIR/rollout.yaml"
echo "Canary v2:    ./serving/cpu/canary/build-canary-placeholder.sh"
