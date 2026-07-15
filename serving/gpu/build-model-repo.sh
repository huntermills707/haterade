#!/usr/bin/env bash
# Build a Triton model repository for the GPU cluster: download the HF model
# from MLflow, export to ONNX with INT32 inputs, bake a TensorRT plan inside
# the Triton 23.05 serving container, and copy the result to the PVC.
#
# Why build inside the Triton container? The repo originally baked the plan on
# the host with trtexec, but workstations often end up with TensorRT 10/11
# packages while Triton 23.05 ships TensorRT 8.6. The engine format is not
# backward-compatible, so we build the plan with the exact trtexec version that
# will serve it.
#
# Prerequisites:
#   - MLflow port-forward: kubectl -n mlflow port-forward svc/mlflow 5000:5000
#   - training/.venv with torch, transformers, mlflow, onnx
#   - kubectl
#
# Usage:
#   MLFLOW_RUN_ID=715720fe79cb44178dfa65ef32da50eb ./serving/gpu/build-model-repo.sh
set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"

MLFLOW_RUN_ID="${MLFLOW_RUN_ID:-715720fe79cb44178dfa65ef32da50eb}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
SEQ_LEN="${SEQ_LEN:-128}"
MAX_BATCH="${MAX_BATCH:-32}"
NUM_LABELS="${NUM_LABELS:-6}"
MODEL_NAME="distilbert-toxicity"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PVC_NAME="triton-model-repo"
HELPER_POD="triton-gpu-build-v1"

export MLFLOW_TRACKING_URI

echo "==> Activating training venv"
source "$REPO_ROOT/training/.venv/bin/activate"

echo "==> Downloading HF model from MLflow run $MLFLOW_RUN_ID"
BUILD_DIR=$(mktemp -d)
cleanup() {
  local code=$?
  rm -rf "$BUILD_DIR"
  kubectl -n default delete pod "$HELPER_POD" job trt-build-gpu-v1 --ignore-not-found >/dev/null 2>&1 || true
  exit "$code"
}
trap cleanup EXIT

python3 - "$MLFLOW_RUN_ID" "$BUILD_DIR" "$MLFLOW_TRACKING_URI" <<'PY'
import sys, mlflow
run_id, dest, tracking_uri = sys.argv[1], sys.argv[2], sys.argv[3]
mlflow.set_tracking_uri(tracking_uri)
client = mlflow.MlflowClient()
model_path = client.download_artifacts(run_id, "model", dest)
tokenizer_path = client.download_artifacts(run_id, "tokenizer", dest)
print(f"model={model_path}\ntokenizer={tokenizer_path}")
PY

MODEL_DIR="$BUILD_DIR/model"
TOKENIZER_DIR="$BUILD_DIR/tokenizer"

# Newer M5 runs log the HF directory directly under the artifact path "model".
if [ -d "$MODEL_DIR/artifacts/model" ]; then
  MODEL_DIR="$MODEL_DIR/artifacts/model"
fi

echo "==> Exporting ONNX with INT32 inputs (seq_len=$SEQ_LEN, labels=$NUM_LABELS)"
python3 "$SCRIPT_DIR/export_onnx.py" \
  --model-uri "$MODEL_DIR" \
  --tokenizer-uri "$TOKENIZER_DIR" \
  --seq-len "$SEQ_LEN" \
  --num-labels "$NUM_LABELS" \
  --int32-inputs \
  --out "$BUILD_DIR/$MODEL_NAME/onnx/model.onnx"

echo "==> Copying config + ONNX to PVC $PVC_NAME"
kubectl -n default delete pod "$HELPER_POD" --ignore-not-found 2>/dev/null || true
kubectl -n default run "$HELPER_POD" --image=busybox --restart=Never \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "'"$HELPER_POD"'",
        "image": "busybox",
        "command": ["sleep", "300"],
        "volumeMounts": [
          {"name": "model-repo", "mountPath": "/mnt/models"}
        ]
      }],
      "volumes": [
        {"name": "model-repo", "persistentVolumeClaim": {"claimName": "'"$PVC_NAME"'"}}
      ]
    }
  }'
kubectl -n default wait "pod/$HELPER_POD" --for=condition=Ready --timeout=120s
kubectl -n default exec "$HELPER_POD" -- mkdir -p "/mnt/models/$MODEL_NAME/1" "/mnt/models/$MODEL_NAME/onnx"
kubectl cp "$SCRIPT_DIR/model-repository/$MODEL_NAME/config.pbtxt" "default/$HELPER_POD:/mnt/models/$MODEL_NAME/config.pbtxt"
kubectl cp "$BUILD_DIR/$MODEL_NAME/onnx/model.onnx" "default/$HELPER_POD:/mnt/models/$MODEL_NAME/onnx/model.onnx"

source "$REPO_ROOT/serving/gpu/lib/trt-job.sh"

echo "==> Building TensorRT plan inside Triton 23.05 container"
kubectl -n default delete job trt-build-gpu-v1 --ignore-not-found
render_trt_build_job "trt-build-gpu-v1" "$PVC_NAME" "$MODEL_NAME" "$SEQ_LEN" "$MAX_BATCH" | kubectl -n default apply -f -
kubectl -n default wait --for=condition=Complete job/trt-build-gpu-v1 --timeout=600s
kubectl -n default logs job/trt-build-gpu-v1 | tail -5
kubectl -n default delete job trt-build-gpu-v1 --ignore-not-found
kubectl -n default exec "$HELPER_POD" -- ls -lh "/mnt/models/$MODEL_NAME/1/"

echo "==> Verifying PVC contents"
kubectl -n default exec "$HELPER_POD" -- find "/mnt/models/$MODEL_NAME" -type f | sort
kubectl -n default delete pod "$HELPER_POD" --ignore-not-found

echo ""
echo "Done. Model repository is on PVC $PVC_NAME."
echo "Deploy with: kubectl apply -f $SCRIPT_DIR/rollout.yaml"
echo "Canary v2:    ./serving/gpu/canary/build-canary-placeholder.sh"
