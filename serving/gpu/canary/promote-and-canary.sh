#!/usr/bin/env bash
# M5 GPU promotion pipeline:
#   1. Fetch the latest Staging model from MLflow.
#   2. Export ONNX + build TensorRT plan into the canary PVC.
#   3. Run the Argo Rollouts canary.
#
# Usage:
#   ./promote-and-canary.sh
#
# Requires: kubectl, helm (for MLflow access), python training venv, Kaggle token.
set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CANARY_PVC="triton-gpu-canary-model-repo"
MODEL_NAME="distilbert-toxicity"
EXPERIMENT_NAME="${MLFLOW_EXPERIMENT:-toxicity-distilbert}"
REGISTERED_MODEL_NAME="${MLFLOW_REGISTERED_MODEL_NAME:-distilbert-toxicity}"
AUROC_THRESHOLD="${AUROC_THRESHOLD:-0.90}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
SEQ_LEN=128
MAX_BATCH=32
HELPER_POD="triton-gpu-build-canary"

export MLFLOW_TRACKING_URI

# -----------------------------------------------------------------------------
# 1. Identify latest Staging run and validate
# -----------------------------------------------------------------------------
echo "==> Activating training venv"
source "$REPO_ROOT/training/.venv/bin/activate"

echo "==> Looking up latest successful run in MLflow experiment '$EXPERIMENT_NAME'"
RUN_INFO=$(python3 - "$EXPERIMENT_NAME" "$AUROC_THRESHOLD" "$MLFLOW_TRACKING_URI" "$REGISTERED_MODEL_NAME" <<'PY'
import sys, mlflow
exp_name, threshold, tracking_uri, registered_name = sys.argv[1:5]
threshold = float(threshold)
mlflow.set_tracking_uri(tracking_uri)
client = mlflow.MlflowClient()
exp = client.get_experiment_by_name(exp_name)
if not exp:
    raise SystemExit(f"experiment {exp_name} not found")
runs = client.search_runs(exp.experiment_id, order_by=["metrics.auroc_macro DESC"], max_results=1)
if not runs:
    raise SystemExit("no runs found")
run = runs[0]
auroc = run.data.metrics.get("auroc_macro", 0.0)
print(f"run_id={run.info.run_id} auroc={auroc}", file=sys.stderr)
if auroc < threshold:
    raise SystemExit(f"auroc {auroc} below threshold {threshold}")
# Compare against current Production model, if any.
prod_run_id, prod_auroc = None, None
try:
    prod_versions = client.get_latest_versions(registered_name, stages=["Production"])
    if prod_versions:
        prod = prod_versions[0]
        prod_run_id = prod.run_id
        prod_run = client.get_run(prod_run_id)
        prod_auroc = prod_run.data.metrics.get("auroc_macro")
except mlflow.exceptions.MlflowException:
    pass
print(f"production_run={prod_run_id} production_auroc={prod_auroc}", file=sys.stderr)
if prod_auroc is not None and auroc <= prod_auroc:
    raise SystemExit(
        f"candidate auroc {auroc} is not better than production auroc {prod_auroc}; "
        "skipping promotion"
    )
print(run.info.run_id)
PY
)

RUN_ID="$RUN_INFO"
CANDIDATE_AUROC=$(python3 - "$RUN_ID" "$MLFLOW_TRACKING_URI" <<'PY'
import sys, mlflow
run_id, tracking_uri = sys.argv[1], sys.argv[2]
mlflow.set_tracking_uri(tracking_uri)
run = mlflow.MlflowClient().get_run(run_id)
print(run.data.metrics.get("auroc_macro", 0.0))
PY
)

echo "Promoting run $RUN_ID (auroc $CANDIDATE_AUROC)"

# -----------------------------------------------------------------------------
# 2. Download artifacts to a temp directory
# -----------------------------------------------------------------------------
BUILD_DIR=$(mktemp -d)
cleanup() {
  local code=$?
  rm -rf "$BUILD_DIR"
  kubectl -n default delete pod "$HELPER_POD" job trt-build-canary --ignore-not-found >/dev/null 2>&1 || true
  exit "$code"
}
trap cleanup EXIT

echo "==> Downloading model artifacts to $BUILD_DIR"
python3 - "$RUN_ID" "$BUILD_DIR" "$MLFLOW_TRACKING_URI" <<'PY'
import sys, mlflow, os
run_id, dest, tracking_uri = sys.argv[1], sys.argv[2], sys.argv[3]
mlflow.set_tracking_uri(tracking_uri)
client = mlflow.MlflowClient()
model_path = client.download_artifacts(run_id, "model", dest)
tokenizer_path = client.download_artifacts(run_id, "tokenizer", dest)
print(f"model={model_path}\ntokenizer={tokenizer_path}")
PY

MODEL_DIR="$BUILD_DIR/model"
TOKENIZER_DIR="$BUILD_DIR/tokenizer"

# -----------------------------------------------------------------------------
# 3. Export ONNX and build TensorRT plan
# -----------------------------------------------------------------------------
echo "==> Exporting ONNX"
python3 "$REPO_ROOT/serving/gpu/export_onnx.py" \
  --model-uri "$MODEL_DIR" \
  --tokenizer-uri "$TOKENIZER_DIR" \
  --seq-len "$SEQ_LEN" \
  --num-labels 6 \
  --int32-inputs \
  --out "$BUILD_DIR/$MODEL_NAME/onnx/model.onnx"

echo "==> Copying config + ONNX to canary PVC"
kubectl -n default delete pod "$HELPER_POD" --ignore-not-found 2>/dev/null || true
kubectl -n default run "$HELPER_POD" --image=busybox --restart=Never \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "'"$HELPER_POD"'",
        "image": "busybox",
        "command": ["sleep", "300"],
        "volumeMounts": [
          {"name": "canary-repo", "mountPath": "/mnt/canary"}
        ]
      }],
      "volumes": [
        {"name": "canary-repo", "persistentVolumeClaim": {"claimName": "'"$CANARY_PVC"'"}}
      ]
    }
  }'
kubectl -n default wait "pod/$HELPER_POD" --for=condition=Ready --timeout=120s
kubectl -n default exec "$HELPER_POD" -- mkdir -p "/mnt/canary/$MODEL_NAME/1" "/mnt/canary/$MODEL_NAME/onnx"
kubectl cp "$REPO_ROOT/serving/gpu/model-repository/$MODEL_NAME/config.pbtxt" "default/$HELPER_POD:/mnt/canary/$MODEL_NAME/config.pbtxt"
kubectl cp "$BUILD_DIR/$MODEL_NAME/onnx/model.onnx" "default/$HELPER_POD:/mnt/canary/$MODEL_NAME/onnx/model.onnx"
kubectl -n default exec "$HELPER_POD" -- ls -lhR "/mnt/canary/$MODEL_NAME"

# -----------------------------------------------------------------------------
# 4. Build TensorRT plan inside the Triton container via a Job
# -----------------------------------------------------------------------------
source "$REPO_ROOT/serving/gpu/lib/trt-job.sh"

echo "==> Building TensorRT plan in canary PVC"
kubectl -n default delete job trt-build-canary --ignore-not-found
render_trt_build_job "trt-build-canary" "$CANARY_PVC" "$MODEL_NAME" "$SEQ_LEN" "$MAX_BATCH" | kubectl -n default apply -f -
kubectl -n default wait --for=condition=Complete job/trt-build-canary --timeout=600s
kubectl -n default logs job/trt-build-canary | tail -5
kubectl -n default delete job trt-build-canary --ignore-not-found
kubectl -n default exec "$HELPER_POD" -- ls -lh "/mnt/canary/$MODEL_NAME/1/"

# Ensure canary config pins version 2 (placeholder promotion path)
kubectl -n default exec "$HELPER_POD" -- sh -c '
  cd /mnt/canary/'"$MODEL_NAME"'
  cp -r 1 2
  if ! grep -q "version_policy" config.pbtxt; then
    echo "" >> config.pbtxt
    echo "version_policy: { specific: { versions: [2] } }" >> config.pbtxt
  fi
'
kubectl -n default exec "$HELPER_POD" -- ls -lhR "/mnt/canary/$MODEL_NAME"
kubectl -n default delete pod "$HELPER_POD" --ignore-not-found

# -----------------------------------------------------------------------------
# 5. Run the canary rollout
# -----------------------------------------------------------------------------
echo "==> Starting canary rollout"
"$SCRIPT_DIR/run-canary.sh"

echo ""
echo "M5 GPU promotion complete. Run ./serving/gpu/query.sh to verify."
