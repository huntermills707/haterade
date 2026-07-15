#!/usr/bin/env bash
# M5 canary promotion orchestrator for the CPU predictor.
#
# 1. Validates the candidate MLflow run against the production AUROC gate.
# 2. Builds a version-2 Triton model repository in the canary PVC.
# 3. Applies the canary Rollout and waits for it to become Healthy or Degraded.
# 4. Optionally completes the Argo Rollouts promotion and marks the MLflow
#    model version as Production.
#
# Usage:
#   ./serving/cpu/canary/promote-and-canary.sh <run-id>
#   ./serving/cpu/canary/promote-and-canary.sh <run-id> --promote
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ROLLOUT_NAME="toxicity-cpu"
PROMOTE=false

usage() {
  echo "Usage: $0 <mlflow-run-id> [--promote]"
  exit 1
}

if [ $# -lt 1 ]; then usage; fi
RUN_ID="$1"
shift || true

while [ $# -gt 0 ]; do
  case "$1" in
    --promote) PROMOTE=true ;;
    -h|--help) usage ;;
    *) echo "unknown arg: $1"; usage ;;
  esac
  shift
done

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
VENV="$REPO_ROOT/training/.venv/bin/python"

echo "==> [1/4] Validating candidate run $RUN_ID against production gate"
"$VENV" -m training.src.promotion validate --run-id "$RUN_ID"

echo ""
echo "==> [2/4] Building canary v2 model repository from run $RUN_ID"
"$SCRIPT_DIR/build-canary-from-run.sh" "$RUN_ID"

echo ""
echo "==> [3/4] Applying canary Rollout + ServiceMonitor"
kubectl apply -f "$SCRIPT_DIR/servicemonitor.yaml"
kubectl apply -f "$SCRIPT_DIR/rollout-v2.yaml"

wait_for_rollout() {
  echo ""
  echo "==> [4/4] Waiting for rollout to complete (Healthy or Degraded)"
  local deadline=$((SECONDS + 900))  # 15 minute timeout

  # After applying rollout-v2.yaml the controller may briefly still report
  # the old stable Healthy state. Wait until it leaves Healthy so we don't
  # treat the pre-canary state as the final result.
  echo "    waiting for rollout to leave the stable Healthy state..."
  while [ $SECONDS -lt $deadline ]; do
    phase=$(kubectl get rollout "$ROLLOUT_NAME" -o jsonpath='{.status.phase}' 2>/dev/null || echo "")
    case "$phase" in
      Progressing|Degraded|Aborted|Paused)
        echo "    rollout started: phase=$phase"
        break
        ;;
      "")
        echo "    rollout status not yet available"
        ;;
    esac
    sleep 5
  done

  echo "    polling .status.phase"
  while [ $SECONDS -lt $deadline ]; do
    phase=$(kubectl get rollout "$ROLLOUT_NAME" -o jsonpath='{.status.phase}' 2>/dev/null || echo "")
    echo "    $(date -Iseconds) phase=$phase"
    case "$phase" in
      Healthy)
        echo "    rollout is Healthy"
        return 0
        ;;
      Degraded|Aborted)
        echo "    rollout is $phase"
        return 1
        ;;
      "")
        echo "    rollout not found or status unavailable"
        ;;
    esac
    sleep 15
  done

  echo "    timeout waiting for rollout"
  return 1
}

has_rollouts_plugin() {
  command -v kubectl-argorollouts >/dev/null 2>/dev/null || kubectl argo rollouts version >/dev/null 2>/dev/null
}

if wait_for_rollout; then
  if [ "$PROMOTE" = true ]; then
    echo ""
    echo "==> Promoting canary to stable + marking MLflow version Production"
    if has_rollouts_plugin; then
      kubectl argo rollouts promote "$ROLLOUT_NAME"
    else
      echo "ERROR: --promote requires the kubectl argo rollouts plugin." >&2
      echo "       Install it from https://argo-rollouts.readthedocs.io/en/stable/installation/#kubectl-plugin-installation" >&2
      echo "       or promote manually with: kubectl argo rollouts promote $ROLLOUT_NAME" >&2
      exit 1
    fi
    "$VENV" -m training.src.promotion promote --run-id "$RUN_ID"
    echo "    promoted. Traffic is now 100% on the new stable ReplicaSet."
  else
    echo ""
    echo "==> Canary is Healthy. To promote manually:"
    echo "    kubectl argo rollouts promote $ROLLOUT_NAME"
    echo "    $0 $RUN_ID --promote"
  fi
else
  echo ""
  echo "==> Canary did not reach Healthy. Rollback with:"
  echo "    $REPO_ROOT/serving/cpu/rollback.sh"
  exit 1
fi
