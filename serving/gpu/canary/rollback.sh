#!/usr/bin/env bash
# Roll the GPU predictor back to the stable v1 rollout.
#
# Aborts any in-progress Argo Rollouts canary and re-applies the stable
# manifest (triton-model-repo / version 1). Traffic returns to 100% stable.
set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROLLOUT_NAME="toxicity-gpu"
SCALEDOBJECT="toxicity-gpu"

has_rollouts_plugin() {
  command -v kubectl-argorollouts >/dev/null 2>/dev/null || kubectl argo rollouts version >/dev/null 2>/dev/null
}

abort_rollout() {
  echo "==> Aborting any in-progress canary"
  if has_rollouts_plugin; then
    kubectl argo rollouts abort "$ROLLOUT_NAME" || true
  else
    echo "    kubectl argo rollouts plugin not found; patching abort flag"
    kubectl patch rollout "$ROLLOUT_NAME" --type merge -p '{"spec":{"abort":true}}' || true
  fi
}

apply_stable() {
  echo "==> Reapplying stable manifests (v1 model repo)"
  kubectl apply -f "$SCRIPT_DIR/../rollout.yaml"
  kubectl apply -f "$SCRIPT_DIR/../scaledobject.yaml"
  kubectl apply -f "$SCRIPT_DIR/../triton-servicemonitor.yaml"
  kubectl apply -f "$SCRIPT_DIR/../analysis-template.yaml"
  echo "==> Resuming KEDA autoscaling"
  kubectl annotate scaledobject "$SCALEDOBJECT" "autoscaling.keda.sh/paused-replicas-" --overwrite 2>/dev/null || true
}

wait_healthy() {
  local wait_seconds="${1:-300}"
  echo "==> Waiting for stable rollout to be Healthy (${wait_seconds}s)"
  if has_rollouts_plugin; then
    kubectl argo rollouts status "$ROLLOUT_NAME" --timeout "${wait_seconds}s"
    return 0
  fi

  local deadline=$((SECONDS + wait_seconds))
  while [ $SECONDS -lt $deadline ]; do
    phase=$(kubectl get rollout "$ROLLOUT_NAME" -o jsonpath='{.status.phase}' 2>/dev/null || echo "")
    echo "    $(date -Iseconds) phase=$phase"
    if [ "$phase" = "Healthy" ]; then
      echo "    rollout is Healthy"
      return 0
    fi
    sleep 10
  done
  return 1
}

abort_rollout
apply_stable

if wait_healthy 300; then
  exit 0
fi

echo ""
echo "==> Rollout still not Healthy; deleting and recreating stable Rollout"
kubectl delete rollout "$ROLLOUT_NAME" --ignore-not-found=true
apply_stable
wait_healthy 300
