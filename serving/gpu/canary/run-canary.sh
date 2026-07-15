#!/usr/bin/env bash
# Run the GPU canary to completion, handling the 2-GPU-node HPA conflict.
#
# Problem: KEDA/HPA wants 2 replicas, but Argo Rollouts canary with
# setCanaryScale:1 + 2-GPU node cannot fit stable=2 + canary=1. We pause KEDA
# at 1 replica during the canary, then resume autoscaling after promotion.
#
# Usage:
#   ./run-canary.sh
#
# After success the canary model repo is promoted to the stable PVC and the
# Rollout points back at the stable PVC.
set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"

NAMESPACE="default"
ROLLOUT="toxicity-gpu"
SCALEDOBJECT="toxicity-gpu"
SEQ_LEN=128
STABLE_PVC="triton-model-repo"
CANARY_PVC="triton-gpu-canary-model-repo"
MODEL_NAME="distilbert-toxicity"
HELPER_POD="triton-gpu-promote-cp"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The training venv has transformers.
source "$SCRIPT_DIR/../../../training/.venv/bin/activate"

echo "==> Pausing KEDA at 1 replica so Argo Rollouts controls the canary"
kubectl -n "$NAMESPACE" annotate scaledobject "$SCALEDOBJECT" "autoscaling.keda.sh/paused-replicas=1" --overwrite

# Wait for KEDA to scale the rollout down to 1.
echo "==> Waiting for KEDA to settle to 1 replica"
for i in $(seq 1 30); do
  READY=$(kubectl -n "$NAMESPACE" get rollout "$ROLLOUT" -o jsonpath='{.status.availableReplicas}' 2>/dev/null || echo 0)
  if [ "${READY:-0}" -eq 1 ]; then
    echo "Rollout is at 1 replica"
    break
  fi
  sleep 5
done

echo "==> Applying canary Rollout (v2 PVC)"
kubectl -n "$NAMESPACE" apply -f "$SCRIPT_DIR/rollout-v2.yaml"

# Start a background load generator so the canary receives requests and the
# Prometheus analysis gates have metrics to evaluate.
echo "==> Starting background load generator"
GATEWAY_IP=$(kubectl -n istio-ingress get svc istio-ingress -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
python3 - "$SEQ_LEN" <<'PY' > /tmp/canary-payload.json
import json, sys
from transformers import AutoTokenizer
seq_len = int(sys.argv[1])
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
enc = tok("you are a worthless idiot", padding="max_length", truncation=True, max_length=seq_len)
print(json.dumps({
    "inputs": [
        {"name": "input_ids",      "shape": [1, seq_len], "datatype": "INT32", "data": enc["input_ids"]},
        {"name": "attention_mask", "shape": [1, seq_len], "datatype": "INT32", "data": enc["attention_mask"]},
    ]
}))
PY

LOAD_PIDS=()
REQUEST_INTERVAL="${REQUEST_INTERVAL:-0.1}"
for i in $(seq 1 20); do
  (
    while true; do
      curl -sS --connect-timeout 5 --max-time 10 \
        -X POST "http://$GATEWAY_IP/v2/models/distilbert-toxicity/infer" \
        -H "Content-Type: application/json" \
        -H "Host: $ROLLOUT-$NAMESPACE.example.com" \
        -d @/tmp/canary-payload.json >/dev/null 2>&1 || true
      sleep "$REQUEST_INTERVAL"
    done
  ) &
  LOAD_PIDS+=("$!")
done

cleanup_load() {
  for pid in "${LOAD_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup_load EXIT

echo "==> Watching canary progress (max 30 min)"
for minute in $(seq 1 30); do
  sleep 60
  WEIGHTS=$(kubectl -n "$NAMESPACE" get virtualservice "$ROLLOUT" -o jsonpath='{.spec.http[0].route[*].weight}' 2>/dev/null || echo "?")
  PHASE=$(kubectl -n "$NAMESPACE" get rollout "$ROLLOUT" -o jsonpath='{.status.phase}' 2>/dev/null || echo "?")
  echo "minute $minute: weights=$WEIGHTS phase=$PHASE"

  if [ "$PHASE" = "Healthy" ] || [ "$PHASE" = "Completed" ]; then
    echo "Canary completed successfully"
    break
  fi
  if [ "$PHASE" = "Degraded" ] || [ "$PHASE" = "Aborted" ]; then
    echo "Canary failed (phase=$PHASE). Run ./rollback.sh and investigate."
    exit 1
  fi
done

PHASE=$(kubectl -n "$NAMESPACE" get rollout "$ROLLOUT" -o jsonpath='{.status.phase}' 2>/dev/null || echo "?")
if [ "$PHASE" != "Healthy" ] && [ "$PHASE" != "Completed" ]; then
  echo "ERROR: canary did not complete (phase=$PHASE). Run ./rollback.sh and investigate."
  exit 1
fi

echo "==> Promoting canary model repo to stable PVC"
kubectl -n "$NAMESPACE" delete pod "$HELPER_POD" --ignore-not-found 2>/dev/null || true
kubectl -n "$NAMESPACE" run "$HELPER_POD" --image=busybox --restart=Never \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "'"$HELPER_POD"'",
        "image": "busybox",
        "command": ["sleep", "300"],
        "volumeMounts": [
          {"name": "stable-repo", "mountPath": "/mnt/stable"},
          {"name": "canary-repo", "mountPath": "/mnt/canary"}
        ]
      }],
      "volumes": [
        {"name": "stable-repo", "persistentVolumeClaim": {"claimName": "'"$STABLE_PVC"'"}},
        {"name": "canary-repo", "persistentVolumeClaim": {"claimName": "'"$CANARY_PVC"'"}}
      ]
    }
  }'
kubectl -n "$NAMESPACE" wait "pod/$HELPER_POD" --for=condition=Ready --timeout=120s
kubectl -n "$NAMESPACE" exec "$HELPER_POD" -- sh -c '
  rm -rf /mnt/stable/'"$MODEL_NAME"'
  cp -r /mnt/canary/'"$MODEL_NAME"' /mnt/stable/'"$MODEL_NAME"'
  # Stable serves the latest (best) version; remove the canary version pin.
  sed -i "/version_policy/d" /mnt/stable/'"$MODEL_NAME"'/config.pbtxt
'
kubectl -n "$NAMESPACE" exec "$HELPER_POD" -- find "/mnt/stable/$MODEL_NAME" -type f | sort
kubectl -n "$NAMESPACE" delete pod "$HELPER_POD" --ignore-not-found

echo "==> Pointing Rollout back at stable PVC"
kubectl -n "$NAMESPACE" patch rollout "$ROLLOUT" --type=merge -p '{"spec":{"template":{"spec":{"volumes":[{"name":"model-repo","persistentVolumeClaim":{"claimName":"'"$STABLE_PVC"'"}}]}}}}'

echo "==> Resuming KEDA autoscaling"
kubectl -n "$NAMESPACE" annotate scaledobject "$SCALEDOBJECT" "autoscaling.keda.sh/paused-replicas-" --overwrite

echo "==> Waiting for rollout to stabilize"
kubectl -n "$NAMESPACE" wait --for=condition=Ready pod -l app=toxicity-gpu --timeout=300s

echo "==> Done. Final state:"
kubectl -n "$NAMESPACE" get rollout "$ROLLOUT"
kubectl -n "$NAMESPACE" get scaledobject "$SCALEDOBJECT" -o wide
