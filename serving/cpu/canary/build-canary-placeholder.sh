#!/usr/bin/env bash
# Build the M4 placeholder v2 model repository for the canary predictor.
#
# This intentionally does NOT retrain a model. It copies the verified v1 ONNX
# from the stable PVC, places it under Triton version "2", and pins the
# canary config to serve only version 2. The result is a distinct serving
# artifact that exercises the canary traffic split while M5 handles the real
# retrain.
#
# Usage:
#   ./serving/cpu/canary/build-canary-placeholder.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_NAME="distilbert-toxicity"
SRC_PVC="triton-cpu-model-repo"
DST_PVC="triton-cpu-canary-model-repo"
HELPER_POD="triton-cpu-canary-cp"

echo "==> Creating canary PVC (if not exists)"
kubectl apply -f "$SCRIPT_DIR/model-pvc.yaml"

echo "==> Launching helper pod with stable and canary PVCs"
kubectl -n default delete pod "$HELPER_POD" --ignore-not-found 2>/dev/null || true
kubectl -n default run "$HELPER_POD" --image=busybox --restart=Never \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "'"$HELPER_POD"'",
        "image": "busybox",
        "command": ["sleep", "300"],
        "volumeMounts": [
          {"name": "src-repo", "mountPath": "/mnt/src"},
          {"name": "dst-repo", "mountPath": "/mnt/dst"}
        ]
      }],
      "volumes": [
        {"name": "src-repo", "persistentVolumeClaim": {"claimName": "'"$SRC_PVC"'"}},
        {"name": "dst-repo", "persistentVolumeClaim": {"claimName": "'"$DST_PVC"'"}}
      ]
    }
  }'

kubectl -n default wait "pod/$HELPER_POD" --for=condition=Ready --timeout=120s

echo "==> Copying v1 model repo into canary repo"
kubectl -n default exec "$HELPER_POD" -- sh -c '
  rm -rf /mnt/dst/*
  mkdir -p /mnt/dst/'"$MODEL_NAME"'
  cp -r /mnt/src/'"$MODEL_NAME"'/* /mnt/dst/'"$MODEL_NAME"'/
'

echo "==> Promoting model to version 2 in canary repo"
kubectl -n default exec "$HELPER_POD" -- sh -c '
  cd /mnt/dst/'"$MODEL_NAME"'
  cp -r 1 2
  # Pin the canary config to serve only version 2. This makes the canary
  # observably different in Triton metrics (version="2") without changing
  # the inference contract.
  if ! grep -q "version_policy" config.pbtxt; then
    echo "" >> config.pbtxt
    echo "version_policy: { specific: { versions: [2] } }" >> config.pbtxt
  fi
'

echo "==> Verifying canary repository layout"
kubectl -n default exec "$HELPER_POD" -- find "/mnt/dst/$MODEL_NAME" -type f | sort

echo "==> Cleaning up helper pod"
kubectl -n default delete pod "$HELPER_POD" --ignore-not-found

echo ""
echo "Done. Canary placeholder v2 is on PVC $DST_PVC."
echo "Deploy with: kubectl apply -f $SCRIPT_DIR/"
