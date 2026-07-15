#!/usr/bin/env bash
# Build the M4 placeholder v2 model repository for the GPU canary predictor.
#
# Copies the verified v1 TensorRT plan from the stable PVC, places it under
# Triton version "2", and pins the canary config to serve only version 2.
# The result is a distinct serving artifact that exercises the canary traffic
# split while M5 handles the real retrain.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_NAME="distilbert-toxicity"
SRC_PVC="triton-model-repo"
DST_PVC="triton-gpu-canary-model-repo"
HELPER_POD="triton-gpu-canary-cp"

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
echo "Deploy with: kubectl apply -f $SCRIPT_DIR/rollout-v2.yaml"
