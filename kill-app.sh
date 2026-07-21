#!/usr/bin/env bash
# Tear down the deployed toxicity app so its compute (pods, GPU, RAM) is freed
# for another deployment. Leaves the platform stack (Istio, KServe, KEDA, Argo
# Rollouts, MLflow, Prometheus) and the model PVCs intact, so redeploying is
# just `kubectl apply -f serving/{cpu,gpu}/rollout.yaml` again.
#
# Safe to run on either cluster and when nothing is deployed: every delete uses
# --ignore-not-found. CPU and GPU resources are listed together; whichever
# cluster you're on, only the matching ones exist and get removed.
#
# Usage:
#   ./kill-app.sh
#
# To also drop the model PVCs (forces a model-repo rebuild on redeploy):
#   ./kill-app.sh --purge-pvcs
set -euo pipefail

NAMESPACE="${NAMESPACE:-default}"
PURGE_PVCS=0
if [ "${1:-}" = "--purge-pvcs" ]; then
  PURGE_PVCS=1
fi

echo "==> Aborting any in-progress Argo Rollouts canaries"
for r in toxicity-cpu toxicity-gpu; do
  if kubectl get rollout "$r" -n "$NAMESPACE" >/dev/null 2>&1; then
    if command -v kubectl-argorollouts >/dev/null 2>&1 || kubectl argo rollouts version >/dev/null 2>&1; then
      kubectl argo rollouts abort "$r" -n "$NAMESPACE" || true
    else
      kubectl patch rollout "$r" -n "$NAMESPACE" --type merge -p '{"spec":{"abort":true}}' || true
    fi
  fi
done

echo "==> Deleting Argo Rollouts (this terminates the Triton pods)"
kubectl delete rollout -n "$NAMESPACE" \
  toxicity-cpu toxicity-gpu --ignore-not-found=true --timeout=120s

echo "==> Deleting KEDA ScaledObjects"
kubectl delete scaledobject -n "$NAMESPACE" \
  toxicity-cpu toxicity-gpu --ignore-not-found=true

echo "==> Deleting Services, VirtualServices, AnalysisTemplates, ServiceMonitors"
kubectl delete service -n "$NAMESPACE" \
  toxicity-cpu-stable toxicity-cpu-canary \
  toxicity-gpu-stable toxicity-gpu-canary \
  toxicity-ui --ignore-not-found=true
kubectl delete virtualservice -n "$NAMESPACE" \
  toxicity-cpu toxicity-gpu toxicity-ui --ignore-not-found=true
kubectl delete analysistemplate -n "$NAMESPACE" \
  toxicity-cpu-canary toxicity-gpu-canary --ignore-not-found=true
kubectl delete servicemonitor -n "$NAMESPACE" \
  triton-cpu-metrics triton-cpu-canary-metrics \
  triton-gpu-metrics triton-gpu-canary-metrics --ignore-not-found=true

echo "==> Deleting frontend Deployment"
kubectl delete deployment -n "$NAMESPACE" toxicity-ui --ignore-not-found=true

echo "==> Cleaning up any leftover helper pods"
kubectl delete pod -n "$NAMESPACE" \
  triton-cpu-canary-cp triton-gpu-canary-cp \
  --ignore-not-found=true

if [ "$PURGE_PVCS" -eq 1 ]; then
  echo "==> --purge-pvcs: deleting model PVCs"
  kubectl delete pvc -n "$NAMESPACE" \
    triton-cpu-model-repo triton-cpu-canary-model-repo \
    triton-gpu-model-repo triton-gpu-canary-model-repo \
    toxicity-ui-data --ignore-not-found=true
else
  echo "==> Keeping model PVCs (pass --purge-pvcs to drop them)"
fi

echo ""
echo "Done. Remaining app resources in $NAMESPACE:"
kubectl get rollout,deploy,svc,vs,scaledobject,analysistemplate,servicemonitor,pvc -n "$NAMESPACE" 2>&1 || true
