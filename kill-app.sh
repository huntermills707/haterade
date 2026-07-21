#!/usr/bin/env bash
# Tear down EVERYTHING this project installed on the cluster: the toxicity app
# (pods, GPU, RAM) AND the platform stack (Istio, KServe, KEDA, Argo Rollouts,
# cert-manager, Prometheus/Grafana, MLflow, MinIO). The k3s cluster itself is
# left running, so you can deploy something else right away.
#
# Not touched: namespaces that don't belong to this project (mlops,
# kuberay-system, kube-system, ...). Model PVCs in $NAMESPACE are kept unless
# --purge-pvcs is given.
#
# Usage:
#   ./kill-app.sh                  # app + platform teardown (k3s stays up)
#   ./kill-app.sh --purge-pvcs     # also drop the model PVCs
#   ./kill-app.sh --app-only       # only the app; keep the platform stack for
#                                  # a fast redeploy (old default behavior)
#
# Safe to re-run: every delete uses --ignore-not-found / || true.
set -euo pipefail

NAMESPACE="${NAMESPACE:-default}"
PURGE_PVCS=0
APP_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --purge-pvcs) PURGE_PVCS=1 ;;
    --app-only)   APP_ONLY=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

# ----------------------------------------------------------------------------
# App teardown (namespace $NAMESPACE)
# ----------------------------------------------------------------------------
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

if [ "$APP_ONLY" -eq 1 ]; then
  echo ""
  echo "Done (app only). Remaining app resources in $NAMESPACE:"
  kubectl get rollout,deploy,svc,vs,scaledobject,analysistemplate,servicemonitor,pvc -n "$NAMESPACE" 2>&1 || true
  exit 0
fi

# ----------------------------------------------------------------------------
# Platform teardown (everything infra/install-platform-stack.sh installed)
# ----------------------------------------------------------------------------
# Helm releases. NOTE: minio is only uninstalled from the mlflow namespace —
# the minio release in mlops belongs to another deployment.
echo "==> Uninstalling platform Helm releases"
helm uninstall argo-rollouts         -n argo-rollouts  2>/dev/null || true
helm uninstall keda                  -n keda           2>/dev/null || true
helm uninstall kube-prometheus-stack -n observability  2>/dev/null || true
helm uninstall minio                 -n mlflow         2>/dev/null || true
helm uninstall istio-ingress         -n istio-ingress  2>/dev/null || true
helm uninstall istiod                -n istio-system   2>/dev/null || true
helm uninstall istio-base            -n istio-system   2>/dev/null || true
helm uninstall cert-manager          -n cert-manager   2>/dev/null || true

echo "==> Removing KServe storage Secret and Istio injection label from $NAMESPACE"
kubectl delete secret storage-config -n "$NAMESPACE" --ignore-not-found=true
kubectl label ns "$NAMESPACE" istio-injection- 2>/dev/null || true

echo "==> Deleting platform namespaces (KServe, MLflow, MinIO, Grafana, ...)"
# Namespace deletion also sweeps up the operator-created StatefulSets
# (Prometheus, Alertmanager) and the manifest-installed MLflow deployment.
kubectl delete ns \
  argo-rollouts keda observability mlflow kserve cert-manager \
  istio-ingress istio-system \
  --ignore-not-found=true --timeout=180s || true

echo "==> Deleting leftover cluster-scoped resources (CRDs, webhooks)"
# Helm uninstall leaves CRDs behind (charts ship them from crds/). Deleting
# the CRDs cascades to any remaining custom resources. --timeout + || true so
# a stuck finalizer can't wedge the script.
kubectl get crd -o name 2>/dev/null \
  | grep -E 'serving\.kserve\.io|keda\.sh|istio\.io|cert-manager\.io|argoproj\.io|monitoring\.coreos\.com' \
  | xargs -r kubectl delete --ignore-not-found=true --timeout=120s || true
kubectl get mutatingwebhookconfiguration,validatingwebhookconfiguration -o name 2>/dev/null \
  | grep -Ei 'istio|kserve|cert-manager|keda|kube-prometheus' \
  | xargs -r kubectl delete --ignore-not-found=true || true

echo ""
echo "Done. k3s is still running; platform namespaces are gone."
echo "Remaining namespaces:"
kubectl get ns
