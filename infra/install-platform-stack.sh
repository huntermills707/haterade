#!/usr/bin/env bash
# Shared platform stack installer. Sourced from both cpu-cluster/bootstrap.sh
# and gpu-cluster/bootstrap.sh. The decision to share this stack across both
# clusters is documented in docs/adr/0001-use-kind-for-cpu-and-k3s-for-gpu.md.
#
# Expects: KUBECONFIG pointed at the target cluster, helm + kubectl + jq on PATH.
set -euo pipefail

# --- Versions verified working together on kind v1.34 / k3s v1.30 ---
ISTIO_VERSION="1.30.2"
KSERVE_VERSION="v0.19.0"
KEDA_CHART_VERSION="2.20.1"
ARGO_ROLLOUTS_CHART_VERSION="2.41.0"
KUBE_PROMETHEUS_STACK_CHART_VERSION="87.15.1"
MINIO_CHART_VERSION="5.4.0"
MLFLOW_IMAGE="ghcr.io/mlflow/mlflow:v2.20.3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Adding Helm repositories"
helm repo add istio                https://istio-release.storage.googleapis.com/charts
helm repo add kedacore             https://kedacore.github.io/charts
helm repo add argo                 https://argoproj.github.io/argo-helm
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add minio                https://charts.min.io/
helm repo update

# ----------------------------------------------------------------------------
echo "==> Istio (minimal profile; Argo Rollouts uses VirtualServices for split)"
kubectl create namespace istio-system  --dry-run=client -o yaml | kubectl apply -f -
kubectl create namespace istio-ingress --dry-run=client -o yaml | kubectl apply -f -
helm upgrade --install istio-base istio/base -n istio-system --version "$ISTIO_VERSION"
helm upgrade --install istiod    istio/istiod -n istio-system --version "$ISTIO_VERSION" \
  --set meshConfig.accessLogFile=true
helm upgrade --install istio-ingress istio/gateway -n istio-ingress --version "$ISTIO_VERSION"
kubectl wait --for=condition=Available deployment/istiod -n istio-system --timeout=180s

# ----------------------------------------------------------------------------
echo "==> KServe ${KSERVE_VERSION} (Standard mode — RawDeployment, no Knative)"
# KServe v0.19 ships a dedicated installer that handles the chicken-and-egg
# between CRDs/webhooks/controller. The raw kserve.yaml + kserve-cluster-resources
# pair has an apply-order issue: cluster-resources depend on a live webhook that
# doesn't exist until the controller runs, and the controller requires the
# namespace that kserve.yaml expects to already exist.
#
# The download can be flaky for unclear reasons (different failure point each
# run, same binary standalone works). Workaround: skip if the file already
# exists, and let the user pre-fetch it via:
#   curl -fL -o /tmp/kserve-install.sh \
#     https://github.com/kserve/kserve/releases/download/v0.19.0/kserve-standard-mode-full-install-with-manifests.sh
KSERVE_INSTALL_SCRIPT="${KSERVE_INSTALL_SCRIPT:-/tmp/kserve-install.sh}"
KSERVE_URL="https://github.com/kserve/kserve/releases/download/${KSERVE_VERSION}/kserve-standard-mode-full-install-with-manifests.sh"
if [ -s "$KSERVE_INSTALL_SCRIPT" ]; then
  echo "    reusing cached $KSERVE_INSTALL_SCRIPT"
else
  echo "    downloading to $KSERVE_INSTALL_SCRIPT"
  for attempt in 1 2 3; do
    if curl -fL -o "$KSERVE_INSTALL_SCRIPT" "$KSERVE_URL"; then
      break
    fi
    echo "    attempt $attempt failed, retrying in 3s..."
    sleep 3
  done
  [ -s "$KSERVE_INSTALL_SCRIPT" ] || { echo "failed to fetch KServe installer"; exit 1; }
fi
bash "$KSERVE_INSTALL_SCRIPT"
kubectl wait --for=condition=Available deployment/kserve-controller-manager -n kserve --timeout=300s

# Force RawDeployment as the default. KServe ships with Standard/Serverless
# as the default; KEDA + Argo Rollouts traffic splitting conflict with Knative
# revision routing (planned ADR 0003 — RawDeployment + KEDA over Serverless).
# Verified live 2026-07-13: the previous bootstrap set "Standard" — a bug
# contradicting this comment — which silently left Knative required (not
# installed) and blocked every InferenceService at reconcile time.
kubectl -n kserve get cm inferenceservice-config -o json \
  | jq '.data.deploy |= (fromjson | .defaultDeploymentMode = "RawDeployment" | tostring)' \
  | kubectl apply -f -

# Istio Gateway that KServe's `kserveIngressGateway: kserve/kserve-ingress-gateway`
# config references. Without it, RawDeployment ISVCs reconcile but their
# VirtualServices have no Gateway to attach to → 404 at the ingress proxy.
# Selector matches the istio-ingress proxy installed above.
kubectl apply -f "${SCRIPT_DIR}/manifests/kserve-ingress-gateway.yaml"

# ----------------------------------------------------------------------------
echo "==> KEDA (autoscaling; RawDeployment-only per KServe docs)"
kubectl create namespace keda --dry-run=client -o yaml | kubectl apply -f -
helm upgrade --install keda kedacore/keda -n keda --version "$KEDA_CHART_VERSION" --wait
kubectl wait --for=condition=Available deployment/keda-operator -n keda --timeout=120s

# ----------------------------------------------------------------------------
echo "==> Argo Rollouts (progressive delivery)"
kubectl create namespace argo-rollouts --dry-run=client -o yaml | kubectl apply -f -
helm upgrade --install argo-rollouts argo/argo-rollouts -n argo-rollouts \
  --version "$ARGO_ROLLOUTS_CHART_VERSION" --wait
kubectl wait --for=condition=Available deployment/argo-rollouts -n argo-rollouts --timeout=120s

# ----------------------------------------------------------------------------
echo "==> kube-prometheus-stack (Prometheus, Grafana, Alertmanager, operator)"
kubectl create namespace observability --dry-run=client -o yaml | kubectl apply -f -
helm upgrade --install kube-prometheus-stack \
  prometheus-community/kube-prometheus-stack -n observability \
  --version "$KUBE_PROMETHEUS_STACK_CHART_VERSION" \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --set grafana.service.type=NodePort \
  --set grafana.adminPassword=admin \
  --wait
# Prometheus + Alertmanager come up as StatefulSets — wait explicitly.
kubectl -n observability wait --for=condition=Ready \
  pod -l app.kubernetes.io/name=prometheus --timeout=300s || true
kubectl -n observability wait --for=condition=Ready \
  pod -l app.kubernetes.io/name=alertmanager --timeout=300s || true

# ----------------------------------------------------------------------------
echo "==> MinIO (artifact store)"
kubectl create namespace mlflow --dry-run=client -o yaml | kubectl apply -f -
# NOTE: helm parses `--set 'buckets[0].name=foo,buckets[0].policy=none'` as a
# single key/value pair (the comma inside quotes is part of the value). Split
# into separate --set flags or the bucket silently doesn't get created and
# MLflow fails on first artifact upload with "bucket does not exist".
helm upgrade --install minio minio/minio -n mlflow --version "$MINIO_CHART_VERSION" \
  --set mode=standalone \
  --set rootUser=minioadmin \
  --set rootPassword=minioadmin \
  --set persistence.size=10Gi \
  --set buckets[0].name=mlflow \
  --set buckets[0].policy=none \
  --wait

# KServe storage-initializer Secret for pulling model artifacts from
# MinIO. The ISVC `storage.key: localMinio` references the JSON blob
# inside. Creds MUST mirror the rootUser/rootPassword above.
kubectl apply -f "${SCRIPT_DIR}/manifests/kserve-storage-secret.yaml"

# ----------------------------------------------------------------------------
echo "==> MLflow tracking server"
# Manifest path is more reliable than the unmaintained community chart.
# The image's default Cmd is `python3` with no entrypoint, so command: ["mlflow"]
# is required in the manifest for `mlflow server ...` to be invoked correctly.
kubectl apply -f "${SCRIPT_DIR}/manifests/mlflow.yaml"
kubectl -n mlflow wait --for=condition=Available deployment/mlflow --timeout=180s

# ----------------------------------------------------------------------------
echo
echo "Platform stack installed."
echo "  MLflow UI:    kubectl -n mlflow port-forward svc/mlflow 5000"
echo "  Grafana:      kubectl -n observability port-forward svc/kube-prometheus-stack-grafana 3000:80"
echo "  Prometheus:   kubectl -n observability port-forward svc/kube-prometheus-stack-prometheus 9090:9090"
