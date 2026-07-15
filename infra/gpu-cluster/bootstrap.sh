#!/usr/bin/env bash
# Bootstrap the GPU cluster (k3s on the workstation, 2x RTX 2060 Super).
#
# Requires root. Re-run as:
#   sudo -E ./infra/gpu-cluster/bootstrap.sh
#
# Prerequisite: NVIDIA driver installed on the host (nvidia-smi works).
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "This script installs k3s and must run as root."
  echo "Re-run as:  sudo -E $0"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_OPERATOR_CHART_VERSION="${GPU_OPERATOR_CHART_VERSION:-v24.6.0}"

# Thin wrapper around the shared installer. GPU cluster = WITH_GPU=true.
export CLUSTER_NAME="${CLUSTER_NAME:-mlops-gpu}"
export WITH_GPU=true

# shellcheck source=../k3s-install.sh
source "$SCRIPT_DIR/../k3s-install.sh"

# ----------------------------------------------------------------------------
# NVIDIA GPU Operator — installs device plugin, DCGM exporter, node feature
# discovery. Driver and container-toolkit are managed on the host (see
# gpu-operator-values.yaml for why).
# ----------------------------------------------------------------------------
echo "==> NVIDIA GPU Operator (device plugin + DCGM exporter + NFD)"
helm repo add nvidia https://nvidia.github.io/gpu-operator
helm repo update
kubectl create namespace gpu-operator --dry-run=client -o yaml | kubectl apply -f -
helm upgrade --install gpu-operator nvidia/gpu-operator -n gpu-operator \
  --version "$GPU_OPERATOR_CHART_VERSION" \
  --values "$SCRIPT_DIR/gpu-operator-values.yaml"

kubectl wait --for=condition=Available deployment/gpu-operator -n gpu-operator --timeout=300s
kubectl rollout status daemonset/nvidia-device-plugin-daemonset -n gpu-operator --timeout=300s
kubectl rollout status daemonset/gpu-feature-discovery -n gpu-operator --timeout=300s
kubectl rollout status daemonset/nvidia-dcgm-exporter  -n gpu-operator --timeout=300s

echo "==> Verifying GPU is schedulable from a pod"
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.capacity.'nvidia\.com/gpu'
cat <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-smoke-test
spec:
  restartPolicy: Never
  containers:
    - name: cuda
      image: nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04
      command: ["nvidia-smi"]
      resources:
        limits:
          nvidia.com/gpu: 1
EOF
kubectl wait --for=condition=Ready pod/gpu-smoke-test --timeout=180s || true
kubectl logs gpu-smoke-test
kubectl delete pod gpu-smoke-test --ignore-not-found

# ----------------------------------------------------------------------------
# Shared platform stack (identical to the CPU cluster).
# ----------------------------------------------------------------------------
# shellcheck source=../install-platform-stack.sh
source "$SCRIPT_DIR/../install-platform-stack.sh"

echo
echo "GPU cluster bootstrap complete."
echo "  Switch context:  export KUBECONFIG=${KUBECONFIG_PATH:-$HOME/.kube/config}"
echo "  GPU metrics:     kubectl -n observability port-forward svc/kube-prometheus-stack-prometheus 9090:9090"
echo "  Query Prometheus: DCGM_FI_DEV_GPU_UTIL"
