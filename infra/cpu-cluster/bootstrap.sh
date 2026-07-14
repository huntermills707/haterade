#!/usr/bin/env bash
# Bootstrap the CPU / local-dev cluster (k3s on the laptop).
#
# Requires root because k3s installs a systemd service. Re-run as:
#   sudo -E ./infra/cpu-cluster/bootstrap.sh
# The -E preserves $HOME so kubeconfig lands in the invoking user's ~/.kube/.
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "This script installs k3s and must run as root."
  echo "Re-run as:  sudo -E $0"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Thin wrapper around the shared installer. CPU cluster = no GPU.
export CLUSTER_NAME="${CLUSTER_NAME:-mlops-cpu}"
export WITH_GPU=false

# shellcheck source=../k3s-install.sh
source "$SCRIPT_DIR/../k3s-install.sh"

# shellcheck source=../install-platform-stack.sh
source "$SCRIPT_DIR/../install-platform-stack.sh"

echo
echo "CPU cluster bootstrap complete."
echo "  Switch context:  export KUBECONFIG=${KUBECONFIG_PATH:-$HOME/.kube/config}"
