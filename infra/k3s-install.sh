#!/usr/bin/env bash
# Shared k3s installer. Sourced from cpu-cluster/bootstrap.sh and
# gpu-cluster/bootstrap.sh. Sourceable so the calling bootstrap keeps the
# KUBECONFIG env var for the subsequent install-platform-stack.sh call.
#
# Required: run as root (the calling bootstrap checks this). We rely on
# `sudo -E ./bootstrap.sh` so $HOME is preserved and kubeconfig lands in the
# invoking user's ~/.kube/config instead of /root/.kube/config.
#
# Reads env vars (set BEFORE sourcing):
#   CLUSTER_NAME     (default: mlops)
#   WITH_GPU         (default: false; set to "true" for GPU clusters)
#   KUBECONFIG_PATH  (default: $HOME/.kube/config)
#   K3S_VERSION      (default: v1.34.9+k3s1, matches k8s 1.34 line)
set -euo pipefail

CLUSTER_NAME="${CLUSTER_NAME:-mlops}"
WITH_GPU="${WITH_GPU:-false}"
KUBECONFIG_PATH="${KUBECONFIG_PATH:-$HOME/.kube/config}"
K3S_VERSION="${K3S_VERSION:-v1.34.9+k3s1}"

echo "==> Cluster: $CLUSTER_NAME   GPU: $WITH_GPU   k3s: $K3S_VERSION"

# Ensure user-installed helm/kubectl/jq (typically in ~/.local/bin) are
# reachable even when this script runs under `sudo -E` whose secure_path
# overrides PATH. $HOME is preserved by -E, so $HOME/.local/bin stays valid.
export PATH="${HOME}/.local/bin:${PATH}"

for tool in curl kubectl; do
  command -v "$tool" >/dev/null 2>&1 || { echo "missing required tool: $tool"; exit 1; }
done

# ----------------------------------------------------------------------------
# VPN DNS-takeover detection.
#
# NordVPN's client (and similar) overwrites /run/systemd/resolve/resolv.conf
# when connected. k3s's kubelet reads that file for upstream resolvers.
# Nord's DNS servers are only reachable via the tunnel, but NordLynx's
# fwmark-based policy routing (fwmark 0xe1f1) only catches host-originated
# traffic — pod-originated traffic (10.42.0.0/16) doesn't get steered through
# the tunnel, so CoreDNS upstream queries time out, breaking cluster-wide DNS.
#
# Disconnecting NordVPN doesn't fix it because CoreDNS snapshots resolv.conf
# at pod start. The real fix is to give k3s its own static resolv.conf:
#   sudo ./infra/scripts/harden-against-vpn-dns.sh
if ip link show nordlynx >/dev/null 2>&1; then
  echo "WARNING: nordlynx interface is active."
  echo "         NordVPN will overwrite k3s's upstream DNS and break cluster DNS."
  echo "         Its firewall also drops pod<->pod traffic on cni0 (even when"
  echo "         disconnected, if Meshnet is on)."
  echo "         Apply the documented fix BEFORE bootstrapping:"
  echo "           sudo ./infra/scripts/harden-against-vpn-dns.sh"
  echo "         Or stop NordVPN entirely:"
  echo "           nordvpn disconnect && sudo ip link delete nordlynx"
  echo ""
  echo "Continuing in 10s anyway (Ctrl-C to abort)..."
  sleep 10
fi

# ----------------------------------------------------------------------------
# GPU host prep — nvidia-container-toolkit + driver check.
# Runs only when WITH_GPU=true. CPU clusters skip this entirely.
# ----------------------------------------------------------------------------
if [ "$WITH_GPU" = "true" ]; then
  echo "==> GPU preflight"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found. Install the NVIDIA driver on the host first."
    exit 1
  fi
  nvidia-smi -L

  if ! command -v nvidia-ctk >/dev/null 2>&1; then
    echo "==> Installing nvidia-container-toolkit (Debian/Ubuntu packages)"
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update
    apt-get install -y nvidia-container-toolkit
  fi
fi

# ----------------------------------------------------------------------------
# k3s install (idempotent). ServiceLB is kept (k3s-native LoadBalancer — the
# Istio ingress gateway needs an LB impl). Traefik is disabled (Istio owns
# ingress). This is the same on both clusters; only the GPU path diverges.
# ----------------------------------------------------------------------------
if ! command -v k3s >/dev/null 2>&1; then
  echo "==> Installing k3s"
  echo "    --disable traefik   (Istio owns ingress)"
  echo "    keep ServiceLB      (k3s-native LB; needed by Istio gateway)"
  curl -sfL https://get.k3s.io \
    | INSTALL_K3S_VERSION="$K3S_VERSION" sh -s - server \
      --disable traefik \
      --node-name "${CLUSTER_NAME}-node" \
      --write-kubeconfig-mode 644 \
      --write-kubeconfig "$KUBECONFIG_PATH"
else
  echo "==> k3s already installed, reusing"
fi

export KUBECONFIG="$KUBECONFIG_PATH"

# ----------------------------------------------------------------------------
# Configure k3s containerd for the NVIDIA runtime. Must run AFTER k3s install
# because containerd's config.toml is generated on first k3s start.
#
# k3s v1.34 generates a version=3 config.toml that imports drop-ins from
# config-v3.toml.d/*.toml. We register the nvidia runtime AND set it as the
# default so GPU pods (and the device plugin's allocations) use
# nvidia-container-runtime instead of runc. nvidia-container-runtime safely
# falls back to runc for containers that request no GPU.
# ----------------------------------------------------------------------------
if [ "$WITH_GPU" = "true" ]; then
  echo "==> Configuring k3s containerd for NVIDIA runtime"
  nvidia-ctk runtime configure \
    --runtime=containerd \
    --config=/var/lib/rancher/k3s/agent/etc/containerd/config.toml

  K3S_CONTAINERD_DROPIN_DIR="/var/lib/rancher/k3s/agent/etc/containerd/config-v3.toml.d"
  mkdir -p "$K3S_CONTAINERD_DROPIN_DIR"
  cat > "$K3S_CONTAINERD_DROPIN_DIR/99-nvidia-default.toml" <<'EOF'
[plugins."io.containerd.cri.v1.runtime".containerd]
  default_runtime_name = "nvidia"
EOF
  chmod 644 "$K3S_CONTAINERD_DROPIN_DIR/99-nvidia-default.toml"

  systemctl restart k3s
fi

# ----------------------------------------------------------------------------
# Wait for the node to be Ready. k3s takes a few seconds after install to
# expose the API server, so retry until kubectl can talk to it.
# ----------------------------------------------------------------------------
echo "==> Waiting for k3s node to be Ready"
for i in $(seq 1 30); do
  if kubectl get nodes >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
kubectl wait --for=condition=Ready nodes --all --timeout=300s
kubectl get nodes
