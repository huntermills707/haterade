#!/usr/bin/env bash
# Make k3s immune to host VPN DNS takeovers (NordVPN, Tailscale MagicDNS, etc).
#
# Problem:
#   - Kubelet detects that /etc/resolv.conf points at systemd-resolved stub
#     (127.0.0.53) and falls back to /run/systemd/resolve/resolv.conf for
#     upstream resolvers (typically your ISP's DNS).
#   - When NordVPN connects, it overwrites that file with Nord's DNS servers
#     (103.86.96.96 / 103.86.99.99 under Threat Protection Lite).
#   - Those servers are only reachable through the NordLynx tunnel.
#   - NordLynx uses fwmark-based policy routing (fwmark 0xe1f1) that only
#     catches host-originated traffic. Pod-originated traffic (source
#     10.42.0.0/16) isn't steered through the tunnel.
#   - CoreDNS (running in a pod) reads resolv.conf at startup and snapshots
#     it. Its upstream queries to Nord's DNS time out, breaking cluster-wide
#     DNS even after the VPN disconnects — until the CoreDNS pod restarts.
#   - Separately (verified 2026-07-13): NordVPN's firewall drops
#     bridge-forwarded pod<->pod traffic on cni0 (iptables FORWARD via
#     br_netfilter), killing pod->CoreDNS entirely — even while
#     "Disconnected", because Meshnet keeps the ruleset loaded. The subnet
#     allowlist does NOT help here; it only covers host INPUT/OUTPUT.
#     Symptom: pod->pod times out while pod->host and pod->internet work.
#
# Fix:
#   Give k3s its own static resolv.conf that the VPN can't touch. Stable public
#   resolvers (1.1.1.1, 9.9.9.9) reachable without the tunnel. Plus disable
#   NordVPN's firewall (not the kill switch — that's a separate setting).
#
# This is idempotent. Run once per k3s host. The bootstrap prints a reminder
# if it detects nordlynx at startup.

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  exec sudo -E "$0" "$@"
fi

RESOLV_CONF="/etc/rancher/k3s/resolv.conf"
K3S_CONFIG="/etc/rancher/k3s/config.yaml"

echo "==> Writing stable upstream resolvers to $RESOLV_CONF"
mkdir -p "$(dirname "$RESOLV_CONF")"
cat >"$RESOLV_CONF" <<'EOF'
# Managed by basic_mlops_pipeline/infra/scripts/harden-against-vpn-dns.sh
# Stable public resolvers so VPN DNS takeover can't break k3s upstream lookup.
nameserver 1.1.1.1
nameserver 9.9.9.9
EOF

echo "==> Pointing kubelet at it via $K3S_CONFIG"
if [ -f "$K3S_CONFIG" ]; then
  if grep -q "resolv-conf=$RESOLV_CONF" "$K3S_CONFIG"; then
    echo "    already configured"
  else
    echo "    $K3S_CONFIG already exists. Append this manually, then re-run:"
    echo "      kubelet-arg:"
    echo '        - "resolv-conf='"$RESOLV_CONF"'"'
    exit 1
  fi
else
  cat >"$K3S_CONFIG" <<EOF
# Managed by basic_mlops_pipeline/infra/scripts/harden-against-vpn-dns.sh
kubelet-arg:
  - "resolv-conf=$RESOLV_CONF"
EOF
fi

echo "==> Restarting k3s"
systemctl restart k3s
# Wait for API server to come back
for i in $(seq 1 30); do
  if kubectl get nodes >/dev/null 2>&1; then break; fi
  sleep 2
done

echo "==> Restarting CoreDNS (it snapshots resolv.conf at pod start)"
if command -v kubectl >/dev/null 2>&1; then
  kubectl -n kube-system rollout restart deployment coredns
  kubectl -n kube-system rollout status deployment coredns --timeout=60s
else
  echo "    kubectl not found; restart coredns manually:"
  echo "      kubectl -n kube-system rollout restart deployment coredns"
fi

echo ""
echo "==> Optional: NordVPN allowlist for cluster CIDRs (defends against fwmark routing edges)"
if command -v nordvpn >/dev/null 2>&1; then
  nordvpn allowlist add subnet 10.42.0.0/16 || true   # k3s pod CIDR
  nordvpn allowlist add subnet 10.43.0.0/16 || true   # k3s service CIDR
  echo "    allowlisted"
else
  echo "    nordvpn not installed; skipping"
fi

echo ""
echo "==> NordVPN firewall (drops pod<->pod FORWARD traffic on cni0)"
if command -v nordvpn >/dev/null 2>&1; then
  if nordvpn settings | grep -q "Firewall: enabled"; then
    echo "    disabling. Its nftables rules drop bridge-forwarded pod->pod"
    echo "    traffic even while disconnected (Meshnet keeps them loaded),"
    echo "    which breaks pod->CoreDNS. Allowlist subnets do not cover FORWARD."
    echo "    Kill Switch is a separate setting and is not touched."
    nordvpn set firewall off
  else
    echo "    already disabled"
  fi
else
  echo "    nordvpn not installed; skipping"
fi

echo ""
echo "==> Done. Verify (with VPN connected):"
echo "    kubectl run -it --rm dnstest --image=busybox:1.36 --restart=Never -- \\"
echo "      nslookup kubernetes.default"
echo "    kubectl run -it --rm dnstest2 --image=busybox:1.36 --restart=Never -- \\"
echo "      nslookup google.com"
