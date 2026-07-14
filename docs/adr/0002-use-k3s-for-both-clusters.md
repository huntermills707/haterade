# 2. Use k3s for both CPU and GPU clusters

Date: 2026-07-13
Status: Accepted
Supersedes: [0001 — Use kind for the CPU cluster and k3s for the GPU cluster](0001-use-kind-for-cpu-and-k3s-for-gpu.md)

## Context

ADR 0001 picked **kind** for the CPU/local-dev cluster and **k3s** for the GPU
cluster, on the theory that kind's Docker-based topology is faster to recreate
on a laptop and that the asymmetry was acceptable. During M0 verification
(actually running `bootstrap.sh` end-to-end), the asymmetry cost more than
predicted and a class of kind-specific issues emerged that have no k3s
equivalent.

### What M0 verification surfaced

1. **MetalLB install + IPv4 detection became a kind-only burden.** Modern
   Docker ships the `kind` network dual-stacked (IPv6 first). The bootstrap's
   `docker network inspect ... .IPAM.Config[0].Subnet` returned an IPv6 subnet
   and the MetalLB pool ended up outside the network. Required a jq-based IPv4
   filter. k3s ships ServiceLB and needs no equivalent step.
2. **Host `ufw` filtering the kind docker bridge blocked cross-node pod
   traffic.** The host runs `ufw`, whose default FORWARD DROP policy filters
   traffic crossing the docker bridge. ICMP passed, but UDP DNS to `kube-dns`
   and TCP data exchange timed out, breaking every component that resolves
   service names. Forced a reversion from the specced multi-node (control-plane
   + 2 workers) to single-node, sacrificing realistic scheduling to dodge the
   host firewall. k3s's flannel doesn't traverse the host netns the same way.
3. **Two bootstrap paths to maintain.** `kind-config.yaml` + bootstrap.sh on
   one side, `k3s-install.sh` + bootstrap.sh on the other. Already drifting.
4. **The "shared platform stack" claim was half-aspirational.**
   `install-platform-stack.sh` is shared, but everything before it (cluster
   creation, LB setup) was forked.

The trigger for revisiting was concrete: multi-node capability was sacrificed
to a kind-specific host-firewall interaction. That is a real platform
capability lost to an aesthetic preference.

## Decision

Use **k3s** for both clusters. The CPU cluster runs k3s on the laptop, the GPU
cluster runs k3s on the workstation. Both use the same shared installer
(`infra/k3s-install.sh`) with a single flag (`WITH_GPU=true`) that toggles the
GPU host-prep steps (nvidia-container-toolkit, containerd config).

### What the shared installer owns

- k3s install (idempotent; safe to re-run)
- `--disable traefik` (Istio owns ingress)
- **Keep** ServiceLB (k3s-native LoadBalancer; the Istio gateway needs an LB
  impl)
- Conditional: NVIDIA driver check + nvidia-container-toolkit + containerd
  config when `WITH_GPU=true`

### What still differs per cluster

- GPU Operator install (`gpu-cluster/bootstrap.sh` only)
- `nodeSelector: nvidia.com/gpu.present=true` on GPU manifests
- Runtime image (MLServer vs Triton + TensorRT)
- StorageClass defaults (`local-path` on both, sufficient)

Everything else — Istio, KServe, KEDA, Argo Rollouts, Prometheus stack,
MinIO, MLflow — is identical and sourced from the same
`install-platform-stack.sh`.

## Consequences

### Positive

- **One runtime to learn, one set of failure modes.** The mental model is
  half the size it was. Reading logs, debugging scheduling, understanding CNI
  behavior — all transferable between clusters.
- **MetalLB disappears entirely.** k3s ServiceLB handles LoadBalancer Services
  natively. No more IP-pool configuration, no more dual-stack subnet detection.
- **Multi-node becomes viable again on the workstation.** Not used day one, but
  the door is open for a multi-node GPU cluster demo without re-engineering.
- **Better interview story.** "Same runtime in dev and prod-like
  environments" reads better than "kind on laptop, k3s on workstation because
  of historical reasons."
- **The shared-platform-stack claim becomes literally true.** Both clusters
  run `install-platform-stack.sh` against a k3s API server. No runtime forks.

### Negative

- **Lost host isolation.** k3s installs systemd units, adds a `cni0` interface
  to the host, runs kubelet/flannel directly. kind's containerization is gone.
  Acceptable for a dev laptop dedicated to this work; would matter for shared
  machines.
- **k3s is heavier on the host than kind-in-Docker.** Slightly more invasive
  to tear down (`k3s-uninstall.sh` plus manual cleanup of `/var/lib/rancher/k3s`
  if it gets wedged).
- **Bootstrap scripts must run as root.** `sudo -E ./bootstrap.sh`. Inline
  sudo doesn't compose cleanly with sourcing patterns, so the root check moved
  to the top of every bootstrap.
- **Host DNS-takeover software breaks cluster DNS.** When NordVPN (or similar
  — Tailscale MagicDNS, some Pi-hole configs) connects, it overwrites
  `/run/systemd/resolve/resolv.conf` to push its own DNS servers. k3s's
  kubelet reads that file for upstream resolvers, and CoreDNS snapshots it at
  pod start. Nord's DNS servers are reachable only through the NordLynx
  tunnel, but NordLynx's fwmark-based policy routing (fwmark 0xe1f1) only
  catches host-originated traffic — pod-originated traffic
  (source `10.42.0.0/16`) isn't steered through the tunnel, so CoreDNS's
  upstream queries time out and cluster-wide DNS fails. Disconnecting the VPN
  doesn't fix it until CoreDNS is restarted (stale snapshot). Mitigation:
  `infra/scripts/harden-against-vpn-dns.sh` gives k3s its own static
  resolv.conf (`/etc/rancher/k3s/resolv.conf` with 1.1.1.1, 9.9.9.9) that
  the VPN can't touch, plus kubelet-arg config to point at it. Run once per
  host. Documented as a host prerequisite. kind's container isolation hid
  this by accident (kind's CoreDNS used the container's resolv.conf, not the
  host's); k3s exposes it.

## Alternatives considered

### Stay with kind (the original ADR 0001 decision)

Defensible in principle — kind's isolation is real, and on a clean host without
ufw issues the multi-node story works. Rejected because:
- The host in question does have ufw issues, and we can't assume a reviewer's
  host won't.
- The asymmetry cost (two bootstraps, two debugging mental models) outweighs
  the isolation benefit for a portfolio that explicitly runs the same
  manifests on both clusters.

### Run both clusters on one host (e.g., laptop)

k3s supports multi-cluster on a single host via separate data dirs and ports,
but the topology doesn't match the project's framing (CPU on laptop, GPU on
workstation with real hardware). Rejected; the value is in running on real
GPU hardware, not in compacting onto one machine.

### Managed cloud (EKS/GKE) for both

Deferred. Will be addressed in a future ADR as a stretch goal (Terraform
overlay) once the local implementation is stable.

## References

- k3s: https://docs.k3s.io/
- k3s ServiceLB: https://docs.k3s.io/networking/networking-services#embedded-service-load-balancer
- M0 verification notes (this repo):
  - `infra/install-platform-stack.sh` (verified versions)
  - `infra/k3s-install.sh` (shared installer)
