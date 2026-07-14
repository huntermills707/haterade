# 1. Use kind for the CPU cluster and k3s for the GPU cluster

Date: 2026-07-12
Status: **Superseded** by [0002 — Use k3s for both clusters](0002-use-k3s-for-both-clusters.md)

This ADR is retained as a historical record. The decision was reversed after
M0 verification surfaced host-firewall issues specific to kind's Docker-in-Docker
topology. Read 0002 for the current architecture.

## Context

This portfolio requires two Kubernetes clusters with materially different
constraints:

- **CPU / local-dev cluster** runs on the laptop and hosts the MLOps-handoff
  story (MLflow → MLServer → KServe, Argo Rollouts canary, KEDA scale-to-zero).
  No GPUs. Fast iteration matters more than realism.
- **GPU cluster** runs on a dedicated workstation with 2× NVIDIA RTX 2060 Super
  (Turing, sm_75). It hosts the Triton + TensorRT performance and DCGM-driven
  autoscaling story. Requires direct GPU access from pods.

Both clusters must support the same platform stack so the application manifests
(InferenceService, Rollout, AnalysisTemplate, ScaledObject) are portable:

- KServe in RawDeployment mode (not Serverless)
- KEDA autoscaling (RawDeployment-only — see KServe docs)
- Argo Rollouts traffic splitting (via Istio)
- kube-prometheus-stack for metrics and Rollout analysis

## Decision

Use **kind** for the CPU cluster and **k3s** for the GPU cluster.

### CPU cluster — kind on Docker

- Tear down with `kind delete cluster`. Recreate in ~2 minutes.
- **Single-node** (control-plane only). Originally specced as control-plane +
  2 workers for realistic scheduling, but reverted to single-node because of
  a host firewall issue (see "Host firewall caveat" below). Multi-node is
  restorable by fixing the host ufw rules.
- MetalLB (installed by `bootstrap.sh`) hands out LoadBalancer IPs inside the
  kind docker network range. The Istio ingress gateway is reachable from the
  host at its MetalLB-allocated IP (e.g. `172.19.255.200`), not via localhost.

### GPU cluster — k3s on bare metal

- Single-binary install, no nested virtualization. Direct GPU access via the
  NVIDIA container toolkit configured against k3s's containerd socket.
- Disable k3s's bundled Traefik (Istio owns ingress). **Keep** k3s ServiceLB
  enabled — it is k3s's native LoadBalancer implementation, and the Istio
  ingress gateway needs *some* LB controller to assign it an external IP. The
  CPU cluster has no bundled LB controller, so it gets MetalLB instead. The
  application manifests remain portable: LB implementation is cluster
  infrastructure, not an application concern, and the shared
  `install-platform-stack.sh` is agnostic to it.
- NVIDIA GPU Operator runs in-cluster to manage the device plugin, DCGM
  exporter, and node feature discovery. Host already has the driver and
  container-toolkit packages installed.

### Node taint policy

GPU nodes are **not tainted**. Workload placement is controlled with
`nodeSelector: nvidia.com/gpu.present: "true"` on GPU manifests only.

This is deliberate, not an oversight. The GPU cluster is single-node: every
system component (Prometheus, MLflow, Istio, KEDA, Argo, GPU Operator) must
co-schedule on the same node as the GPU workload pods. A `NoSchedule` taint
would prevent system pods from scheduling and break the cluster. Even a
`PreferNoSchedule` soft taint is pointless here because there is no other node
to schedule onto.

When the GPU cluster expands to multiple nodes with a dedicated GPU node pool
separated from CPU-only nodes, revisit this: taint GPU nodes with
`nvidia.com/gpu.present=true:NoSchedule` and add tolerations to every GPU
workload manifest. Until then, nodeSelector-only is correct.

## Consequences

### Positive

- The same `infra/install-platform-stack.sh` installs KServe/KEDA/Argo/Istio/
  Prometheus/MLflow on both clusters. Application YAML is portable; only the
  runtime image (MLServer vs Triton) and node selector differ.
- The laptop dev loop stays fast (kind recreate beats VM provisioning).
- The workstation gets uncomplicated GPU access — no fighting docker-in-docker
  GPU passthrough.

### Negative

- Two cluster runtimes to learn. Mitigated by keeping the platform stack shared
  and isolating GPU-specific concerns in `infra/gpu-cluster/`.
- The GPU workstation must be powered on and networked for Act 2 work.
- k3s uses containerd (not dockerd) and ships a non-default config path at
  `/var/lib/rancher/k3s/agent/etc/containerd/config.toml`. The
  nvidia-container-toolkit config step must target that path or GPU pods fail
  silently. `infra/gpu-cluster/k3s-install.sh` handles this; documented here so
  it is not forgotten.

### Risks

- **TensorRT engine plans are GPU-architecture-bound.** A plan baked on sm_75
  (Turing / 2060 Super) will not run on the kind cluster (no GPU) or on any
  other arch. Engine compilation must happen per target architecture, ideally in
  CI. This will be covered by a future ADR on the release pipeline.
- **Cold-start cost on GPU pods.** Loading a TensorRT engine on a cold GPU adds
  seconds to the first request after scale-from-zero. Mitigation: keep the model
  small (DistilBERT fp16 plan ≈ 250 MB), accept ~5–10 s cold start, document it.

### Host firewall caveat (discovered during M0 verification)

The CPU cluster runs **single-node**, not the originally-specced
control-plane + 2 workers. Reason: the host runs `ufw`, whose default FORWARD
DROP policy filters inter-node pod traffic on the kind docker bridge
(`br-<hash>` at `172.19.0.1`). ICMP passes, but UDP DNS to `kube-dns` and TCP
data transfer time out, breaking every component that resolves service names.

Single-node sidesteps this entirely: all pod traffic stays inside the kind
container's network namespace and never traverses the host's FORWARD chain.
kube-proxy runs in its default iptables mode.

To restore multi-node, fix ufw on the host first:

```
sudo ufw route allow in on br-<kind-bridge> out on br-<kind-bridge>
```

Then revert `infra/cpu-cluster/kind-config.yaml` to add the two worker nodes.
The single-node tradeoff is acceptable for this portfolio: scheduling
decisions are less interesting but every platform component (KServe, KEDA,
Argo Rollouts, Istio, MetalLB, Prometheus, MLflow) is exercised identically.

## Alternatives considered

### minikube (both clusters)

Slower startup than kind on the laptop. Adds a driver/virtualization layer
between the host and the GPUs on the workstation. Rejected for both.

### k3s for both clusters

Workable for the CPU cluster too, but kind's tear-down-and-recreate loop is
faster on a laptop and its Docker-based topology is easier to reason about
during iterative platform development. The asymmetry is intentional: each tool
matches its environment.

### kind for both clusters

GPU passthrough into kind (Docker-in-Docker) is possible but fragile: requires
`--gpus all` on the kind node container, NVIDIA runtime configuration inside
the container, and careful device-plugin scheduling. Not worth the friction on a
single-host dev loop. Rejected for the GPU cluster.

### Managed cloud (EKS / GKE / AKS)

Deferred. Will be addressed in a future ADR as a stretch goal (Terraform
overlay) once the local implementation is stable. Out of scope for the first
pass.

## References

- KServe KEDA autoscaler (RawDeployment-only):
  https://kserve.github.io/website/docs/model-serving/predictive-inference/autoscaling/keda-autoscaler/
- KServe supported runtimes (MLServer, Triton, etc.):
  https://kserve.github.io/website/docs/model-serving/predictive-inference/frameworks/overview/
- kind: https://kind.sigs.k8s.io/
- k3s: https://docs.k3s.io/
- NVIDIA GPU Operator: https://github.com/NVIDIA/gpu-operator
- NVIDIA container toolkit on k3s:
  https://docs.k3s.io/advanced#using-an-custom-containerd-configuration
