# 8. Argo Rollouts owns the CPU predictor for canary delivery

Date: 2026-07-14
Status: Accepted

## Context

Milestone M4 requires canary delivery with Prometheus analysis gates:
progressively shift traffic from a stable model version to a new version,
automatically rolling back if error rate or latency degrades. The project
chose Argo Rollouts with Istio traffic splitting (ADR 0003).

Two implementation options were considered:

1. **KServe-managed stable predictor + separate Argo Rollout canary**
   (Option A). Keep the existing KServe InferenceService as the stable
   version and add a separate Rollout for the canary version. A shared
   Istio VirtualService would split traffic between the two.

2. **Argo Rollouts owns the predictor** (Option B). Replace the KServe
   InferenceService predictor with an Argo Rollout that manages both
   stable and canary ReplicaSets directly.

## Decision

Use **Option 3: Argo Rollouts owns the CPU predictor**.

The `toxicity-cpu` InferenceService is replaced by:

- An Argo `Rollout` named `toxicity-cpu`.
- Two Services, `toxicity-cpu-stable` and `toxicity-cpu-canary`, whose
  selectors Argo Rollouts mutates to point at the active ReplicaSets.
- An Istio `VirtualService` named `toxicity-cpu` that Argo Rollouts owns
  and mutates during canary progressions.
- A KEDA `ScaledObject` targeting the Rollout directly for 1 → N → 1
  autoscaling on Triton queue duration.
- A `ServiceMonitor` selecting the Rollout's pod labels.

The KServe ISVC predictor is removed from the CPU path. KServe remains
installed on the cluster and continues to manage the GPU predictor; only
the CPU predictor's lifecycle moves to Argo Rollouts.

## Consequences

### Positive

- **Argo Rollouts has full control of the canary lifecycle.** No
  second controller (KServe) touches the same Deployment or Service
  selectors. The traffic-split mechanics are the ones Argo Rollouts is
  designed for.
- **No selector-mutation conflict.** Argo Rollouts' Istio integration
  requires it to own both `stableService` and `canaryService`. Option A
  would have meant Argo Rollouts mutating a KServe-managed Service,
  which is fragile and operationally confusing.
- **Simpler mental model.** One Rollout, one VirtualService, one
  ScaledObject. The stable and canary versions are just different
  ReplicaSets of the same Rollout.
- **KServe RawDeployment rationale from ADR 0003 remains valid.** The
  reason for choosing RawDeployment (avoid Knative/Argo Rollouts
  VirtualService conflict) still holds; we are simply extending the
  "plain Kubernetes resources" choice to let Argo Rollouts own the
  Deployment shape.

### Negative

- **KServe abstraction is lost for the CPU predictor.** We no longer get
  KServe's `InferenceService` status, storage initializer, or predictor
  lifecycle management for CPU. The model still moves via PVC built by
  `build-model-repo.sh`.
- **Two serving controllers in the repo.** CPU uses Argo Rollouts; GPU
  still uses KServe. This divergence is documented and bounded to M4/M5;
  if the GPU path later needs canary, it would follow the same Rollout
  pattern.
- **KEDA ScaledObject must be managed manually.** KServe previously
  created the ScaledObject from ISVC annotations; now it lives in
  `serving/cpu/scaledobject.yaml`.
- **Reverting to KServe later requires deleting the Rollout and
  re-creating the ISVC.** This is a one-way conversion for a given
  environment, though the manifest files make it reproducible.

## Alternatives considered

### Option A: KServe stable + separate Rollout canary

Keep the KServe ISVC as the stable predictor and run a separate Rollout
for canary. A shared VirtualService would split traffic between the
KServe Service and the canary Service.

Rejected because:
- Argo Rollouts' Istio integration requires `stableService` and
  `canaryService` references. It mutates the selectors of both Services
  to point at its own stable and canary ReplicaSets. Using the KServe
  Service as `stableService` would cause Argo Rollouts to detach it from
  the KServe Deployment, leaving two stable predictors (one idle) and a
  fragile cross-controller relationship.
- There is no supported way to tell Argo Rollouts "the stable Service is
  external and managed by KServe."

### Keep KServe and use Knative revisions for canary

Use KServe Serverless mode and Knative's built-in traffic splitting
between revisions.

Rejected because:
- ADR 0003 already rejected Serverless mode to avoid the Knative Route /
  VirtualService conflict with Argo Rollouts. Revisiting this would undo
  that decision and reintroduce the conflict.

## References

- `serving/cpu/rollout.yaml` — stable Rollout, Services, VirtualService.
- `serving/cpu/scaledobject.yaml` — KEDA autoscaling for the Rollout.
- `serving/cpu/triton-servicemonitor.yaml` — Prometheus scraping.
- `serving/cpu/canary/rollout-v2.yaml` — canary update (placeholder v2).
- `serving/cpu/analysis-template.yaml` — Prometheus analysis gates.
- `docs/adr/0003-rawdeployment-and-keda-over-serverless.md` — original
  platform choice that enabled this move.
- `docs/adr/0009-placeholder-v2-artifact-for-m4.md` — placeholder model
  used to exercise the canary.
- Argo Rollouts Istio traffic management:
  https://argo-rollouts.readthedocs.io/en/stable/features/traffic-management/istio/
- KEDA scaling Rollouts:
  https://keda.sh/docs/latest/concepts/scaling-deployments/
