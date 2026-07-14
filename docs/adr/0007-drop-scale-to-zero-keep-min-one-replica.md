# 7. Drop true scale-to-zero from M3; keep minimum one replica in RawDeployment

Date: 2026-07-14
Status: Accepted

## Context

ADR 0003 chose KServe **RawDeployment + KEDA** for inference services so that
M3 could demonstrate scale-to-zero and M4 could use Argo Rollouts + Istio
VirtualService for canary traffic splitting without conflicting with Knative's
own traffic controller.

During live verification of M3 on the CPU cluster, two practical problems
surfaced:

1. **Pod-level metrics cannot scale from zero.** The KEDA trigger uses
   Triton's `nv_inference_queue_duration_us`, which only exists while Triton
   pods are running. When the deployment scales to 0, the metric disappears,
   so KEDA never sees load and cannot trigger the 0 → 1 transition.

2. **Cold-start latency and complexity.** Even with an ingress-level metric
   (e.g. Istio `istio_requests_total`), scaling from 0 in RawDeployment means
   scheduling a pod, pulling the image, loading the model, and warming the
   runtime before the first request can succeed. That is precisely the job
   Knative Serving was built for.

The original M3 headline was "Traffic sim + observe scale-to-zero." The
desired observation is still valuable: show the predictor autoscaling under
synthetic load. Requiring at least one replica is a reasonable trade-off for
a RawDeployment platform and keeps M3 focused on KEDA autoscaling rather than
re-implementing serverless semantics.

## Decision

Drop **true scale-to-zero** from the M3 scope.

- M3 is redefined as **"Traffic sim + observe autoscaling"**: drive load with
  Locust, watch KEDA scale the predictor up, then watch it scale back down to
  the minimum.
- Set `minReplicas: 1` on the CPU predictor (`serving/cpu/inferenceservice-triton.yaml`).
- Keep `maxReplicas: 3` and the Triton queue-duration KEDA trigger.
- Keep the 60 s scale-down stabilization window for responsive demos.
- Leave the GPU predictor at `minReplicas: 1` as well (pending hardware).
- Do **not** add an ingress-request-rate KEDA trigger solely to enable
  scale-from-zero. If scale-from-zero becomes a hard requirement later, the
  correct path is to re-evaluate Serverless mode (Knative), not to bolt a
  second metric onto RawDeployment.

## Consequences

### Positive

- **M3 is tractable and verifiable.** The demo now shows 1 → N → 1 scaling
  without needing a separate scale-from-zero mechanism.
- **No metric gymnastics.** We avoid adding Istio request-rate triggers,
  fallback scalers, or Cron-based wake-ups to compensate for pod-level
  metrics disappearing at zero replicas.
- **Latency under load is the focus.** The demo narrative shifts from
  "resource savings at idle" (serverless's core value prop) to "KEDA scales
  on the right inference signal" (queue duration), which is the genuine
  RawDeployment story.
- **Keeps ADR 0003 intact.** RawDeployment + KEDA remains the platform
  choice; only the floor replica count changes.

### Negative

- **No resource-free idle state.** The CPU predictor always keeps one pod
  running, consuming ~2 GiB RAM and one CPU request even when no traffic
  flows. For a laptop demo this is acceptable; for a cost-optimized
  production service it would not be.
- **Concedes scale-to-zero to Knative.** If a future requirement demands
  true scale-to-zero, the project should revisit Serverless mode and accept
  the traffic-routing trade-offs ADR 0003 already documented.
- **Grafana dashboard name is slightly misleading.** The existing
  `m3-cpu-scale-to-zero` dashboard title reflects the old milestone name;
  the panels themselves (request rate, replicas, queue duration, latency,
  CPU) are still correct for autoscaling observation.

## Alternatives considered

### Keep scale-to-zero and add an ingress request-rate trigger

Add a second KEDA trigger on Istio `istio_requests_total` (or a similar
Envoy metric) so KEDA can detect traffic before any backend pods exist.

Rejected because:
- It requires collecting and trusting Istio ingress metrics in Prometheus,
  which the project had not previously configured.
- It is a workaround for a problem that Knative already solved: real
  scale-to-zero belongs in serverless mode.
- It introduces another moving part (ingress metric labels, gateway
  scrape config) for a demo that no longer claims to prove scale-to-zero.

### Keep scale-to-zero with a Cron or fallback scaler

Use KEDA fallback or a Cron scaler to ensure at least one pod wakes up
periodically.

Rejected because:
- Fallback would cause oscillation: metric absent → 1 pod → metric 0 → 0
  pods → metric absent → 1 pod.
- Cron scaler makes the scaling schedule-driven, not traffic-driven, which
  contradicts the goal of demonstrating reactive autoscaling.

### Switch M3 to Serverless mode

Use KServe Serverless (Knative) for M3 to get native scale-to-zero, then
switch back to RawDeployment for M4.

Rejected because:
- It partially undoes ADR 0003 and reintroduces the Knative/Argo Rollouts
  VirtualSource conflict.
- The migration cost (delete and recreate ISVCs, retest routing) outweighs
  the demo value.
- The user explicitly accepts that serverless exists for scale-to-zero and
  is fine leaving that capability out of RawDeployment M3.

## References

- ADR 0003 — RawDeployment + KEDA over Serverless (the platform choice that
  this ADR narrows, not reverses).
- `serving/cpu/inferenceservice-triton.yaml` — CPU predictor manifest with
  `minReplicas: 1` and KEDA queue-duration trigger.
- `traffic/locustfile.py` — Locust load shape used for M3 verification.
- `monitoring/dashboards/m3-cpu-autoscaling.json` — Grafana dashboard for
  observing autoscaling.
- KServe autoscaling docs:
  https://kserve.github.io/website/latest/modelserving/autoscaling/autoscaling/
- KEDA Prometheus scaler:
  https://keda.sh/docs/latest/scalers/prometheus/
