# 3. RawDeployment + KEDA over Serverless for KServe inference services

Date: 2026-07-14
Status: Accepted

## Context

KServe v0.19 supports two deployment modes for InferenceServices:

1. **Serverless** (default): backed by Knative Serving. Provides
   scale-to-zero natively via Knative's autoscaler (KPA). Traffic
   routing is owned by Knative's networking layer, which produces
   Knative Route + Configuration resources and manages Istio
   VirtualService weights for traffic splitting between revisions.

2. **RawDeployment**: backed by plain Kubernetes Deployment + Service.
   No built-in autoscaler. Traffic routing is left to standard K8s
   mechanisms (Ingress, Istio VirtualService, Gateway API). Autoscaling
   must be wired externally — KServe supports KEDA as an autoscaler
   class for this mode.

The project requires two capabilities that appear to conflict:

- **Scale-to-zero** (M3): when no inference traffic is flowing, the
  predictor pod should scale to 0 replicas to free resources. This is
  the headline M3 demo.

  > **Amendment (2026-07-14):** M3 was redefined as "traffic sim +
  > observe autoscaling" with `minReplicas: 1`. True scale-to-zero was
  > dropped per ADR 0007.

- **Canary traffic splitting** (M4): Argo Rollouts will progressively
  shift traffic between an old and new model version (e.g., 5% → 25% →
  100%) using Istio VirtualService weight manipulation, with Prometheus
  analysis gates at each step.

The conflict: Knative's networking layer also manipulates Istio
VirtualService weights to split traffic between revisions. Two
controllers writing VirtualService weights to the same resource is a
well-documented footgun — the last writer wins, traffic flaps, and
neither controller's intent is preserved. Argo Rollouts supports
traffic routing via Istio VirtualService, Nginx Ingress, and SMI
TrafficSplit — **not** Knative Route — so Knative's traffic-split
layer cannot be delegated to Argo.

## Decision

Use **RawDeployment + KEDA** for all InferenceServices on both clusters.

> **Amendment (2026-07-14):** M3 scope changed after live verification.
> True scale-to-zero was dropped; `minReplicas` is now `1` on the CPU
> predictor. See ADR 0007 for the rationale. The platform choice of
> RawDeployment + KEDA remains unchanged.

- `defaultDeploymentMode: RawDeployment` in the `inferenceservice-config`
  ConfigMap (set by `infra/install-platform-stack.sh:68-76`).
- `serving.kserve.io/autoscalerClass: keda` annotation on each ISVC.
- ~~`minReplicas: 0` on the predictor for scale-to-zero.~~
  `minReplicas: 1` on the CPU predictor (ADR 0007).
- KEDA `External` triggers backed by Prometheus queries against
  runtime-specific metrics.

> **Amendment (2026-07-14):** M4 further changed the CPU predictor
> architecture. Because Argo Rollouts' Istio integration requires owning
> both stable and canary Services, the CPU predictor is now an Argo
> Rollout rather than a KServe InferenceService. The GPU predictor remains
> a KServe ISVC. See ADR 0008.

### GPU scaling triggers (scaffolded, pending GPU hardware)

Defined in `serving/gpu/inferenceservice-triton.yaml`:

1. **Triton queue duration** (primary) —
   `avg(rate(nv_inference_queue_duration_us{model="distilbert-toxicity"}[30s]))`,
   threshold 50000 (µs/s ≈ 50 ms/s average queue pressure per pod).
   When requests pile up faster than they drain, scale out.
2. **DCGM GPU utilization** (backup) —
   `avg_over_time(DCGM_FI_DEV_GPU_UTIL{Namespace="default"}[60s])`,
   threshold 70 (%). Catches GPU saturation that queue depth might
   miss under batching.

Scraped via `serving/gpu/triton-servicemonitor.yaml` (ServiceMonitor
for Triton's `/metrics` endpoint on port 8002).

### CPU scaling trigger (verified against running Triton 2.34.0)

With ADR 0005 (Triton on both CPU and GPU), the CPU ISVC uses the same
Triton image and metrics as the GPU path. Triton exposes Prometheus
metrics on **port 8002** at `/metrics`. Verified live against the
`toxicity-cpu` predictor:

```
# counter — cumulative inference request duration (microseconds)
nv_inference_request_duration_us{model="distilbert-toxicity",version="1"} 168572

# counter — successful inference requests
nv_inference_request_success{model="distilbert-toxicity",version="1"} 2
```

The KEDA trigger for the CPU ISVC uses the same primary signal as GPU:

```promql
avg(rate(nv_inference_queue_duration_us{model="distilbert-toxicity"}[30s]))
```

Threshold: 50000 (µs/s, same as GPU). When requests pile up faster
than they drain, scale out. When traffic stops and the queue drains,
KEDA scales to zero after the cooldown.

**Why the same trigger on both clusters:** this is the primary benefit
of ADR 0005 — one runtime, one metric, one PromQL query. The only
difference is that the GPU ISVC adds a second trigger (DCGM GPU util)
that doesn't apply on CPU.

**Prerequisites for M3 implementation:**

1. **ServiceMonitor.** Already created at
   `serving/cpu/triton-servicemonitor.yaml` — targets port `metrics`
   (8002) on the `toxicity-cpu-predictor` Service.

2. **ISVC annotations.** The current `serving/cpu/inferenceservice-triton.yaml`
   ships with `minReplicas: 1, maxReplicas: 1` and KEDA annotations
   commented out. M3 must uncomment `autoscalerClass: keda`, set
   `minReplicas: 0`, and add the `autoScaling.metrics` block with the
   External trigger above.

   > **Amendment (2026-07-14):** The CPU predictor is no longer a KServe
   > ISVC. It was replaced by an Argo Rollout in M4. The equivalent KEDA
   > configuration now lives in `serving/cpu/scaledobject.yaml` and
   > targets the Rollout directly. The `serving/cpu/inferenceservice-triton.yaml`
   > file has been removed.

## Consequences

### Positive

- **No traffic-splitting conflict.** Argo Rollouts owns the
  VirtualService weights. No second controller fights over them.
  This is the primary reason for the decision — it unblocks M4.
- **Scale-to-zero works on both clusters.** KEDA's external scaler on
  Prometheus metrics handles the idle → zero transition. Both clusters
  use Triton's `nv_inference_queue_duration_us` as the primary trigger
  (ADR 0005 unified the runtime); GPU adds DCGM util as a backup.
- **Plain K8s resources.** Deployment + Service + Ingress/VirtualService.
  No Knative CRDs (Configuration, Route, Revision) to reason about.
  `kubectl get deployment` shows what's running; no indirection
  through Knative revisions.
- **Consistent with the project's "same manifests on both clusters"
  story.** RawDeployment is used identically on CPU and GPU. With ADR
  0005, the KEDA trigger queries are also identical (same Triton metric);
  only the GPU DCGM backup trigger differs.

### Negative

- **No Knative revision history.** Serverless mode tracks revisions
  with traffic-split metadata built in. With RawDeployment, M4's
  canary must be managed entirely by Argo Rollouts (which is the plan,
  but it means more Argo Rollouts YAML to write).

  > **Amendment (2026-07-14):** The plan changed from "more Argo
  > Rollouts YAML alongside KServe" to "Argo Rollouts owns the CPU
  > predictor entirely." See ADR 0008.
- **KEDA adds a moving part.** The KEDA operator + its ScaledObject
  controller must stay healthy. If KEDA goes down, autoscaling stops.
  Knative's autoscaler is embedded in the Serving controller — fewer
  components.
- **Scale-to-zero is slower than Knative's.** KEDA polls Prometheus
  at its configured interval (default 30s). Knative's KPA reacts to
  request concurrency in real time via the queue-proxy sidecar. For
  the demo this is fine (the narrative is "scale to zero saves
  resources," not "sub-second cold start"); for production it would
  matter.
- **Cold-start latency.** When scaling from 0 → 1, the predictor pod
  must be scheduled, the image pulled, the model loaded from MinIO
  (storage-initializer), and the runtime must load the model. Observed
  at ~10s for the CPU ISVC after image cache (Triton loads ONNX directly
  from PVC — no pip install, no pyfunc). Knative's pre-pull and
  pre-warm features would help here, but they're not available in
  RawDeployment.

## Alternatives considered

### Serverless (Knative Serving) + accept Knative traffic splitting

Use KServe's default Serverless mode. Get scale-to-zero for free via
KPA. Use Knative's own traffic splitting for canaries (Knative Route
with `traffic:` splits between revisions).

Rejected because:
- **Argo Rollouts can't control Knative Route traffic.** Argo
  Rollouts supports Istio VirtualService, Nginx Ingress, and SMI
  TrafficSplit for traffic routing — not Knative Route. The M4 canary
  plan depends on Argo Rollouts + Istio VirtualService weights, which
  Knative would overwrite.
- **Knative adds its own CRDs and controller.** Configuration, Route,
  Revision, Service, DomainMapping, ServerlessService — more surface
  area for a project that already runs Istio, KServe, KEDA, Argo
  Rollouts, and the Prometheus stack.
- **Knative's networking layer is opinionated.** It manages Istio
  VirtualServices for its own routing, which complicates the bundled
  VS pattern we use for ISVC routing (see
  `serving/cpu/inferenceservice-triton.yaml` and the Known
  Limitations entry in the root README).

### RawDeployment without KEDA (HPA only)

Use RawDeployment + Kubernetes HPA on CPU utilization. Simpler — no
KEDA operator needed.

Rejected because:
- **HPA can't scale to zero.** `minReplicas` on an HPA must be ≥ 1.
  The M3 demo's headline is scale-to-zero.
- **CPU utilization is a poor signal for inference workloads.** A
  single inference spikes CPU to 100% briefly, then drops to
  near-zero. HPA would thrash between 1 and 2 replicas on sparse
  traffic. Request concurrency (the gauge) is a cleaner signal: it's
  0 when idle and > 0 when any work is arriving.

### Serverless for scale-to-zero (M3) + switch to RawDeployment for canary (M4)

Use Serverless mode for M3, then migrate to RawDeployment when M4
lands.

Rejected because:
- **Migration cost.** Changing `defaultDeploymentMode` affects all
  existing ISVCs. Each would need to be deleted and recreated. The
  GPU ISVC is already scaffolded in RawDeployment; mixing modes adds
  cognitive overhead.
- **The conflict exists regardless of when you hit it.** Deferring the
  decision to M4 doesn't make it easier — it just means re-testing
  the serving path a second time.

> **Amendment (2026-07-14):** The project did end up making a
> significant serving-path change at M4, but it was moving the CPU
> predictor from a KServe ISVC to an Argo Rollout rather than toggling
> KServe deployment modes. The reason was the same: avoid two
> controllers fighting over the same resources.

## References

- KServe deployment modes:
  https://kserve.github.io/website/latest/modelserving/inference_service/
- KServe RawDeployment + KEDA autoscaling:
  https://kserve.github.io/website/latest/modelserving/autoscaling/autoscaling/
- KEDA Prometheus scaler:
  https://keda.sh/docs/latest/scalers/prometheus/
- Argo Rollouts traffic routing (supported providers):
  https://argo-rollouts.readthedocs.io/en/stable/features/traffic-management/
- GPU ISVC with KEDA triggers: `serving/gpu/inferenceservice-triton.yaml`
- GPU ServiceMonitor: `serving/gpu/triton-servicemonitor.yaml`
- Bootstrap (RawDeployment default): `infra/install-platform-stack.sh:68-76`
- KEDA install: `infra/install-platform-stack.sh:85-88`
- GPU ISVC with KEDA triggers: `serving/gpu/inferenceservice-triton.yaml`
- GPU ServiceMonitor: `serving/gpu/triton-servicemonitor.yaml`
- CPU predictor (Argo Rollout): `serving/cpu/rollout.yaml`
- CPU KEDA ScaledObject: `serving/cpu/scaledobject.yaml`
- CPU ServiceMonitor: `serving/cpu/triton-servicemonitor.yaml`
- Triton metrics (verified against `nvcr.io/nvidia/tritonserver:23.05-py3`
  on the running `toxicity-cpu` predictor, 2026-07-14)
- ADR 0005 (Triton on both clusters) — unifies the runtime and metrics
  across CPU and GPU.
- ADR 0002 (k3s for both clusters) — references KEDA + Argo Rollouts
  as shared platform stack components.
