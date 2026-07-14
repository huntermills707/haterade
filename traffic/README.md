# Traffic simulation (M3)

Locust load test for the CPU Triton toxicity predictor. Generates spiky
traffic so KEDA scales the predictor up under load and back down to the
minimum replica count when idle.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

1. Resolve the Istio ingress gateway IP:

```bash
export GATEWAY_IP=$(kubectl -n istio-ingress get svc istio-ingress -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
```

2. Start Locust with the built-in spiky wave shape:

```bash
locust -f locustfile.py --headless --run-time 10m
```

Or use the helper script:

```bash
./run.sh
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `GATEWAY_IP` | required | Istio ingress gateway external IP |
| `ISVC_HOST` | `toxicity-cpu-default.example.com` | VirtualService host header |
| `SEQ_LEN` | `128` | Token sequence length |
| `TARGET_RPS` | `5` | Target requests per second per user |

## Traffic shape

`SpikeWave` in `locustfile.py` produces a 5-minute repeating cycle:

- 0–30 s: 1 user baseline
- 30–120 s: ramp 1 → 20 users
- 120–240 s: 20 users sustained
- 240–300 s: ramp down to 1 user

Set `RUN_TIME` (e.g. `5m`, `300s`) to stop after a fixed duration; otherwise the
wave loops forever.

## Autoscaling scope

The CPU ISVC uses a Triton pod-level metric (`nv_inference_queue_duration_us`)
as its KEDA trigger. Per [ADR 0007](../docs/adr/0007-drop-scale-to-zero-keep-min-one-replica.md),
true scale-to-zero is intentionally out of scope for RawDeployment M3. The
deployment keeps `minReplicas: 1` so the metric is always available, and KEDA
scales it 1 → N → 1 under load.

## Observing autoscaling

While Locust runs, watch the predictor pods:

```bash
watch kubectl get pods -l serving.kserve.io/inferenceservice=toxicity-cpu
```

Or query KEDA directly:

```bash
kubectl get scaledobject toxicity-cpu-predictor
kubectl describe scaledobject toxicity-cpu-predictor
```

The Prometheus query driving the scaler is:

```promql
avg(rate(nv_inference_queue_duration_us{model="distilbert-toxicity"}[30s]))
```

KEDA scales up when the average queue pressure per pod exceeds `50000` µs/s
(≈ 50 ms/s) and scales back down to `minReplicas` after the cooldown.
