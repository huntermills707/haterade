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
as its KEDA trigger. True scale-to-zero is intentionally out of scope for
RawDeployment M3 — it is not supported without Knative, and the workarounds
cost more than they return here. The
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

## Known noise: gevent shutdown traceback

Every `locust` invocation — including `locust --version` — ends with an
`Exception ignored ... RuntimeError: greenlet is being finalized` traceback on
stderr, raised from `gevent/thread.py` while the logging module tears down its
weakref handlers.

It is cosmetic. The exception is raised during interpreter finalization, after
the run has completed; Python swallows it and the process still exits 0. Load
test results are unaffected.

Reproduced identically on Python 3.12.13 and 3.14.6 with gevent 25.9.1, so it
is a gevent/greenlet interaction, not a Python version problem — pinning the
interpreter does not help. `requirements.txt` floats gevent transitively via
`locust>=2.32,<3.0`; pinning gevent below 25.9 would be the lever if the noise
ever needs to go away.
