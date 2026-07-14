# serving/cpu/ — DistilBERT on Triton (ONNX backend)

Serves the M1-trained DistilBERT toxicity classifier on the CPU cluster
using Triton's ONNX backend. As of M4 the predictor is an Argo Rollout so
we can run Istio traffic-split canaries with Prometheus analysis gates.

Status:
- M2/M3 verified (2026-07-14): inference through Istio Gateway works; KEDA
  scales the predictor 1 → 3 → 1 on Triton queue duration.
- M4 in progress: Argo Rollouts canary plumbing is scaffolded; live canary
  verification pending.

## Layout

```
serving/cpu/
├── rollout.yaml                          # Argo Rollout + Services + VirtualService
├── scaledobject.yaml                     # KEDA ScaledObject targeting the Rollout
├── model-repository/
│   └── distilbert-toxicity/
│       └── config.pbtxt                  # backend: onnxruntime, KIND_CPU, dynamic batching
├── model-pvc.yaml                        # PVC for the v1 model repository
├── triton-servicemonitor.yaml            # ServiceMonitor for Triton /metrics (port 8002)
├── build-model-repo.sh                   # MLflow → ONNX export → assemble repo → copy to PVC
├── query.sh                              # one-shot V2 smoke test (tokenize → POST → sigmoid)
├── README.md                             # this file
└── canary/                               # M4 canary artifacts
    ├── model-pvc.yaml                    # PVC for the placeholder v2 repo
    ├── build-canary-placeholder.sh       # copy v1 ONNX into v2 PVC + bump version label
    ├── analysis-template.yaml            # Prometheus success-rate + latency gates
    ├── rollout-v2.yaml                   # Rollout update that mounts v2 PVC
    └── README.md                         # canary walkthrough
```

## Topology

```
MLflow (M1 run) → MinIO (s3://mlflow/1/<run_id>/artifacts/model)
                     │
                     ▼  build-model-repo.sh (one-time, host-side)
                 model.onnx (256 MiB, opset 17, dynamic batch axis)
                     │
                     ▼  kubectl cp to PVC
                 triton-cpu-model-repo PVC
                     │
                     ▼  mounted at /mnt/models (readOnly)
             Triton (nvcr.io/nvidia/tritonserver:23.05-py3)
               onnxruntime backend loads model.onnx
                     │
                     ▼  KServe V2 HTTP/gRPC
   toxicity-cpu-stable / toxicity-cpu-canary Services
                     │
                     ▼  Istio VirtualService (managed by Argo Rollouts)
             Istio Gateway → ServiceLB ExternalIP
```

## Prerequisites

1. **M1 done.** A trained DistilBERT model in MLflow with a known `run_id`.
   The build script defaults to `4927d59563184da6a5861765de043394`
   (verified run, `auroc_macro=0.9795`). Override with `MLFLOW_RUN_ID=…`.

2. **MLflow port-forward.** The build script downloads artifacts:
   ```
   kubectl -n mlflow port-forward svc/mlflow 5000:5000
   ```

3. **training/.venv** with torch, transformers, mlflow, onnx.

4. **Platform stack installed** (`infra/install-platform-stack.sh`):
   - Istio ingress gateway in `istio-ingress` namespace
   - ServiceLB has allocated an ExternalIP on `svc/istio-ingress`
   - Argo Rollouts controller running in `argo-rollouts`
   - KEDA installed in `keda`

## Deploy

```bash
# 1. Build the model repo and populate the PVC (one-time per run_id)
MLFLOW_RUN_ID=4927d59563184da6a5861765de043394 ./serving/cpu/build-model-repo.sh

# 2. Apply the Rollout, KEDA autoscaler, and ServiceMonitor
kubectl apply -f serving/cpu/rollout.yaml
kubectl apply -f serving/cpu/scaledobject.yaml
kubectl apply -f serving/cpu/triton-servicemonitor.yaml

# 3. Wait for the Rollout to be healthy
kubectl argo rollouts get rollout toxicity-cpu --watch
```

First deploy takes ~3.5 min for the Triton image pull (~8 GB). Cached
thereafter; subsequent deploys are ~10s to healthy.

## Query

```bash
./serving/cpu/query.sh                                # default sample text
SAMPLE_TEXT="this is a hostile comment" ./serving/cpu/query.sh
```

Output:

```
==> Tokenizing: you are a wonderful person
==> POST /v2/models/distilbert-toxicity/infer
    via http://192.168.68.57/  (Host: toxicity-cpu-default.example.com)
==> Raw V2 response:
{
  "model_name": "distilbert-toxicity",
  "model_version": "1",
  "outputs": [
    {"name": "logits", "shape": [1, 6], "datatype": "FP32", "data": [-1.37, -2.25, ...]}
  ]
}

==> Decoded sigmoid scores per label:
  input text: 'you are a wonderful person'
  toxic            0.202  ########
  severe_toxic     0.095  ###
  ...
```

Note on the response shape: Triton returns a **single `[1,6]` logits
tensor** (one V2 output, 6 values). Same as the GPU path — the two
are interchangeable from the client's perspective.

## Inference contract

Pre-tokenized input — same as `serving/gpu/` until the KServe transformer
container lands (stretch goal). Six labels, multi-label (sigmoid, not
softmax). Order is fixed by M1 (`training/src/env.py:Config.label_columns`):

```
toxic, severe_toxic, obscene, threat, insult, identity_hate
```

```
POST /v2/models/distilbert-toxicity/infer
{
  "inputs": [
    {"name": "input_ids",      "shape": [1, 128], "datatype": "INT64", "data": [...]},
    {"name": "attention_mask", "shape": [1, 128], "datatype": "INT64", "data": [...]}
  ]
}
→ {"outputs": [{"name": "logits", "shape": [1, 6], "datatype": "FP32", "data": [-1.37, -2.25, ...]}]}
```

Apply `sigmoid` client-side to each logit to get per-class probabilities.

## Routing — how traffic reaches the predictor

The VirtualService host is `toxicity-cpu-default.example.com`
(KServe's `ingressDomain` default is `example.com`). That hostname
isn't DNS-resolvable from outside the cluster. `query.sh` works around
this by hitting the Gateway at its ServiceLB ExternalIP and setting
`Host: toxicity-cpu-default.example.com` as a header. Istio's Gateway
matches on the host header and routes to the VirtualService.

The VirtualService is now owned by Argo Rollouts. It routes between the
stable Service (`toxicity-cpu-stable`) and canary Service
(`toxicity-cpu-canary`). During a rollout, Argo Rollouts mutates the
weights to shift traffic progressively. See `serving/cpu/canary/README.md`
for the canary walkthrough.

## Comparison with `serving/gpu/`

| | `serving/cpu/` (this) | `serving/gpu/` |
|---|---|---|
| Runtime | `nvcr.io/nvidia/tritonserver:23.05-py3` | same |
| Backend | `onnxruntime` | `tensorrt` |
| Model file | `model.onnx` (256 MiB) | `model.plan` (TRT engine, sm_75-bound) |
| Model source | PVC (populated by `build-model-repo.sh`) | PVC (populated by `build-engine.sh`) |
| Export step | ONNX export from MLflow (`export_onnx.py`) | ONNX export → `trtexec` engine bake |
| Predictor owner | Argo Rollout (M4) | KServe ISVC |
| Autoscaler | KEDA: Triton queue depth | KEDA: Triton queue depth + DCGM GPU util |
| Resources | 1–2 CPU, 2–4 GiB | 2–4 CPU, 4–8 GiB + 1× nvidia.com/gpu |
| `nodeSelector` | none | `nvidia.com/gpu.present=true` |
| V2 output | `logits` `[B,6]` FP32 | same |

Both paths share the same Triton image, config format, metrics endpoint
(port 8002), dynamic batching config, and V2 I/O contract. The only
differences are the backend, the model file, GPU resources, and that the
CPU predictor is now managed by Argo Rollouts for canary delivery.

## Known limitations (this side)

- **Model is pinned to a specific `run_id`.** M5 (retrain) will produce a
  new run_id; re-run `build-model-repo.sh` and start a canary with
  `serving/cpu/canary/rollout-v2.yaml` to promote it.
- **Pre-tokenized input only.** Raw text → tokens is the KServe
  transformer stretch goal.
- **Triton image includes CUDA libraries it doesn't use.** The
  `nvcr.io/nvidia/tritonserver:23.05-py3` image is ~8 GB and includes
  the CUDA runtime, which is unused on CPU. A CPU-only Triton build
  exists upstream but is not published as an official image.
- **Argo Rollouts owns the predictor.** KServe's ISVC abstraction is no
  longer used for the CPU predictor. This was a deliberate M4 trade-off;
  see `docs/adr/0008-argo-rollouts-canary-with-kserve-rawdeployment.md`.
