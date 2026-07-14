# serving/cpu/ — M2: DistilBERT on MLServer via KServe

Serves the M1-trained DistilBERT toxicity classifier on the CPU cluster
using KServe's `kserve-mlserver` runtime + MLflow artifact handoff.

Status: scaffolded + spike-verified (2026-07-13). Production ISVC waits
on first deploy (`kubectl apply -f inferenceservice-mlserver.yaml`).

## Layout

```
serving/cpu/
├── inferenceservice-mlserver.yaml   # KServe InferenceService (RawDeployment)
├── query.sh                         # one-shot V2 smoke test with Host header
└── README.md                        # this file
```

## Topology

```
MLflow (M1 run) → MinIO (s3://mlflow/1/<run_id>/artifacts/model)
                     │
                     ▼  storage-initializer (init container)
                 /mnt/models/{MLmodel, data/model.pth, …}
                     │
                     ▼  kserve-mlserver (seldonio/mlserver:1.7.1)
              mlserver-mlflow runtime loads via pyfunc
                     │
                     ▼  KServe V2 HTTP/gRPC
              Service (ClusterIP) → Istio Gateway → ServiceLB ExternalIP
```

## Prerequisites

1. **M1 done.** A trained DistilBERT model in MLflow with a known `run_id`.
   The manifest pins to `1/18c785f7036143869547d97fc2476c40/artifacts/model`
   (verified run, `auroc_macro=0.9795`). Swap the path for a different run.

2. **Platform stack installed** (`infra/install-platform-stack.sh`):
   - `kserve-mlserver` ClusterServingRuntime present
   - `storage-config` Secret in `default` ns with `localMinio` JSON blob
   - `kserve-ingress-gateway` Gateway in `kserve` ns
   - ServiceLB has allocated an ExternalIP on `svc/istio-ingress`

3. **KServe in RawDeployment mode** (verified live 2026-07-13 as part of M2).

## Deploy

```
kubectl apply -f serving/cpu/inferenceservice-mlserver.yaml
kubectl get isvc toxicity-cpu -w     # wait for Ready=True
```

First deploy takes ~2.5 min: storage-initializer pulls the 256 MiB model
from MinIO (~1s), then MLServer image is pulled fresh (~2 min on first
run, cached thereafter), then the model loads (~5 s).

## Query

```
./serving/cpu/query.sh                                # default sample text
SAMPLE_TEXT="this is a hostile comment" ./serving/cpu/query.sh
ISVC_NAME=toxicity-cpu ./serving/cpu/query.sh         # explicit name
```

Output:

```
==> Tokenizing: you are a wonderful person
==> POST /v2/models/toxicity-cpu/infer
    via http://192.168.68.57/  (Host: toxicity-cpu-default.example.com)
==> Raw V2 response:
{
  "outputs": [
    {"name": "toxic",         "shape": [1,1], "datatype": "FP32", "data": [-1.68]},
    {"name": "severe_toxic",  "shape": [1,1], "datatype": "FP32", "data": [-2.54]},
    ...
  ]
}

==> Decoded sigmoid scores per label:
  input text: 'you are a wonderful person'
  toxic            0.202  ########
  severe_toxic     0.095  ###
  ...
```

Note on the response shape: mlserver-mlflow returns **one V2 output per
DataFrame column** (6 separate `[1,1]` outputs, not a single `[1,6]`
tensor). The wrapper returns a typed DataFrame; mlserver splits it
across columns at serialization. `query.sh` decodes by name.

## Inference contract

Pre-tokenized input — same as `serving/gpu/` until the KServe transformer
container lands (stretch goal). Six labels, multi-label (sigmoid, not
softmax). Order is fixed by M1 (`training/src/env.py:Config.label_columns`):

```
toxic, severe_toxic, obscene, threat, insult, identity_hate
```

```
POST /v2/models/toxicity-cpu/infer
{
  "inputs": [
    {"name": "input_ids",      "shape": [1, 128], "datatype": "INT64", "data": [...]},
    {"name": "attention_mask", "shape": [1, 128], "datatype": "INT64", "data": [...]}
  ]
}
→ {"outputs": [
     {"name": "toxic",         "shape": [1,1], "datatype": "FP32", "data": [-1.68]},
     {"name": "severe_toxic",  "shape": [1,1], "datatype": "FP32", "data": [-2.54]},
     ...one V2 output per DataFrame column (6 total)
   ]}
```

Apply `sigmoid` client-side to each logit to get per-class probabilities.

## Routing — how traffic reaches the predictor

The ISVC's `.status.url` is `http://toxicity-cpu-default.example.com`
(KServe's `ingressDomain` default is `example.com`). That hostname
isn't DNS-resolvable from outside the cluster. `query.sh` works around
this by hitting the Gateway at its ServiceLB ExternalIP and setting
`Host: toxicity-cpu-default.example.com` as a header. Istio's Gateway
matches on the host header and routes to the right VirtualService.

The VirtualService is **bundled into `inferenceservice-mlserver.yaml`**
as a second document — `kubectl apply -f` creates both. KServe v0.19
RawDeployment does create a K8s `networking.k8s.io/Ingress`
automatically, but our istio-ingress proxy doesn't serve that resource
type (only Gateway CRDs), so the bundled VS is what actually routes
traffic. See root README "Known limitations" for the longer-term fix
paths.

To use a real hostname instead:
```
echo "192.168.68.57 toxicity-cpu-default.example.com" | sudo tee -a /etc/hosts
```
(then `curl http://toxicity-cpu-default.example.com/v2/...` works directly).

For a permanent shorter domain, set `ingress.ingressDomain` in the
`inferenceservice-config` ConfigMap (e.g. to `"local"`).

## Differences from `serving/gpu/` (the Triton path)

| | `serving/cpu/` (this) | `serving/gpu/` |
|---|---|---|
| Runtime | `kserve-mlserver` (Seldon MLServer 1.7.1) | `nvcr.io/nvidia/tritonserver:23.05-py3` |
| Model source | S3 pull from MinIO at deploy time (storage-initializer) | Pre-baked TensorRT plan on a PVC |
| Handoff story | MLflow artifact → live prediction | ONNX → trtexec → TRT plan → Triton |
| Autoscaler | (deferred to M3) | KEDA: Triton queue depth + DCGM GPU util |
| Resources | 1–2 CPU, 2–4 GiB | 2–4 CPU, 4–8 GiB + 1× nvidia.com/gpu |
| `nodeSelector` | none | `nvidia.com/gpu.present=true` |

## Known limitations (this side)

- **Model is pinned to a specific `run_id`.** M5 (retrain) will produce a
  new run_id; we'll handle promotion via Argo Rollouts, not by editing
  this manifest. Alternatively, register the model in MLflow and switch
  to a `models:/…/<version>` URI once the storage-initializer is taught
  to resolve MLflow URIs.
- **Python/package version skew.** MLServer 1.7.1 image is Python 3.10;
  M1 trained on 3.12.13. Model loaded fine in the M2 spike (2026-07-13)
  but inference results may subtly differ from training-time expectations
  due to torch/transformers version drift. Long-term fix is a custom
  MLServer image with our training-time pins (ADR 0004 candidate).
- **No autoscaling yet.** M2 ships fixed at 1 replica. M3 adds KEDA on
  request concurrency + scale-to-zero.
- **Pre-tokenized input only.** Raw text → tokens is the KServe
  transformer stretch goal.
