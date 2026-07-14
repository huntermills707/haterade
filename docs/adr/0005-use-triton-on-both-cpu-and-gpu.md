# 5. Use Triton on both CPU and GPU clusters

Date: 2026-07-14
Status: Accepted
Supersedes: [0004 — Use MLServer for the MLflow serving handoff on CPU](0004-use-mlserver-for-mlflow-handoff-on-cpu.md)

## Context

ADR 0004 chose MLServer for the CPU serving path and Triton+TensorRT for
the GPU path, creating a two-runtime split:

- **CPU**: MLflow pyfunc → MLServer (Python runtime, zero ONNX export)
- **GPU**: MLflow → ONNX export → TRT engine → Triton (C++ runtime)

The justification was a "same model, two runtimes" comparison narrative
and avoiding an ONNX export step on CPU. On review, both justifications
are weaker than they appeared.

### ONNX export risk was overstated

ADR 0004 listed "export compatibility risk" and "subtly different
numerics" as reasons to avoid ONNX on the CPU path. This is wrong. ONNX
is a faithful serialization of the computational graph — the same
weights, the same operations, the same math. The export is a format
conversion, not an approximation. Any numerical difference between the
PyTorch model and its ONNX export is floating-point precision noise at
the level of `1e-7`, which is irrelevant for a sigmoid-thresholded
toxicity classifier. If a model's behavior meaningfully changes from an
ONNX round-trip, the model is overfit to implementation artifacts, not
to learned representations.

The DistilBERT ONNX export path is also well-trodden: `torch.onnx.export`
with opset 17 produces a valid graph that both Triton's ONNX backend and
TensorRT's engine builder consume without issue. Verified live during
this ADR (2026-07-14): the exported model has the correct I/O contract
(`input_ids` `[B,128]`, `attention_mask` `[B,128]` → `logits` `[B,6]`)
and is 256 MB — identical in size to the PyTorch state_dict.

### Two runtimes is unnecessary complexity

The MLServer path introduced:

1. **`ToxicityV2Wrapper`** — a custom `mlflow.pyfunc.PythonModel` needed
   because `mlflow.pytorch`'s pyfunc passes a single positional tensor
   to `forward()`, but DistilBERT requires named kwargs
   (`input_ids=`, `attention_mask=`). The wrapper also handles
   MLServer's pyfunc contract violation (passing `InferenceRequest`
   instead of DataFrame to `predict()`). This is permanent code coupled
   to MLServer internals.

2. **Version mismatch** — MLServer's `seldonio/mlserver:1.7.1` image
   ships torch 2.4.1 / transformers 4.51.3 / Python 3.10, while
   training uses torch 2.5.1 / transformers 4.46.3 / Python 3.12. The
   `extra_pip_requirements` in `mlflow.pyfunc.log_model` are advisory —
   MLServer logs warnings and uses the pre-baked versions. The model
   works by luck of forward-compatibility, not by design.

3. **V2 output asymmetry** — MLServer serializes the wrapper's DataFrame
   return as 6 separate `[1,1]` V2 outputs (one per column). Triton
   returns a single `[1,6]` logits tensor. The two paths have different
   client-side decoding logic, which is a documentation burden and a
   barrier to sharing test tooling.

4. **No dynamic batching on CPU** — MLServer processes requests
   individually. Triton's dynamic batching coalesces concurrent requests
   into batches for throughput, available on both CPU and GPU.

A single Triton runtime on both clusters eliminates all four issues.

## Decision

Use **Triton** (`nvcr.io/nvidia/tritonserver:23.05-py3`) on both
clusters:

- **CPU cluster**: ONNX backend (`backend: "onnx"`), `model.onnx`,
  `instance_group: KIND_CPU`, no GPU resources.
- **GPU cluster**: TensorRT backend (`backend: "tensorrt"`),
  `model.plan`, `instance_group: KIND_GPU`, fp16, GPU resources +
  nodeSelector.

Both share:
- Same Triton image (`nvcr.io/nvidia/tritonserver:23.05-py3`)
- Same `config.pbtxt` input/output spec (`input_ids` `[128] INT64`,
  `attention_mask` `[128] INT64` → `logits` `[6] FP32`)
- Same dynamic batching config
- Same V2 output shape (single `[B,6]` logits tensor)
- Same metrics endpoint (port 8002, `nv_inference_*` Prometheus metrics)
- Same ServiceMonitor pattern

### Model repository layout

```
<model-repo>/
  distilbert-toxicity/
    config.pbtxt          # backend differs (onnx vs tensorrt)
    1/
      model.onnx          # CPU
      # OR
      model.plan          # GPU (built from model.onnx via trtexec)
```

### ONNX export

The ONNX export is a post-training step (not part of `train.py`).
`serving/gpu/export_onnx.py` is reused — it's generic, taking any HF
model directory and producing an ONNX file with a dynamic batch axis.
The CPU build script (`serving/cpu/build-model-repo.sh`) downloads the
HF model from MLflow, exports to ONNX, assembles the model repo, and
copies to the PVC. The GPU build script (`serving/gpu/build-engine.sh`)
does the same but additionally runs `trtexec` to bake the TRT engine.

### What gets deleted

- `toxicity_wrapper.py` — no longer needed; Triton loads ONNX directly.
- `serving/cpu/inferenceservice-mlserver.yaml` — replaced by
  `inferenceservice-triton.yaml`.
- The `mlflow.pyfunc.log_model` call in `train.py` — replaced by plain
  `mlflow.log_artifacts` for the HF model directory. The ONNX export
  script downloads from MLflow and converts.

## Consequences

### Positive

- **One runtime, one mental model.** Triton on both clusters means one
  config format (`config.pbtxt`), one metrics endpoint (`/metrics` on
  8002), one V2 output shape (`[B,6]` logits), one set of KEDA triggers
  (`nv_inference_queue_duration_us`). Debugging knowledge transfers
  directly between CPU and GPU.
- **No wrapper, no version mismatch.** Triton loads an ONNX file, which
  is runtime-agnostic. There is no Python pyfunc, no `PythonModel`
  class, no `predict()` contract to violate, no torch/transformers/Python
  version coupling between training and serving. The ONNX graph is
  self-contained.
- **Dynamic batching on CPU.** Triton's dynamic batching coalesces
  concurrent requests into batches. On the CPU cluster this means
  better throughput under load (relevant for M3 traffic simulation).
- **Better CPU-vs-GPU comparison.** The "same model, two runtimes"
  narrative becomes "same runtime, two backends" — a cleaner comparison
  that isolates the execution backend (ONNX Runtime on CPU vs TensorRT
  on GPU) as the only variable.
- **Shared export step.** The ONNX export was already needed for the GPU
  TRT engine path. Using it for CPU too is zero net-new work — the
  export script already exists (`serving/gpu/export_onnx.py`) and is
  generic.
- **Shared query tooling.** `query.sh` on CPU and GPU can be identical
  (same V2 input, same V2 output shape, same decoding logic).

### Negative

- **Loses MLflow-native handoff on CPU.** The MLServer path pointed the
  ISVC directly at the MLflow artifact URI; the storage-initializer
  pulled the pyfunc model. With Triton, the model must be exported to
  ONNX and placed in a PVC-backed model repository — an extra step.
  This is the same step the GPU path already does, so it's not
  net-new complexity, but it is a step that MLServer avoided.
- **Triton image is larger.** `nvcr.io/nvidia/tritonserver:23.05-py3`
  is ~8 GB (includes CUDA libraries even on CPU). MLServer was ~2 GB.
  On a laptop with limited disk, this matters. The image is pulled once
  and cached, but the first deploy is slower.
- **Triton on CPU pulls CUDA libraries it doesn't use.** The standard
  Triton image includes the CUDA runtime, which is unused on a CPU-only
  node. This is cosmetic (the libraries are loaded but inert) but
  wastes ~4 GB of image layers. A CPU-only Triton build exists in
  upstream but is not published as an official image.
- **PVC instead of S3 pull.** The MLServer path used KServe's
  storage-initializer to pull from MinIO at deploy time. The Triton
  path uses a PVC that must be populated manually (via
  `build-model-repo.sh` + `kubectl cp`). This is less elegant but
  mirrors the GPU path exactly.

## Alternatives considered

### MLServer on CPU (ADR 0004, superseded)

The original M2 decision. Worked end-to-end but introduced the wrapper,
the version mismatch, the output asymmetry, and the lack of dynamic
batching. Superseded by this ADR.

### ONNX Runtime via KServe (not Triton)

Use KServe's ONNX runtime directly (not via Triton). Simpler than
Triton — no model repository, no `config.pbtxt`, just point the ISVC at
the ONNX file.

Rejected because:
- **Loses Triton's dynamic batching.** KServe's ONNX runtime processes
  requests individually, same as MLServer.
- **Loses `nv_inference_*` metrics.** Triton's queue depth metric is
  the KEDA trigger (see ADR 0003). KServe's ONNX runtime exposes
  different metrics, requiring a different PromQL query.
- **Breaks the "one runtime" goal.** We'd still need Triton on GPU for
  TensorRT. Two runtimes again, just different ones.
- **Not enough simpler to justify.** The model repository + PVC pattern
  is already built for the GPU path. Reusing it on CPU is less work
  than building a new ONNX-runtime ISVC from scratch.

### Raw Flask/FastAPI server

Same arguments as ADR 0004 — loses KServe integration, no V2 protocol,
no autoscaling, more code to maintain.

## References

- Triton Inference Server: https://docs.nvidia.com/deeplearning/triton-inference-server/
- Triton ONNX backend:
  https://github.com/triton-inference-server/onnxruntime_backend
- Triton model repository:
  https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/model_repository.html
- ONNX export script (shared): `serving/gpu/export_onnx.py`
- GPU build script (TRT engine): `serving/gpu/build-engine.sh`
- CPU build script (ONNX model repo): `serving/cpu/build-model-repo.sh`
- GPU ISVC manifest: `serving/gpu/inferenceservice-triton.yaml`
- CPU ISVC manifest: `serving/cpu/inferenceservice-triton.yaml`
- ADR 0003 (RawDeployment + KEDA) — Triton's `nv_inference_queue_duration_us`
  is the KEDA trigger on both clusters.
- ADR 0004 (MLServer, superseded) — the decision this ADR overturns.
- ADR 0006 (DistilBERT) — the model being served.
- ONNX export verified live: 2026-07-14, 256 MB, opset 17,
  `input_ids`/`attention_mask` `[B,128]` → `logits` `[B,6]`.
