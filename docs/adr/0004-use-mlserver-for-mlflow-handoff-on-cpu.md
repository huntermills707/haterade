# 4. Use MLServer for the MLflow serving handoff on CPU

Date: 2026-07-14
Status: Superseded by [0005 — Use Triton on both CPU and GPU clusters](0005-use-triton-on-both-cpu-and-gpu.md)

## Context

M2 serves the M1-trained DistilBERT toxicity classifier on the CPU
cluster. The model is logged to MLflow as a pyfunc artifact (run
`4927d59563184da6a5861765de043394`). The serving runtime must:

1. **Pull the artifact from MLflow/MinIO** at deploy time, with no
   manual export or conversion step.
2. **Speak the KServe V2 protocol** (pre-tokenized `input_ids` +
   `attention_mask` in, 6 sigmoid logits out).
3. **Run on CPU** with no GPU dependencies.
4. **Integrate with KServe's ISVC abstraction** — the runtime must be
   discoverable by `modelFormat` + `runtime` so KServe's controller
   can reconcile it.

The project's narrative depends on a **"same model, two runtimes"**
comparison: the CPU path (MLflow → MLServer, zero export) vs the GPU
path (MLflow → ONNX → TRT engine → Triton, optimized). The CPU path
is intentionally the simpler one — its value is showing the baseline
cost of serving a Python model with no compilation, against which the
GPU path's optimization payoff can be measured.

### The pyfunc wrapper problem

The first M2 attempt used `mlflow.pytorch.log_model`, which produces
a pyfunc whose `predict()` passes a single positional tensor to
`model.forward(input)`. DistilBERT's `forward()` requires named
kwargs — `forward(input_ids=…, attention_mask=…)` — so the pyfunc
raises `TypeError: forward() got an unexpected keyword argument` on
every V2 inference. MLServer's mlflow runtime decodes the V2 request
into a DataFrame with `input_ids` and `attention_mask` columns, but
`mlflow.pytorch`'s pyfunc wrapper passes the entire DataFrame as one
positional argument.

This is a known limitation of `mlflow.pytorch`'s pyfunc flavor, not a
bug in our code — the flavor assumes single-tensor models (CNNs, MLPs)
and doesn't handle multi-input transformer architectures.

## Decision

Use **MLServer** (`kserve-mlserver` runtime, `seldonio/mlserver:1.7.1`
image) with the **MLflow pyfunc backend** and a custom
`ToxicityV2Wrapper` PythonModel.

### How the model is logged

`training/train.py` uses `mlflow.pyfunc.log_model` (not
`mlflow.pytorch.log_model`) with:

```python
mlflow.pyfunc.log_model(
    python_model=ToxicityV2Wrapper(),           # custom PythonModel
    artifact_path="model",
    artifacts={"model_path": str(model_dir)},   # HF save_pretrained() dir
    code_path=["toxicity_wrapper.py"],           # ships the wrapper module
    extra_pip_requirements=["torch==2.5.1", "transformers==4.46.3"],
)
```

### What ToxicityV2Wrapper does

`toxicity_wrapper.py` (repo root, not under `training/`) implements
`mlflow.pyfunc.PythonModel` with two methods:

- **`load_context`**: loads the HF model from `context.artifacts["model_path"]`
  via `AutoModelForSequenceClassification.from_pretrained()`. Sets
  `num_labels=6` and `problem_type="multi_label_classification"` explicitly
  (they're in config.json but `from_pretrained` doesn't read them for the
  classification head).

- **`predict`**: bridges two input shapes:
  1. **V2 path** (production): MLServer passes an `InferenceRequest` object
     (not a DataFrame — despite the pyfunc contract, `mlserver-mlflow`
     calls `predict(inference_request)` directly). The wrapper extracts
     `input_ids` and `attention_mask` by name from `request.inputs`,
     reshapes via the V2 shape field, and converts to torch tensors.
  2. **DataFrame path** (tests/direct pyfunc calls): standard pandas
     DataFrame with `input_ids` and `attention_mask` columns.

  Returns a typed DataFrame with 6 columns (`toxic`, `severe_toxic`,
  `obscene`, `threat`, `insult`, `identity_hate`). MLServer serializes
  each column as a separate V2 output — the client sees 6 named
  `[1,1]` FP32 outputs, not a single `[1,6]` tensor.

### Why the wrapper lives at the repo root

`code_path` ships files alongside the model artifact. If the wrapper
were under `training/`, `code_path=["training"]` would ship the entire
training directory — including `training/runs/` with multi-GB
checkpoint folders. Placing `toxicity_wrapper.py` at the repo root and
using `code_path=["toxicity_wrapper.py"]` ships a single 4 KB file.

### Version mismatch reality

The `extra_pip_requirements` in `log_model` are **advisory, not
enforced**. MLServer logs warnings at startup and uses whatever is
pre-baked in the `seldonio/mlserver:1.7.1` image:

| Package       | Pinned (training) | Actual (MLServer image) |
|---------------|-------------------|-------------------------|
| torch         | 2.5.1             | 2.4.1                   |
| transformers  | 4.46.3            | 4.51.3                  |
| pandas        | 2.1.4             | 2.2.3                   |
| scipy         | 1.17.1            | 1.15.3                  |
| cloudpickle   | 3.1.2             | 3.1.1                   |
| Python        | 3.12.13           | 3.10.12 (conda)         |

The model loads and serves correctly despite every one of these
mismatches. DistilBERT's `from_pretrained()` is version-tolerant
(safetensors weights + config.json are forward/backward compatible
across minor versions). The cloudpickle 3.1.2 → 3.1.1 deserialization
works because the pickle protocol hasn't changed between these patches.

This is fragile by accident, not by design. If a future transformers
release changes the `forward()` signature or the `save_pretrained()`
format, the model could fail to load at serving time with no test
coverage catching it first (the training venv and the MLServer image
have different versions).

## Consequences

### Positive

- **Zero export step.** The MLflow artifact is the serving artifact.
  No ONNX export, no TRT engine baking, no format conversion. `git
  push` → train → log to MLflow → ISVC pulls the same artifact. This
  is the primary value — the CPU path has the fewest moving parts
  between "model trained" and "model served."
- **MLflow-native handoff.** The ISVC's `storage.path` points directly
  at the MLflow run's `artifacts/model` directory in MinIO. KServe's
  storage-initializer pulls it using the `storage-config` Secret. No
  intermediate registry, no manual copy step.
- **KServe V2 protocol.** MLServer speaks V2 natively. The ISVC's
  `protocolVersion: v2` is honored without custom HTTP handling.
  Same input/output contract as the GPU Triton path.
- **Fast cold-start.** ~3s from container start to "model loaded"
  (no pip install — pre-baked libraries are used despite version
  warnings). Total ISVC Ready time ~20s including scheduling +
  storage-initializer pull.
- **Prometheus metrics on port 8082.** `rest_server_requests_in_progress`
  (gauge) and `rest_server_requests_total` (counter) are available
  for KEDA autoscaling in M3 (see ADR 0003).

### Negative

- **Custom wrapper is permanent.** `ToxicityV2Wrapper` is not a
  temporary workaround — it's a required component of the serving
  path for as long as the model is served via MLflow pyfunc. If the
  model architecture changes (e.g., a different transformer with
  different `forward()` kwargs), the wrapper must be updated and the
  model re-logged.
- **No dynamic batching.** MLServer processes requests individually.
  Triton's dynamic batching (see `serving/gpu/model-repository/
  distilbert-toxicity/config.pbtxt`) coalesces requests into batches
  for throughput — MLServer has no equivalent. This is acceptable for
  the CPU demo (single laptop, low RPS) but would be a bottleneck at
  production scale.
- **Version drift risk.** The training venv (torch 2.5.1, Python
  3.12) and the MLServer image (torch 2.4.1, Python 3.10) are
  different environments. The model works today by luck of
  forward-compatibility, not by design. A breaking change in
  transformers or torch would fail at serving time, not at training
  time. There is no CI step that validates the model artifact
  against the MLServer image.
- **`extra_pip_requirements` are misleading.** They suggest the
  pinned versions will be installed at serving time. In reality,
  MLServer logs warnings and uses whatever is in the image. The
  pins are documentation, not enforcement. This could confuse a
  reviewer reading `train.py` into thinking the versions are
  controlled.
- **MLServer's pyfunc contract violation.** MLServer passes an
  `InferenceRequest` object to `predict()`, not a DataFrame as
  the `PythonModel` contract specifies. The wrapper handles both
  shapes, but this means the wrapper is coupled to MLServer's
  implementation detail, not just to the MLflow pyfunc spec. If
  MLServer changes how it calls `predict()` in a future version,
  the wrapper must be updated.
- **6 separate V2 outputs.** MLServer serializes the DataFrame
  return as one V2 output per column (6 × `[1,1]` FP32), not a
  single `[1,6]` tensor. This is surprising compared to Triton's
  single `logits` output. `serving/cpu/query.sh` handles it, but
  the asymmetry with the GPU path is a documentation burden.

## Alternatives considered

### `mlflow.pytorch.log_model` (the first attempt)

The natural choice — log the PyTorch model directly, let MLflow's
pyfunc flavor handle serving. Rejected because its pyfunc wrapper
calls `model(input)` with a single positional argument, which
DistilBERT's `forward()` rejects (requires `input_ids` + 
`attention_mask` as named kwargs). This is the reason
`ToxicityV2Wrapper` exists.

### Triton on CPU (PyTorch backend)

Use the same Triton runtime as the GPU path, but with the PyTorch
backend (no TRT engine) on CPU. The model would be exported to a
Triton model repository with `backend: "pytorch"`.

Rejected because:
- **Loses the MLflow handoff.** Triton reads from a model repository
  (PVC or local dir), not from MLflow. We'd need a manual copy step
  from MLflow → PVC, which is what the GPU path does (and has its
  own complexity). The CPU path's value is avoiding exactly this.
- **Double config maintenance.** Two Triton `config.pbtxt` files
  (one for PyTorch CPU, one for TRT GPU) with different backends,
  different instance groups, different batching configs.
- **No CPU-specific advantage.** Triton's value is dynamic batching
  + GPU execution. On CPU with low RPS, neither matters for the demo.

### ONNX Runtime via KServe

Export DistilBERT to ONNX (`optimum.exporters`), serve via KServe's
ONNX runtime or Triton's ONNX backend on CPU.

Rejected because:
- **Adds an export step.** The ONNX file is a new artifact to manage,
  version, and validate. The MLflow → MLServer path has zero export.
- **Export compatibility risk.** DistilBERT ONNX export is well-trodden,
  but it's still a conversion step that could fail or produce subtly
  different numerics. Not worth the risk for the CPU baseline.
- **The GPU path already covers ONNX.** The TRT engine is built from
  an ONNX export (see `serving/gpu/`). Having the CPU path also use
  ONNX would collapse the "two runtimes" comparison into "one runtime,
  two backends" — less interesting for the portfolio narrative.

### Raw Flask/FastAPI server

Wrap the HF model in a Flask app, serve behind a K8s Deployment +
Service. Maximum control, no KServe/MLServer dependency.

Rejected because:
- **Loses KServe integration.** No ISVC abstraction, no
  storage-initializer, no V2 protocol, no KEDA autoscaling without
  custom wiring. The project's value is in the platform integration,
  not the model serving logic.
- **More code to maintain.** Health checks, metrics endpoints,
  model loading, request parsing — all hand-rolled. MLServer gives
  these for free.
- **Breaks the "same manifests on both clusters" story.** The GPU
  path uses KServe ISVC; the CPU path would use a raw Deployment.
  Asymmetric and harder to reason about.

## References

- MLServer: https://mlserver.readthedocs.io/
- MLServer MLflow runtime:
  https://mlserver.readthedocs.io/en/latest/runtimes/mlflow.html
- MLflow PythonModel:
  https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#pythonmodel
- KServe supported runtimes:
  https://kserve.github.io/website/latest/modelserving/v1beta1/supported-runtimes/
- Wrapper implementation: `toxicity_wrapper.py`
- Training entrypoint (log_model call): `training/train.py:196-212`
- CPU ISVC manifest: `serving/cpu/inferenceservice-mlserver.yaml`
- GPU ISVC manifest (Triton path for comparison):
  `serving/gpu/inferenceservice-triton.yaml`
- ADR 0003 (RawDeployment + KEDA) — MLServer is the runtime that
  KEDA scales on the CPU cluster.
- ADR 0006 (DistilBERT over BERT) — the model being served.
- MLServer startup logs (version mismatch warnings): verified live
  against `toxicity-cpu` predictor, 2026-07-14.
