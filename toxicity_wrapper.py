"""KServe V2 ↔ HF DistilBERT adapter for MLflow pyfunc deployment.

Why this exists: `mlflow.pytorch.log_model` produces a pyfunc whose
`predict()` only accepts pandas.DataFrame or numpy.ndarray (see
`mlflow/pytorch/__init__.py:777`). KServe's V2 protocol sends
multi-tensor input that MLServer decodes into a DataFrame where each
input is a column. DistilBERT's `forward(input_ids, attention_mask)`
expects a dict of named tensors, not a single positional tensor —
so `mlflow.pytorch`'s pyfunc raises `TypeError` on every inference.

This PythonModel wraps the HF model and bridges the two:
  V2 DataFrame {input_ids, attention_mask} → batched torch tensors →
  model(input_ids=…, attention_mask=…) → logits DataFrame

The external contract is unchanged from the original M1 ISVC: still
pre-tokenized input, still 6 sigmoid logits out. Parity with the GPU
Triton path is preserved.

Model loading: the artifact `model_path` points at a directory written
by `transformers.AutoModelForSequenceClassification.save_pretrained()`.
Loading via `from_pretrained(dir)` is version-tolerant — the safetensors
weights + config.json travel together, so the runtime image's torch/
transformers versions just need to be roughly compatible, not exact.
"""
from __future__ import annotations

import pandas as pd
import torch
from mlflow.pyfunc import PythonModel, PythonModelContext


LABEL_COLUMNS = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)


class ToxicityV2Wrapper(PythonModel):
    """Bridges KServe V2 multi-tensor input to a HF DistilBERT forward() call.

    Lifecycle: instantiated at training time by `mlflow.pyfunc.log_model`,
    pickled into the model artifact, and re-instantiated by MLServer at
    container startup. `load_context` runs once per process; `predict`
    runs per request.
    """

    def load_context(self, context: PythonModelContext) -> None:
        # Lazy import so `import training.src.wrapper` doesn't drag in
        # transformers/torch at unrelated call sites.
        from transformers import AutoModelForSequenceClassification

        model_path = context.artifacts["model_path"]
        # num_labels and problem_type MUST match training. They're in
        # config.json but `from_pretrained` doesn't read them from there
        # for the classification head — set explicitly to be safe.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=len(LABEL_COLUMNS),
            problem_type="multi_label_classification",
        )
        self.model.eval()
        # Stay on CPU — M2 is the CPU cluster. The GPU path uses Triton,
        # not MLServer, so we don't need device selection here.
        self.device = torch.device("cpu")

    def predict(
        self,
        context: PythonModelContext,
        model_input,
        params: dict | None = None,
    ) -> pd.DataFrame:
        # mlserver-mlflow's runtime calls predict with the V2
        # InferenceRequest object directly (NOT a pandas DataFrame,
        # despite the docstring contract — multi-input V2 has no clean
        # single-DataFrame representation). Reach into the inputs list
        # by name. Reference: mlserver_mlflow/runtime.py:202.
        import numpy as np

        if hasattr(model_input, "inputs"):
            # KServe V2 / mlserver path. Each RequestInput has name, shape,
            # datatype, data. data is a flat list; reshape using shape.
            by_name = {inp.name: inp for inp in model_input.inputs}
            ids_flat = by_name["input_ids"].data
            ids_shape = by_name["input_ids"].shape
            mask_flat = by_name["attention_mask"].data
            mask_shape = by_name["attention_mask"].shape
            ids_arr = np.asarray(ids_flat, dtype=np.int64).reshape(ids_shape)
            mask_arr = np.asarray(mask_flat, dtype=np.int64).reshape(mask_shape)
        elif hasattr(model_input, "columns"):
            # pandas DataFrame path — what tests / direct pyfunc calls use.
            ids_arr = np.asarray(list(model_input["input_ids"]), dtype=np.int64)
            mask_arr = np.asarray(list(model_input["attention_mask"]), dtype=np.int64)
        else:
            raise ValueError(
                "predict() expected an InferenceRequest (V2 path) or a "
                "pandas DataFrame with `input_ids` and `attention_mask` "
                f"columns; got {type(model_input).__name__}"
            )

        input_ids = torch.tensor(ids_arr, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(mask_arr, dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # outputs.logits: [B, 6]. Return as a typed DataFrame so MLServer
        # can map to V2 outputs cleanly.
        return pd.DataFrame(
            outputs.logits.cpu().numpy(),
            columns=list(LABEL_COLUMNS),
        )
