"""Model + metrics for multi-label Jigsaw classification with DistilBERT."""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification

from .env import Config


def build_model(cfg: Config) -> AutoModelForSequenceClassification:
    """DistilBERT with a 6-way classification head. Labels are float
    targets so BCEWithLogitsLoss is used (set problem_type in the model
    so HF Trainer wires it up automatically)."""
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(cfg.label_columns),
        problem_type="multi_label_classification",
    )
    return model


def compute_metrics(cfg: Config):
    """Per-class + macro AUROC. Returns a callable compatible with HF
    Trainer. Falls back to per-class accuracy if only one class is
    present in eval (small EVAL_SAMPLE_LIMIT edge case)."""
    from sklearn.metrics import roc_auc_score

    label_cols = list(cfg.label_columns)

    def _metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        probs = torch.sigmoid(torch.as_tensor(logits, dtype=torch.float32)).numpy()
        labels = np.asarray(labels)
        metrics: dict[str, float] = {}

        per_class = []
        for i, name in enumerate(label_cols):
            y = labels[:, i]
            if y.sum() == 0 or y.sum() == len(y):
                # AUROC undefined when only one class is present — skip.
                continue
            try:
                score = float(roc_auc_score(y, probs[:, i]))
            except ValueError:
                continue
            metrics[f"auroc_{name}"] = score
            per_class.append(score)

        if per_class:
            metrics["auroc_macro"] = float(np.mean(per_class))
        return metrics

    return _metrics
