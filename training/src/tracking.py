"""MLflow client wiring.

The in-cluster MLflow server is started with `--serve-artifacts`, so the
client talks only to MLFLOW_TRACKING_URI and MLflow proxies artifacts to
MinIO on its side. No S3 creds or MLFLOW_S3_ENDPOINT_URL needed here.
"""
from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient

from .env import Config


def init_tracking(cfg: Config) -> MlflowClient:
    """Idempotent: set tracking URI, ensure experiment exists, return client."""
    mlflow.set_tracking_uri(cfg.tracking_uri)
    client = MlflowClient(tracking_uri=cfg.tracking_uri)
    exp = client.get_experiment_by_name(cfg.experiment_name)
    if exp is None:
        exp_id = client.create_experiment(cfg.experiment_name)
        exp = client.get_experiment(exp_id)
    assert exp is not None
    mlflow.set_experiment(cfg.experiment_name)
    return client


def maybe_register_model(
    cfg: Config, model_uri: str, run_id: str
) -> str | None:
    """Register the model under cfg.registered_model_name if
    MLFLOW_REGISTER_MODEL=true. Returns the model name (if registered)
    or None. Existing registered model + version alias is left to the
    promotion step (M5)."""
    if not cfg.register_model:
        return None
    client = MlflowClient(tracking_uri=cfg.tracking_uri)
    try:
        client.create_registered_model(cfg.registered_model_name)
    except mlflow.exceptions.MlflowException:
        # Already exists — fine.
        pass
    result = client.create_model_version(
        name=cfg.registered_model_name,
        source=model_uri,
        run_id=run_id,
    )
    return result.name


def log_training_summary(
    cfg: Config,
    run_id: str,
    eval_metrics: dict[str, float],
    train_df_rows: int,
    eval_df_rows: int,
) -> None:
    """Log the non-HF-Trainer side artifacts: dataset shapes + a small
    JSON manifest the KServe deploy step (M2) can consume."""
    import json
    client = MlflowClient(tracking_uri=cfg.tracking_uri)

    client.log_metric(run_id, "train_rows", float(train_df_rows))
    client.log_metric(run_id, "eval_rows", float(eval_df_rows))

    manifest = {
        "model_name": cfg.model_name,
        "labels": list(cfg.label_columns),
        "max_length": cfg.max_length,
        "problem_type": "multi_label_classification",
        "loss": "BCEWithLogitsLoss",
    }
    client.log_dict(run_id, json.dumps(manifest, indent=2), "manifest.json")
