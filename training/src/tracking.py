"""MLflow client wiring + M5 promotion helpers.

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
    or None.

    The M5 promotion gate (training.src.promotion.stage_candidate) decides
    whether the new version is good enough to move to cfg.candidate_stage;
    this helper only creates the registry entry so the gate can transition
    it after the AUROC check."""
    if not cfg.register_model:
        return None
    client = MlflowClient(tracking_uri=cfg.tracking_uri)
    try:
        client.create_registered_model(cfg.registered_model_name)
    except mlflow.exceptions.MlflowException:
        # Already exists — fine.
        pass
    version = client.create_model_version(
        name=cfg.registered_model_name,
        source=model_uri,
        run_id=run_id,
    )
    return version.name


def get_production_version_info(
    cfg: Config,
) -> tuple[str | None, float | None]:
    """Return (run_id, auroc_macro) of the current Production model version.

    Returns (None, None) if no Production version exists. The caller decides
    whether to fall back to a configured threshold or fail open."""
    client = MlflowClient(tracking_uri=cfg.tracking_uri)
    try:
        versions = client.get_latest_versions(
            cfg.registered_model_name, stages=[cfg.production_stage]
        )
    except mlflow.exceptions.MlflowException:
        return None, None
    if not versions:
        return None, None
    prod = versions[0]
    run_id = prod.run_id
    run = client.get_run(run_id)
    auroc = run.data.metrics.get("auroc_macro")
    return run_id, auroc


def resolve_production_auroc_threshold(cfg: Config) -> float:
    """Return the AUROC threshold a candidate must beat.

    If cfg.production_auroc_threshold is explicitly set (> 0), use it.
    Otherwise use the current Production version's auroc_macro. If there is
    no Production version yet, return 0.0 (anything passes)."""
    if cfg.production_auroc_threshold > 0:
        return cfg.production_auroc_threshold
    _, prod_auroc = get_production_version_info(cfg)
    return prod_auroc if prod_auroc is not None else 0.0


def transition_to_production(cfg: Config, run_id: str) -> None:
    """Transition the model version for run_id to Production stage."""
    client = MlflowClient(tracking_uri=cfg.tracking_uri)
    versions = client.search_model_versions(
        f"name='{cfg.registered_model_name}' and run_id='{run_id}'"
    )
    if not versions:
        raise RuntimeError(
            f"No registered model version found for run {run_id}"
        )
    version = versions[0]
    client.transition_model_version_stage(
        name=cfg.registered_model_name,
        version=version.version,
        stage=cfg.production_stage,
        archive_existing_versions=True,
    )


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
