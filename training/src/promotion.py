"""M5 promotion gate.

Validates a candidate MLflow run against the current Production model and,
if it passes, transitions the candidate to the Staging stage in the MLflow
Model Registry. The canary deploy step consumes the Staging candidate.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `python -m training.src.promotion` work from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from training.src.env import from_env
from training.src.tracking import (
    MlflowClient,
    get_production_version_info,
    resolve_production_auroc_threshold,
    transition_to_production,
)


def validate_candidate(run_id: str | None = None) -> tuple[str, float, float, bool]:
    """Validate a candidate run against the production threshold.

    If run_id is None, reads CANDIDATE_RUN_ID from the environment.

    Returns:
        (candidate_run_id, candidate_auroc, threshold, passed)
    """
    cfg = from_env()
    if run_id is None:
        run_id = _require_env("CANDIDATE_RUN_ID")

    client = MlflowClient(tracking_uri=cfg.tracking_uri)
    candidate_run = client.get_run(run_id)
    candidate_auroc = candidate_run.data.metrics.get("auroc_macro")
    if candidate_auroc is None:
        raise ValueError(f"Run {run_id} has no auroc_macro metric")

    threshold = resolve_production_auroc_threshold(cfg)
    passed = candidate_auroc >= threshold

    prod_run_id, prod_auroc = get_production_version_info(cfg)
    print(f"  production run:      {prod_run_id or '(none)'}")
    print(f"  production auroc:    {prod_auroc if prod_auroc is not None else '(none)'}")
    print(f"  candidate run:       {run_id}")
    print(f"  candidate auroc:     {candidate_auroc:.4f}")
    print(f"  threshold:           {threshold:.4f}")
    print(f"  passed:              {passed}")

    return run_id, candidate_auroc, threshold, passed


def stage_candidate(run_id: str | None = None) -> None:
    """Validate and, if passing, transition the candidate to Staging."""
    cfg = from_env()
    if run_id is None:
        run_id = _require_env("CANDIDATE_RUN_ID")

    _, _, _, passed = validate_candidate(run_id)
    if not passed:
        raise SystemExit(1)

    client = MlflowClient(tracking_uri=cfg.tracking_uri)
    versions = client.search_model_versions(
        f"name='{cfg.registered_model_name}' and run_id='{run_id}'"
    )
    if not versions:
        raise RuntimeError(
            f"Run {run_id} is not registered under {cfg.registered_model_name}"
        )
    version = versions[0]
    client.transition_model_version_stage(
        name=cfg.registered_model_name,
        version=version.version,
        stage=cfg.candidate_stage,
        archive_existing_versions=False,
    )
    print(f"  transitioned version {version.version} to {cfg.candidate_stage}")


def promote_to_production(run_id: str | None = None) -> None:
    """Transition the candidate run's model version to Production."""
    cfg = from_env()
    if run_id is None:
        run_id = _require_env("CANDIDATE_RUN_ID")
    transition_to_production(cfg, run_id)
    print(f"  transitioned run {run_id} to {cfg.production_stage}")


def _require_env(name: str) -> str:
    import os
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} environment variable is required")
    return value


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="M5 promotion gate")
    parser.add_argument("--run-id", help="Candidate MLflow run_id")
    parser.add_argument(
        "action",
        choices=["validate", "stage", "promote"],
        default="validate",
        nargs="?",
        help="validate: print comparison; stage: validate + move to Staging; "
             "promote: move to Production",
    )
    args = parser.parse_args()

    if args.action == "validate":
        validate_candidate(args.run_id)
    elif args.action == "stage":
        stage_candidate(args.run_id)
    elif args.action == "promote":
        promote_to_production(args.run_id)
