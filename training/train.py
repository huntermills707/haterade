"""M5 entrypoint: train DistilBERT on Jigsaw, log to MLflow, and optionally
stage the best model for the canary promotion gate.

Usage:
    # 1. forward the in-cluster MLflow to localhost
    kubectl -n mlflow port-forward svc/mlflow 5000:5000 &

    # 2. (first time) install deps in a venv
    python3.12 -m venv training/.venv
    training/.venv/bin/pip install -r training/requirements.txt

    # 3. run
    cp training/.env.example training/.env   # then edit if needed
    training/.venv/bin/python -m training.train

Defaults (training/.env.example) target short CPU runs: 5k train rows,
1k eval rows, 1 epoch. Tune via env vars (env wins over .env).

Set MLFLOW_PROMOTE_MODEL=true to enable the M5 gate: the run is registered,
its eval auroc_macro is compared against the current Production threshold,
and passing runs are transitioned to Staging for the canary deploy.
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import mlflow

# Make `python -m training.train` work from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from training.src.env import Config, from_env, load_kaggle_token
from training.src.data import load_jigsaw_split, tokenize
from training.src.model import build_model, compute_metrics
from training.src.tracking import (
    init_tracking,
    maybe_register_model,
    log_training_summary,
    resolve_production_auroc_threshold,
)
from training.src.promotion import stage_candidate


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(False)  # triaged slow on CPU
    except Exception:
        pass


def main() -> int:
    cfg: Config = from_env()
    seed_everything(cfg.seed)

    print(f"== M1 — Jigsaw DistilBERT ==")
    print(f"  tracking_uri     = {cfg.tracking_uri}")
    print(f"  experiment       = {cfg.experiment_name}")
    print(f"  train/eval limit = {cfg.train_sample_limit}/{cfg.eval_sample_limit}")
    print(f"  epochs           = {cfg.epochs}")
    print(f"  batch            = {cfg.train_batch_size}")
    print(f"  max_length       = {cfg.max_length}")
    print(f"  model            = {cfg.model_name}")
    print(f"  register_model   = {cfg.register_model}")
    print(f"  promote_model    = {cfg.promote_model}")
    print(f"  prod_threshold   = {cfg.production_auroc_threshold or 'auto'}")

    load_kaggle_token(cfg)
    print(f"  kaggle token     = loaded ({cfg.kaggle_token_file})")

    # Late imports — torch/transformers are heavy; we want cfg printout
    # to appear first when something is going to be slow.
    import torch
    from transformers import (
        AutoTokenizer,
        TrainingArguments,
        Trainer,
    )

    client = init_tracking(cfg)

    print(f"\n[1/4] Loading Jigsaw…")
    train_df, eval_df = load_jigsaw_split(cfg)
    print(f"      train_rows={len(train_df)}  eval_rows={len(eval_df)}")

    print(f"\n[2/4] Tokenizing…")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_ds = tokenize(train_df, tokenizer, cfg)
    eval_ds = tokenize(eval_df, tokenizer, cfg)

    print(f"\n[3/4] Training (CPU)…")
    model = build_model(cfg)
    out_dir = cfg.output_dir / f"run-{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="auroc_macro",
        greater_is_better=True,
        logging_steps=25,
        report_to="none",  # MLflow handled explicitly below
        use_cpu=True,
        seed=cfg.seed,
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics(cfg),
    )

    run = client.create_run(
        experiment_id=client.get_experiment_by_name(cfg.experiment_name).experiment_id,
        tags={
            "framework": "transformers",
            "dataset": cfg.kaggle_dataset,
            "milestone": "M1",
        },
    )
    run_id = run.info.run_id
    print(f"      mlflow run_id  = {run_id}")
    print(f"      view: {cfg.tracking_uri.rstrip('/')}/#/experiments/"
          f"{run.info.experiment_id}/runs/{run_id}")

    # Activate this run as the global active run so mlflow.* helpers log
    # into it. Using client.create_run + mlflow.start_run(run_id=...) keeps
    # both APIs pointing at the same run.
    with mlflow.start_run(run_id=run_id):
        mlflow.log_params({
            "model_name": cfg.model_name,
            "epochs": cfg.epochs,
            "train_batch_size": cfg.train_batch_size,
            "eval_batch_size": cfg.eval_batch_size,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "max_length": cfg.max_length,
            "train_sample_limit": cfg.train_sample_limit,
            "eval_sample_limit": cfg.eval_sample_limit,
            "seed": cfg.seed,
            "torch_version": torch.__version__,
        })

        t0 = time.time()
        trainer.train()
        train_seconds = time.time() - t0
        mlflow.log_metric("train_seconds", train_seconds)
        print(f"      train_seconds = {train_seconds:.1f}s")

        print(f"      evaluating ({len(eval_df)} rows, batch={cfg.eval_batch_size})…")
        t0 = time.time()
        eval_metrics = trainer.evaluate()
        eval_seconds = time.time() - t0
        mlflow.log_metric("eval_seconds", eval_seconds)
        print(f"      eval_seconds  = {eval_seconds:.1f}s")
        # HF prefixes eval metrics with "eval_"; strip it for MLflow so
        # dashboard tiles line up across runs.
        cleaned = {k.removeprefix("eval_"): float(v) for k, v in eval_metrics.items()}
        mlflow.log_metrics(cleaned)
        log_training_summary(
            cfg, run_id, cleaned, len(train_df), len(eval_df)
        )

        print(f"\n[4/4] Logging model + tokenizer to MLflow…")
        model_dir = out_dir / "model"
        tokenizer_dir = out_dir / "tokenizer"
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(tokenizer_dir)

        # Log HF model + tokenizer as plain artifacts. The serving path
        # (Triton + ONNX, see ADR 0005) exports to ONNX from these
        # artifacts via serving/gpu/export_onnx.py — no pyfunc needed.
        # Previously this used mlflow.pyfunc.log_model with a custom
        # ToxicityV2Wrapper for MLServer (ADR 0004, superseded by 0005).
        print(f"      logging HF model artifacts…")
        t0 = time.time()
        mlflow.log_artifacts(str(model_dir), artifact_path="model")
        mlflow.log_artifacts(str(tokenizer_dir), artifact_path="tokenizer")
        print(f"      done in {time.time()-t0:.1f}s")

        # Belt + suspenders: client-side create_model_version in case the
        # auto-register above didn't fire (it's a no-op when off).
        maybe_register_model(cfg, f"runs:/{run_id}/model", run_id)

        if cfg.promote_model:
            print(f"\n[M5] Promotion gate")
            threshold = resolve_production_auroc_threshold(cfg)
            auroc = cleaned.get("auroc_macro")
            print(f"  candidate auroc_macro = {auroc:.4f}")
            print(f"  production threshold  = {threshold:.4f}")
            if auroc is None or auroc < threshold:
                print("  FAILED: candidate does not meet the production threshold")
                mlflow.set_tag("promotion", "failed")
                raise SystemExit(1)
            stage_candidate(run_id)
            mlflow.set_tag("milestone", "M5")
            mlflow.set_tag("promotion", "staged")
            print("  candidate staged for canary deploy")

    print(f"\n== done ==")
    print(f"  mlflow run:   {cfg.tracking_uri}/#/experiments/"
          f"{run.info.experiment_id}/runs/{run_id}")
    print(f"  eval auroc_macro: {cleaned.get('auroc_macro', 'n/a')}")
    print(f"  train_seconds: {train_seconds:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
