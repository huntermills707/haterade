"""Environment + secret loading.

Reads .env (if present), loads the Kaggle KGAT_ token from disk into
KAGGLE_TOKEN, and returns typed config to the entrypoint. No secrets
are logged.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on", "y"}


def _int(v: Optional[str], default: int) -> int:
    try:
        return int(v) if v is not None and v.strip() != "" else default
    except ValueError:
        return default


def _float(v: Optional[str], default: float) -> float:
    try:
        return float(v) if v is not None and v.strip() != "" else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    # Kaggle
    kaggle_token_file: Path
    kaggle_dataset: str

    # MLflow
    tracking_uri: str
    experiment_name: str
    register_model: bool
    registered_model_name: str
    promote_model: bool
    production_auroc_threshold: float
    candidate_stage: str
    production_stage: str

    # Training
    train_sample_limit: int
    eval_sample_limit: int
    max_length: int
    train_batch_size: int
    eval_batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    seed: int
    model_name: str
    output_dir: Path

    @property
    def label_columns(self) -> tuple[str, ...]:
        # Jigsaw multi-label contract. Keep order stable — M2/KServe and
        # the GPU Triton plan both depend on this exact ordering.
        return ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")


def load_dotenv_if_present() -> None:
    """Best-effort .env loader. No dependency on python-dotenv — the format
    we support is a flat KEY=VALUE file with leading `export ` optional
    and `#` comments. Skips if no .env so this is safe to call unconditionally."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        # Don't clobber real env vars — env wins.
        os.environ.setdefault(k, v)


def load_kaggle_token(cfg: Config) -> None:
    """Read the raw KGAT_... token from disk into KAGGLE_TOKEN.
    kagglehub >= 0.4.1 picks up KAGGLE_TOKEN automatically."""
    path = cfg.kaggle_token_file.expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Kaggle token file not found at {path}. "
            "Either create it (KGAT_... token from kaggle.com/settings → "
            "API Tokens) or point KAGGLE_TOKEN_FILE at it."
        )
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        # Don't hard-fail — kagglehub tolerates it, but warn loudly.
        print(f"WARNING: {path} is mode {mode:o} (group/other can read). "
              f"Run:  chmod 600 {path}")
    token = path.read_text().strip()
    if not token.startswith("KGAT_"):
        raise ValueError(
            f"{path} does not contain a KGAT_... token (modern Kaggle API "
            "token). Re-download from kaggle.com/settings → API Tokens."
        )
    os.environ["KAGGLE_TOKEN"] = token


def from_env() -> Config:
    load_dotenv_if_present()
    cfg = Config(
        kaggle_token_file=Path(
            os.environ.get("KAGGLE_TOKEN_FILE", "~/.kaggle/access_token")
        ),
        kaggle_dataset=os.environ.get(
            "KAGGLE_DATASET",
            "julian3833/jigsaw-toxic-comment-classification-challenge",
        ),
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
        experiment_name=os.environ.get("MLFLOW_EXPERIMENT", "toxicity-distilbert"),
        register_model=_bool(os.environ.get("MLFLOW_REGISTER_MODEL"), False),
        registered_model_name=os.environ.get(
            "MLFLOW_REGISTERED_MODEL_NAME", "distilbert-toxicity"
        ),
        promote_model=_bool(os.environ.get("MLFLOW_PROMOTE_MODEL"), False),
        production_auroc_threshold=_float(
            os.environ.get("MLFLOW_PRODUCTION_AUROC_THRESHOLD"), 0.0
        ),
        candidate_stage=os.environ.get("MLFLOW_CANDIDATE_STAGE", "Staging"),
        production_stage=os.environ.get("MLFLOW_PRODUCTION_STAGE", "Production"),
        train_sample_limit=_int(os.environ.get("TRAIN_SAMPLE_LIMIT"), 2000),
        eval_sample_limit=_int(os.environ.get("EVAL_SAMPLE_LIMIT"), 200),
        max_length=_int(os.environ.get("MAX_LENGTH"), 128),
        train_batch_size=_int(os.environ.get("TRAIN_BATCH_SIZE"), 16),
        eval_batch_size=_int(os.environ.get("EVAL_BATCH_SIZE"), 32),
        epochs=_int(os.environ.get("EPOCHS"), 1),
        learning_rate=_float(os.environ.get("LEARNING_RATE"), 2e-5),
        weight_decay=_float(os.environ.get("WEIGHT_DECAY"), 0.01),
        seed=_int(os.environ.get("SEED"), 42),
        model_name=os.environ.get("MODEL_NAME", "distilbert-base-uncased"),
        output_dir=Path(os.environ.get("OUTPUT_DIR", "training/runs")),
    )
    return cfg
