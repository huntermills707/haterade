"""Jigsaw data loading via kagglehub + HF datasets.

The original 2018 competition ships train.csv with columns:
    id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate
Six binary labels per row. We hold out EVAL_SAMPLE_LIMIT rows from the end
of the train set for evaluation (the official test set has no labels).
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, Features, Sequence, Value
from transformers import PreTrainedTokenizerBase

from .env import Config


def find_train_csv(dataset_dir: Path) -> Path:
    """kagglehub may extract into a subfolder. Find the train.csv with
    the labeled columns regardless of layout."""
    candidates = list(dataset_dir.rglob("train.csv"))
    for c in candidates:
        try:
            head = pd.read_csv(c, nrows=0)
        except Exception:
            continue
        if "comment_text" in head.columns and "toxic" in head.columns:
            return c
    raise FileNotFoundError(
        f"No train.csv with Jigsaw columns under {dataset_dir} "
        f"(found: {[str(c) for c in candidates]})"
    )


def download_jigsaw(cfg: Config) -> Path:
    """Download (or hit kagglehub cache) the Jigsaw dataset and return
    the directory containing train.csv. Requires KAGGLE_TOKEN in env,
    which src.env.load_kaggle_token() sets before this is called."""
    import kagglehub
    path = Path(kagglehub.dataset_download(cfg.kaggle_dataset))
    return path


def load_jigsaw_split(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, eval_df) with comment_text + 6 label columns.
    Subsamples per cfg.train_sample_limit / cfg.eval_sample_limit."""
    csv = find_train_csv(download_jigsaw(cfg))
    df = pd.read_csv(csv)
    keep = ["comment_text", *cfg.label_columns]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"{csv} missing expected columns: {missing}")

    df = df[keep].dropna(subset=["comment_text"]).reset_index(drop=True)

    # Hold out the tail for eval. Stratification on a 6-label vector is
    # fiddly and unnecessary at this scale; a tail slice is reproducible
    # (seed-fixed via the data order from kaggle) and good enough for M1.
    n_eval = cfg.eval_sample_limit if cfg.eval_sample_limit > 0 else max(1, len(df) // 10)
    eval_df = df.tail(n_eval).reset_index(drop=True)
    train_df = df.iloc[: len(df) - n_eval].reset_index(drop=True)

    if cfg.train_sample_limit > 0 and len(train_df) > cfg.train_sample_limit:
        train_df = train_df.head(cfg.train_sample_limit).reset_index(drop=True)

    return train_df, eval_df


def tokenize(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    cfg: Config,
) -> Dataset:
    """Convert a Jigsaw DataFrame into a tokenized HF Dataset with
    `labels` as a float tensor (BCEWithLogitsLoss expects float [B,6])."""
    ds = Dataset.from_pandas(df, preserve_index=False)

    # Force the output schema on tokenization so labels land as float32.
    # Without this, HF infers float64 (from Python float()) and
    # BCEWithLogitsLoss (used by problem_type="multi_label_classification")
    # trips on a Float→Long cast inside the loss kernel.
    out_features = Features({
        "input_ids": Sequence(Value("int64")),
        "attention_mask": Sequence(Value("int64")),
        "labels": Sequence(Value("float32"), length=len(cfg.label_columns)),
    })

    def _tok(batch: dict[str, Any]) -> dict[str, Any]:
        enc = tokenizer(
            batch["comment_text"],
            padding="max_length",
            truncation=True,
            max_length=cfg.max_length,
        )
        n = len(enc["input_ids"])
        enc["labels"] = [
            [float(batch[c][i]) for c in cfg.label_columns] for i in range(n)
        ]
        return enc

    ds = ds.map(
        _tok,
        batched=True,
        batch_size=1000,
        remove_columns=ds.column_names,
        features=out_features,
    )
    ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    return ds
