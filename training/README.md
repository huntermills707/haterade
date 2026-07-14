# training/ — M1: Jigsaw DistilBERT → MLflow

Trains a `distilbert-base-uncased` classifier on the Jigsaw Toxic Comment
dataset, logs params/metrics/model to the in-cluster MLflow (CPU laptop,
short runs by default).

Status: scaffolded, smoke-tested for imports. Full end-to-end run
pending (see **Smoke test** below).

## Layout

```
training/
├── train.py            # entrypoint: `python -m training.train`
├── requirements.txt    # CPU-only torch + transformers + mlflow + kagglehub
├── .env.example        # all tunables with defaults
└── src/
    ├── env.py          # .env loader, Kaggle token loader, Config dataclass
    ├── data.py         # kagglehub download + Jigsaw CSV → tokenized HF Dataset
    ├── model.py        # DistilBERT (multi-label) + AUROC metrics
    └── tracking.py     # MLflow client wiring + Model Registry helper
```

## Prerequisites

1. **k3s up + MLflow reachable** — see repo root `README.md`. Forward MLflow:
   ```
   kubectl -n mlflow port-forward svc/mlflow 5000:5000
   ```
   `curl -s http://127.0.0.1:5000/health` should return 200.

2. **Kaggle API token.** The modern KGAT_ format (kaggle.com → Settings →
   API Tokens → "BasicMlOps") goes in `~/.kaggle/access_token`, mode 600.
   `train.py` reads it and sets `KAGGLE_TOKEN` for `kagglehub>=0.4.1`.
   **Do not** use the legacy `kaggle.json` flow here.

3. **Python 3.12 venv** (PyTorch wheels for 3.13+ are still spotty on
   some distros as of this writing):
   ```
   python3.12 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

## Run

```
cp .env.example .env    # edit if you want different defaults
.venv/bin/python -m training.train
```

What lands where:

| Thing | Location |
|---|---|
| Run + params + metrics | MLflow experiment `toxicity-distilbert` |
| Model artifact (`MLmodel`, weights) | same run, artifact_path `model` |
| HF `model_save/` + `tokenizer/` | same run, artifact_paths |
| Raw checkpoint dir | `training/runs/run-<ts>/` (outside MLflow) |
| Model Registry entry | Only if `MLFLOW_REGISTER_MODEL=true` |

## Tuning knobs

All env vars (env wins over `.env`). The defaults are deliberately small
for a ~3-min CPU smoke run:

| Var | Default | Effect |
|---|---|---|
| `TRAIN_SAMPLE_LIMIT` | `2000` | Head of 160k train rows used. `-1` = full data. |
| `EVAL_SAMPLE_LIMIT` | `200` | Held-out tail of train.csv (test.csv is unlabeled). |
| `EPOCHS` | `1` | |
| `TRAIN_BATCH_SIZE` | `16` | |
| `MAX_LENGTH` | `128` | Matches the Triton SEQ_LEN baked in M2. |
| `MODEL_NAME` | `distilbert-base-uncased` | HF hub id. |
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5000` | Port-forward of in-cluster svc. |
| `MLFLOW_REGISTER_MODEL` | `false` | Flip to `true` to publish to Model Registry. |

## Label contract

Six labels, this exact order — M2 (KServe) and M5 (retrain) depend on it:

```
toxic, severe_toxic, obscene, threat, insult, identity_hate
```

Multi-label (not multi-class): sigmoid, not softmax. Loss is
`BCEWithLogitsLoss` (set via HF `problem_type="multi_label_classification"`).

## Artifact storage flow

```
train.py → MLflow REST (localhost:5000) → MinIO S3 (s3://mlflow/)
                                            bucket on the cluster
```

Because MLflow is started with `--serve-artifacts` (see
`infra/manifests/mlflow.yaml`), the client never talks to MinIO directly.
No `AWS_*` creds or `MLFLOW_S3_ENDPOINT_URL` needed in the training env.

## Smoke test

```
# Verify env + Kaggle token + MLflow reachability without a real train.
.venv/bin/python -c "
from training.src.env import from_env, load_kaggle_token
cfg = from_env(); load_kaggle_token(cfg)
print('cfg ok:', cfg.model_name, cfg.train_sample_limit)
import mlflow
mlflow.set_tracking_uri(cfg.tracking_uri)
print('experiments:', [e.name for e in mlflow.MlflowClient().search_experiments()])
"
```

Expected: prints config + a `toxicity-distilbert` experiment (created on
first real run).

## Out of scope here

- GPU training (M0 GPU cluster pending hardware).
- KServe serving (M2 — consumes this run's `runs:/<id>/model` URI).
- Transformer container for raw-text input (stretch).
- Bigger backbone (ADR 0006 once filed).
