"""Toxicity UI: raw-text frontend for the Triton-served DistilBERT model.

Triton only accepts pre-tokenized KServe V2 tensors (see serving/cpu/query.sh),
so this service tokenizes raw text, calls the V2 /infer endpoint, applies
sigmoid to the logits, and returns per-label scores.

Every input is assigned a UUID and logged (text + scores) as JSONL on the
PVC-mounted DATA_DIR. Users who disagree with a prediction can submit their
own labels, which are logged alongside. export_feedback.py joins the two
logs into Jigsaw-schema train/test CSVs for the next training run
(FEEDBACK_CSV_DIR in training).
"""
from __future__ import annotations

import json
import math
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

# Label contract — must match training/src/env.py:label_columns exactly.
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

TRITON_INFER_URL = os.environ.get(
    "TRITON_INFER_URL",
    "http://toxicity-cpu-stable.default.svc.cluster.local"
    "/v2/models/distilbert-toxicity/infer",
)
# CPU cluster Triton uses INT64 inputs; the GPU tensorrt build uses INT32.
TRITON_DATATYPE = os.environ.get("TRITON_DATATYPE", "INT64")
TRITON_TIMEOUT_S = float(os.environ.get("TRITON_TIMEOUT_S", "30"))
SEQ_LEN = int(os.environ.get("SEQ_LEN", "128"))
MODEL_NAME = os.environ.get("MODEL_NAME", "distilbert-base-uncased")
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))

STATIC_DIR = Path(__file__).resolve().parent / "static"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_log_lock = threading.Lock()

app = FastAPI(title="toxicity-ui")


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)


class FeedbackRequest(BaseModel):
    id: str = Field(min_length=1)
    labels: dict[str, bool]


def _log_jsonl(filename: str, record: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with _log_lock, (DATA_DIR / filename).open("a", encoding="utf-8") as f:
        f.write(line + "\n")


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/predict")
def predict(req: PredictRequest) -> dict:
    enc = tokenizer(
        req.text, padding="max_length", truncation=True, max_length=SEQ_LEN
    )
    payload = {
        "inputs": [
            {
                "name": "input_ids",
                "shape": [1, SEQ_LEN],
                "datatype": TRITON_DATATYPE,
                "data": enc["input_ids"],
            },
            {
                "name": "attention_mask",
                "shape": [1, SEQ_LEN],
                "datatype": TRITON_DATATYPE,
                "data": enc["attention_mask"],
            },
        ]
    }
    try:
        resp = requests.post(TRITON_INFER_URL, json=payload, timeout=TRITON_TIMEOUT_S)
    except requests.RequestException as e:
        raise HTTPException(502, f"Triton unreachable: {e}") from e
    if resp.status_code != 200:
        raise HTTPException(502, f"Triton {resp.status_code}: {resp.text[:500]}")

    logits = resp.json()["outputs"][0]["data"]
    scores = {label: 1.0 / (1.0 + math.exp(-x)) for label, x in zip(LABELS, logits)}

    pred_id = uuid.uuid4().hex
    _log_jsonl(
        "predictions.jsonl",
        {
            "id": pred_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "text": req.text,
            "scores": scores,
        },
    )
    return {"id": pred_id, "scores": scores}


@app.post("/api/feedback")
def feedback(req: FeedbackRequest) -> dict:
    unknown = set(req.labels) - set(LABELS)
    if unknown:
        raise HTTPException(400, f"unknown labels: {sorted(unknown)}")
    _log_jsonl(
        "feedback.jsonl",
        {
            "id": req.id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "labels": {label: bool(req.labels.get(label, False)) for label in LABELS},
        },
    )
    return {"status": "recorded", "id": req.id}
