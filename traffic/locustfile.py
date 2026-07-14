"""Locust load test for the Triton toxicity predictor.

Generates spiky, tokenized inference traffic against the KServe V2 endpoint
through the Istio ingress gateway. Designed to make KEDA scale the predictor
up from the minimum replica count under load and back down when idle.

Environment variables:
  GATEWAY_IP    Istio ingress gateway external IP (required).
  ISVC_HOST     Host header for routing, default toxicity-cpu-default.example.com.
  SEQ_LEN       Token sequence length, default 128.
  TARGET_RPS    Steady-state target requests per second per user, default 5.
"""
import json
import math
import os
import random
from typing import Optional

from locust import FastHttpUser, LoadTestShape, between, events, task
from transformers import AutoTokenizer

GATEWAY_IP = os.getenv("GATEWAY_IP")
ISVC_HOST = os.getenv("ISVC_HOST", "toxicity-cpu-default.example.com")
SEQ_LEN = int(os.getenv("SEQ_LEN", "128"))
TARGET_RPS = float(os.getenv("TARGET_RPS", "5"))
MODEL_NAME = "distilbert-toxicity"

SAMPLE_TEXTS = [
    "you are a wonderful person",
    "this is a great comment thank you",
    "i completely disagree but respect your opinion",
    "you are the worst kind of idiot",
    "shut up and go away",
    "this article is trash and so are the authors",
    "i love this post it made my day",
    "kill yourself you worthless human",
    "thanks for sharing this was really helpful",
    "nobody cares what you think loser",
]


class TokenizerCache:
    """Load the tokenizer once per worker process."""

    _tokenizer: Optional[AutoTokenizer] = None

    @classmethod
    def get(cls) -> AutoTokenizer:
        if cls._tokenizer is None:
            cls._tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        return cls._tokenizer


@events.init.add_listener
def on_locust_init(environment, **kwargs):
    if not GATEWAY_IP:
        raise RuntimeError(
            "GATEWAY_IP environment variable is required. "
            "Resolve it with: kubectl -n istio-ingress get svc istio-ingress"
        )
    # Force FastHttpUser base_url to the gateway so Host header routing works.
    environment.host = f"http://{GATEWAY_IP}"


class TritonToxicityUser(FastHttpUser):
    """Single tokenized inference request per task iteration."""

    # Fixed_wait gives a rough target RPS per user independent of response time.
    wait_time = between(max(0.05, 1.0 / TARGET_RPS - 0.05), 1.0 / TARGET_RPS + 0.05)

    def on_start(self):
        self.tokenizer = TokenizerCache.get()
        self.headers = {
            "Host": ISVC_HOST,
            "Content-Type": "application/json",
        }

    @task
    def infer(self):
        text = random.choice(SAMPLE_TEXTS)
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=SEQ_LEN,
        )
        payload = {
            "inputs": [
                {
                    "name": "input_ids",
                    "shape": [1, SEQ_LEN],
                    "datatype": "INT64",
                    "data": enc["input_ids"],
                },
                {
                    "name": "attention_mask",
                    "shape": [1, SEQ_LEN],
                    "datatype": "INT64",
                    "data": enc["attention_mask"],
                },
            ]
        }
        with self.client.post(
            f"/v2/models/{MODEL_NAME}/infer",
            headers=self.headers,
            data=json.dumps(payload),
            catch_response=True,
            name="/v2/models/distilbert-toxicity/infer",
        ) as response:
            if response.status_code == 200:
                try:
                    body = response.json()
                    logits = body["outputs"][0]["data"]
                    if len(logits) == 6:
                        response.success()
                    else:
                        response.failure(f"unexpected logits length: {len(logits)}")
                except Exception as exc:
                    response.failure(f"invalid response: {exc}")
            else:
                response.failure(f"status {response.status_code}: {response.text[:200]}")


class SpikeWave(LoadTestShape):
    """Spiky traffic: 30 s quiet, 90 s ramp, 120 s sustain, 60 s ramp down.

    The wave repeats so the demo can show multiple scale-up/scale-down cycles.
    If LOCUST_RUN_TIME is set (e.g. "5m", "300s"), the shape stops after that
    duration; otherwise it loops forever.
    """

    WAVE_SECONDS = 300  # 5 minutes per cycle

    def __init__(self):
        super().__init__()
        raw = os.getenv("LOCUST_RUN_TIME", "")
        self.max_run_time: Optional[float] = None
        if raw:
            try:
                # Accept integer seconds or Go-style duration like "5m", "300s".
                total = 0.0
                remaining = raw.strip()
                for suffix, factor in [("h", 3600), ("m", 60), ("s", 1)]:
                    if suffix in remaining:
                        part, remaining = remaining.split(suffix, 1)
                        total += float(part) * factor
                if remaining:
                    total += float(remaining)
                self.max_run_time = total
            except ValueError:
                pass

    def tick(self):
        run_time = self.get_run_time()
        if self.max_run_time is not None and run_time >= self.max_run_time:
            return None

        phase = run_time % self.WAVE_SECONDS

        if phase < 30:
            # Baseline — one user keeps a minimal warm load on the predictor.
            user_count = 1
        elif phase < 120:
            # Ramp from 1 to 20 users.
            user_count = 1 + int(19 * (phase - 30) / 90)
        elif phase < 240:
            # Sustain peak load.
            user_count = 20
        else:
            # Ramp down.
            user_count = max(1, int(20 * (self.WAVE_SECONDS - phase) / 60))

        spawn_rate = max(1, user_count)
        return user_count, spawn_rate
