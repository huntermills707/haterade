# Toxicity UI

A small web frontend + JSON API in front of the Triton-served toxicity model.
Triton only accepts pre-tokenized KServe V2 tensors, so this service does the
`distilbert-base-uncased` tokenization itself and calls
`POST /v2/models/distilbert-toxicity/infer` in-cluster.

Features:

- `GET /` — web page: paste text, get per-label toxicity scores.
- `POST /api/predict` `{"text": "..."}` — returns `{"id", "scores"}`.
  Every input is assigned a UUID and logged with its text and scores to
  `$DATA_DIR/predictions.jsonl`.
- `POST /api/feedback` `{"id": "...", "labels": {"toxic": true, ...}}` — for
  users who disagree with a prediction. Logged to `$DATA_DIR/feedback.jsonl`.
- `export_feedback.py` — joins the two logs on id, shuffles, and splits into
  Jigsaw-schema `feedback_train.csv` / `feedback_test.csv`.

## Config (env vars)

| Var | Default | Notes |
|---|---|---|
| `TRITON_INFER_URL` | `http://toxicity-cpu-stable.default.svc.cluster.local/v2/models/distilbert-toxicity/infer` | CPU cluster stable endpoint |
| `TRITON_DATATYPE` | `INT64` | GPU cluster (tensorrt build) needs `INT32` and the `toxicity-gpu-stable` URL |
| `SEQ_LEN` | `128` | must match the Triton model config |
| `MODEL_NAME` | `distilbert-base-uncased` | tokenizer |
| `DATA_DIR` | `/data` | PVC mount for the JSONL logs |

## Build & deploy (k3s, no registry)

```bash
docker build -t toxicity-ui:latest frontend/
docker save toxicity-ui:latest | sudo k3s ctr images import -
kubectl apply -f frontend/k8s.yaml
kubectl rollout status deploy/toxicity-ui
```

Then reach it through the same Istio gateway as everything else, via Host
header:

```bash
GATEWAY_IP=$(kubectl -n istio-ingress get svc istio-ingress -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl -H "Host: toxicity-ui-default.example.com" "http://$GATEWAY_IP/"            # UI
curl -H "Host: toxicity-ui-default.example.com" -H "Content-Type: application/json" \
  -d '{"text":"you are a wonderful person"}' "http://$GATEWAY_IP/api/predict"
```

For the GPU cluster, edit `TRITON_INFER_URL` / `TRITON_DATATYPE` in
`k8s.yaml` before applying.

## Closing the loop: feedback → training

```bash
# 1. Copy the logs out of the pod
POD=$(kubectl get pod -l app=toxicity-ui -o jsonpath='{.items[0].metadata.name}')
kubectl cp "default/$POD:/data" ./data

# 2. Split into train/test CSVs (Jigsaw schema)
python3 frontend/export_feedback.py --data-dir ./data

# 3. Train with feedback appended to the Jigsaw split
FEEDBACK_CSV_DIR=$PWD/data MLFLOW_REGISTER_MODEL=true MLFLOW_PROMOTE_MODEL=true \
  .venv/bin/python -m training.train    # from training/ per the main README
```

The usual M5 promotion gate (`training/src/promotion.py`) and the canary
pipeline (`serving/cpu/canary/promote-and-canary.sh`) are unchanged — a model
trained on feedback is only promoted if it beats the current Production
`auroc_macro`.

## Local development

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
TRITON_INFER_URL=http://<gateway-ip>/v2/models/distilbert-toxicity/infer \
DATA_DIR=./data .venv/bin/uvicorn app:app --port 8000
# open http://localhost:8000 — no Host header needed when hitting it directly
```
