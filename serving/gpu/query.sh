#!/usr/bin/env bash
# Smoke-test the deployed Triton toxicity model with a single tokenized input.
#
# Tokenization happens client-side for now. The KServe transformer (raw text in
# -> tokens -> predictor) is a separate milestone; without it, raw text cannot
# be sent to this predictor directly.
set -euo pipefail

ISVC_NAME="${ISVC_NAME:-toxicity-gpu}"
NAMESPACE="${NAMESPACE:-default}"
SEQ_LEN="${SEQ_LEN:-128}"
SAMPLE_TEXT="${SAMPLE_TEXT:-you are a wonderful person}"

for tool in kubectl curl python3 jq; do
  command -v "$tool" >/dev/null 2>&1 || { echo "missing: $tool"; exit 1; }
done
python3 -c "import transformers" 2>/dev/null || { echo "pip install transformers"; exit 1; }

echo "==> Tokenizing: $SAMPLE_TEXT"
PAYLOAD=$(python3 - "$SAMPLE_TEXT" "$SEQ_LEN" <<'PY'
import json, sys
from transformers import AutoTokenizer
text, seq_len = sys.argv[1], int(sys.argv[2])
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
enc = tok(text, padding="max_length", truncation=True, max_length=seq_len)
print(json.dumps({
    "inputs": [
        {"name": "input_ids",      "shape": [1, seq_len], "datatype": "INT64", "data": enc["input_ids"]},
        {"name": "attention_mask", "shape": [1, seq_len], "datatype": "INT64", "data": enc["attention_mask"]},
    ]
}))
PY
)

echo "==> Resolving ISVC host"
HOST=$(kubectl get inferenceservice "$ISVC_NAME" -n "$NAMESPACE" \
         -o jsonpath='{.status.url}' | sed -E 's|^https?://||')
[ -n "$HOST" ] || { echo "ISVC has no URL yet; is it ready?"; exit 1; }

echo "==> POST /v2/models/distilbert-toxicity/infer to $HOST"
curl -sS -X POST "http://${HOST}/v2/models/distilbert-toxicity/infer" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD" | jq .
