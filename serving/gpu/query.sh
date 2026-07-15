#!/usr/bin/env bash
# Smoke-test the deployed GPU Triton toxicity model with a single tokenized input.
#
# Tokenization happens client-side for now. The KServe transformer (raw text in
# → tokens → predictor) is a separate milestone; without it, raw text cannot be
# sent to this predictor directly.
set -euo pipefail

ISVC_NAME="${ISVC_NAME:-toxicity-gpu}"
NAMESPACE="${NAMESPACE:-default}"
SEQ_LEN="${SEQ_LEN:-128}"
SAMPLE_TEXT="${SAMPLE_TEXT:-you are a wonderful person}"
MODEL_NAME="${MODEL_NAME:-distilbert-toxicity}"

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
        {"name": "input_ids",      "shape": [1, seq_len], "datatype": "INT32", "data": enc["input_ids"]},
        {"name": "attention_mask", "shape": [1, seq_len], "datatype": "INT32", "data": enc["attention_mask"]},
    ]
}))
PY
)

echo "==> Resolving Gateway ExternalIP"
GATEWAY_IP=$(kubectl -n istio-ingress get svc istio-ingress -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
[ -n "$GATEWAY_IP" ] || { echo "no ExternalIP on istio-ingress gateway"; exit 1; }

HOST="${ISVC_NAME}-${NAMESPACE}.example.com"

echo "==> POST /v2/models/${MODEL_NAME}/infer"
echo "    via http://${GATEWAY_IP}/  (Host: ${HOST})"
RAW=$(curl -sS -X POST "http://${GATEWAY_IP}/v2/models/${MODEL_NAME}/infer" \
  -H "Content-Type: application/json" \
  -H "Host: ${HOST}" \
  -d "$PAYLOAD")

echo "==> Raw V2 response:"
echo "$RAW" | jq .

echo ""
echo "==> Decoded sigmoid scores per label:"
python3 - "$SAMPLE_TEXT" "$RAW" <<'PY'
import json, sys, math

text = sys.argv[1]
resp = json.loads(sys.argv[2])
logits = resp["outputs"][0]["data"]

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
print(f"  input text: '{text}'")
for label, logit in zip(labels, logits):
    prob = 1.0 / (1.0 + math.exp(-logit))
    bar = "#" * int(prob * 40)
    print(f"  {label:<16s} {prob:.3f}  {bar}")
PY
