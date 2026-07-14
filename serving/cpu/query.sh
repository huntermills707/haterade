#!/usr/bin/env bash
# Smoke-test the deployed MLServer toxicity model with a single tokenized input.
#
# Tokenization happens client-side — same pre-tokenized-input contract as
# serving/gpu/query.sh. The KServe transformer (raw text -> tokens) is a
# separate milestone; without it, raw text cannot be sent to this predictor.
#
# Routing: hits the Istio Gateway at its ServiceLB ExternalIP, with a
# `Host:` header matching the ISVC's URL. That avoids needing /etc/hosts
# entries for `*.example.com`. If you'd rather use the URL directly:
#   echo "<extIP> <host>" | sudo tee -a /etc/hosts
# then `URL=$HOST curl http://$URL/...` will work too.
set -euo pipefail

ISVC_NAME="${ISVC_NAME:-toxicity-cpu}"
NAMESPACE="${NAMESPACE:-default}"
SEQ_LEN="${SEQ_LEN:-128}"
# Export so the python subprocess in the decode step can read it.
export SAMPLE_TEXT="${SAMPLE_TEXT:-you are a wonderful person}"

# Prefer the repo-local training venv (it already has transformers pinned
# to match the trained model). Fall back to whatever `python3` is on PATH.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -x "$REPO_ROOT/training/.venv/bin/python" ]; then
  PYTHON="$REPO_ROOT/training/.venv/bin/python"
else
  PYTHON="python3"
fi

for tool in kubectl curl jq; do
  command -v "$tool" >/dev/null 2>&1 || { echo "missing: $tool"; exit 1; }
done
"$PYTHON" -c "import transformers" 2>/dev/null || {
  echo "transformers not importable by $PYTHON; install with:"
  echo "  $PYTHON -m pip install transformers"
  echo "or use the repo training venv: training/.venv/bin/pip install transformers"
  exit 1
}

echo "==> Tokenizing: $SAMPLE_TEXT"
PAYLOAD=$("$PYTHON" - "$SAMPLE_TEXT" "$SEQ_LEN" <<'PY'
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

echo "==> Resolving ISVC URL + Gateway ExternalIP"
# URL form: http://<name>-<namespace>.<ingressDomain> — we only want the host.
HOST=$(kubectl get inferenceservice "$ISVC_NAME" -n "$NAMESPACE" \
         -o jsonpath='{.status.url}' 2>/dev/null | sed -E 's|^https?://||')
[ -n "$HOST" ] || { echo "ISVC $ISVC_NAME has no URL yet; is it Ready?"; exit 1; }

EXT_IP=$(kubectl -n istio-ingress get svc istio-ingress \
           -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
[ -n "$EXT_IP" ] || {
  echo "no ServiceLB ExternalIP on istio-ingress; port-forward instead:"
  echo "  kubectl -n istio-ingress port-forward svc/istio-ingress 8080:80"
  echo "  curl -H \"Host: $HOST\" http://127.0.0.1:8080/..."
  exit 1
}

echo "==> POST /v2/models/$ISVC_NAME/infer"
echo "    via http://$EXT_IP/  (Host: $HOST)"
RESPONSE=$(curl -sS -X POST "http://$EXT_IP/v2/models/$ISVC_NAME/infer" \
  -H "Content-Type: application/json" \
  -H "Host: $HOST" \
  -d "$PAYLOAD")

echo "==> Raw V2 response:"
echo "$RESPONSE" | jq . 2>/dev/null || echo "$RESPONSE"

echo ""
echo "==> Decoded sigmoid scores per label:"
echo "$RESPONSE" | "$PYTHON" -c "
import json, sys, math, os
r = json.load(sys.stdin)
# mlserver-mlflow returns one V2 output per DataFrame column. Each has
# shape [1,1] and a single FP32 value (the logit for that label).
logits_by_name = {o['name']: o['data'][0] for o in r['outputs']}
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
text = os.environ.get('SAMPLE_TEXT', '?')
print(f'  input text: {text!r}')
for name in labels:
    raw = logits_by_name.get(name)
    if raw is None:
        print(f'  {name:15s} MISSING from response')
        continue
    p = 1.0 / (1.0 + math.exp(-raw))
    bar = '#' * int(p * 40)
    print(f'  {name:15s} {p:6.3f}  {bar}')
"
