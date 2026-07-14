#!/usr/bin/env bash
# Convenience wrapper: start the M3 spiky Locust test against the CPU cluster.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -z "${GATEWAY_IP:-}" ]; then
    GATEWAY_IP=$(kubectl -n istio-ingress get svc istio-ingress -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)
    if [ -z "$GATEWAY_IP" ]; then
        echo "ERROR: GATEWAY_IP not set and could not be auto-resolved." >&2
        echo "       Make sure the CPU cluster is running and istio-ingress has an ExternalIP." >&2
        exit 1
    fi
    export GATEWAY_IP
fi

export ISVC_HOST="${ISVC_HOST:-toxicity-cpu-default.example.com}"
export SEQ_LEN="${SEQ_LEN:-128}"
export TARGET_RPS="${TARGET_RPS:-5}"

RUN_TIME="${RUN_TIME:-10m}"
export LOCUST_RUN_TIME="$RUN_TIME"

echo "==> Starting Locust against http://${GATEWAY_IP} (Host: ${ISVC_HOST})"
echo "    Run time: ${RUN_TIME}  |  Target RPS/user: ${TARGET_RPS}"

locust -f locustfile.py \
    --headless \
    "$@"
