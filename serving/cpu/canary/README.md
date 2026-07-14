# M4 — Argo Rollouts canary for the CPU predictor

This directory contains the canary artifacts for milestone M4. The CPU
predictor is now an Argo Rollout (see `serving/cpu/rollout.yaml`), so Argo
Rollouts can shift Istio traffic between stable (v1) and canary (placeholder
v2) using Prometheus analysis gates.

## Files

| File | Purpose |
|---|---|
| `model-pvc.yaml` | PVC for the placeholder v2 model repository. |
| `build-canary-placeholder.sh` | Copies the verified v1 ONNX into the v2 PVC and bumps the Triton version label to 2. |
| `rollout-v2.yaml` | Rollout update that mounts the v2 PVC and triggers the canary. |

The stable Rollout, Services, VirtualService, KEDA ScaledObject, and
AnalysisTemplate live in `serving/cpu/`:

- `serving/cpu/rollout.yaml`
- `serving/cpu/scaledobject.yaml`
- `serving/cpu/triton-servicemonitor.yaml`
- `serving/cpu/analysis-template.yaml`

## Quickstart

1. **Deploy the stable predictor** (if not already deployed):

   ```bash
   kubectl apply -f serving/cpu/rollout.yaml
   kubectl apply -f serving/cpu/scaledobject.yaml
   kubectl apply -f serving/cpu/triton-servicemonitor.yaml
   kubectl apply -f serving/cpu/analysis-template.yaml
   ```

2. **Verify stable inference** works through the Istio gateway:

   ```bash
   ./serving/cpu/query.sh
   ```

3. **Build the placeholder v2 repository**:

   ```bash
   ./serving/cpu/canary/build-canary-placeholder.sh
   ```

4. **Start load** so the canary has traffic to analyze:

   ```bash
   # In another terminal
   export GATEWAY_IP=$(kubectl -n istio-ingress get svc istio-ingress -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
   cd traffic/
   ./run.sh
   ```

5. **Trigger the canary** by applying the v2 Rollout:

   ```bash
   kubectl apply -f serving/cpu/canary/rollout-v2.yaml
   ```

6. **Watch the rollout** progress:

   ```bash
   # Requires the kubectl argo rollouts plugin:
   #   https://argo-rollouts.readthedocs.io/en/stable/installation/#kubectl-plugin-installation
   kubectl argo rollouts get rollout toxicity-cpu --watch

   # Fallback without the plugin:
   kubectl get rollout toxicity-cpu -w
   ```

   Argo Rollouts will shift VirtualService weights 5% → 25% → 50% → 100%
   canary, running the AnalysisTemplate at each step.

7. **Promote or abort**:

   ```bash
   # If analysis passes and you want to complete the rollout:
   kubectl argo rollouts promote toxicity-cpu

   # If metrics look bad and you want to roll back:
   kubectl argo rollouts abort toxicity-cpu

   # Fallbacks without the plugin:
   kubectl patch rollout toxicity-cpu --type merge -p '{"spec":{"paused":false}}'
   kubectl patch rollout toxicity-cpu --type merge -p '{"spec":{"abort":true}}'
   ```

   Aborting reverts traffic to 100% stable.

8. **Observe**:

   ```bash
   # Watch VirtualService weights mutate
   kubectl get virtualservice toxicity-cpu -o yaml | grep -A 20 route

   # Watch replicas
   watch kubectl get pods -l app=toxicity-cpu

   # Grafana (M4 dashboard)
   kubectl -n observability port-forward svc/kube-prometheus-stack-grafana 3000:80
   ```

## How the placeholder v2 works

M5 will produce a genuinely retrained model. For M4, the "v2" artifact is the
same ONNX file as v1, placed under Triton version `2` in a separate PVC. The
inference contract is unchanged, but Triton metrics carry a `version="2"`
label so the AnalysisTemplate can measure canary-specific success rate and
latency.

## Replacing v2 with a real retrained model (M5)

Once M5 produces a new MLflow run:

1. Export the new ONNX via `serving/cpu/build-model-repo.sh` using the new
   `MLFLOW_RUN_ID`.
2. Copy the resulting model repo into `triton-cpu-canary-model-repo` instead
   of the placeholder.
3. Apply `serving/cpu/canary/rollout-v2.yaml` to run the canary again.

## Design note

M4 originally considered keeping the KServe InferenceService as the stable
predictor and adding a separate Rollout for canary. Argo Rollouts' Istio
integration requires owning both stable and canary Service selectors, so the
predictor was converted to an Argo Rollout. See
`docs/adr/0008-argo-rollouts-canary-with-kserve-rawdeployment.md`.
