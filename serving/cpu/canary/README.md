# M4/M5 — Argo Rollouts canary for the CPU predictor

This directory contains the canary artifacts for milestones M4 and M5. The CPU
predictor is an Argo Rollout (see `serving/cpu/rollout.yaml`), so Argo
Rollouts can shift Istio traffic between stable (v1) and canary (v2) using
Prometheus analysis gates.

In **M4** the v2 artifact was a placeholder — the same v1 ONNX under Triton
version `2`. In **M5** the canary is built from a real retrained MLflow run
and promoted automatically.

## Files

| File | Purpose |
|---|---|
| `model-pvc.yaml` | PVC for the canary model repository. |
| `build-canary-placeholder.sh` | **M4 only.** Copies the verified v1 ONNX into the v2 PVC and bumps the Triton version label to 2. |
| `build-canary-from-run.sh` | **M5.** Builds a real v2 repository from an MLflow run. |
| `promote-and-canary.sh` | **M5.** Validates a run, builds the canary, deploys it, and optionally promotes. |
| `rollback.sh` | Reverts to the stable v1 rollout. |
| `rollout-v2.yaml` | Rollout update that mounts the canary PVC and triggers the canary. |
| `servicemonitor.yaml` | ServiceMonitor so Prometheus scrapes canary pods. |

The stable Rollout, Services, VirtualService, KEDA ScaledObject, and
AnalysisTemplate live in `serving/cpu/`:

- `serving/cpu/rollout.yaml`
- `serving/cpu/scaledobject.yaml`
- `serving/cpu/triton-servicemonitor.yaml`
- `serving/cpu/analysis-template.yaml`

## Quickstart (M5 — real retrain + promotion)

1. **Retrain with the promotion gate enabled:**

   ```bash
   cd training
   MLFLOW_REGISTER_MODEL=true MLFLOW_PROMOTE_MODEL=true \
     .venv/bin/python -m training.train
   ```

   Note the `run_id` printed at the end.

2. **Run the automated canary promotion:**

   ```bash
   ./serving/cpu/canary/promote-and-canary.sh <run-id> --promote
   ```

   This validates the run, builds the canary v2 repository, applies
   `rollout-v2.yaml`, waits for the Rollout to become `Healthy`, and then
   promotes the canary to stable and marks the MLflow version as `Production`.

3. **If metrics look bad, roll back:**

   ```bash
   ./serving/cpu/rollback.sh
   ```

4. **Observe:**

   ```bash
   # Watch VirtualService weights mutate
   kubectl get virtualservice toxicity-cpu -o yaml | grep -A 20 route

   # Watch replicas
   watch kubectl get pods -l app=toxicity-cpu

   # Grafana (M4/M5 dashboard)
   kubectl -n observability port-forward svc/kube-prometheus-stack-grafana 3000:80
   ```

## Quickstart (M4 — placeholder v2)

The placeholder v2 workflow is kept for reproducing the original M4 demo.

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

## How the placeholder v2 works

M5 produces a genuinely retrained model. For M4, the "v2" artifact is the
same ONNX file as v1, placed under Triton version `2` in a separate PVC. The
inference contract is unchanged, but Triton metrics carry a `version="2"`
label so the AnalysisTemplate can measure canary-specific success rate and
latency.

## Replacing v2 with a real retrained model (M5)

Once M5 produces a new MLflow run:

1. Run `serving/cpu/canary/build-canary-from-run.sh <run-id>` to build the
   canary repository directly from the retrained artifacts.
2. Apply `serving/cpu/canary/rollout-v2.yaml` to run the canary.
3. (Optional) Run `serving/cpu/canary/promote-and-canary.sh <run-id>
   --promote` to validate, build, deploy, and promote automatically.

The M4 placeholder script is kept for historical reproduction but is no
longer the path to v2.

## Design note

M4 originally considered keeping the KServe InferenceService as the stable
predictor and adding a separate Rollout for canary. Argo Rollouts' Istio
integration requires owning both stable and canary Service selectors, so the
predictor was converted to an Argo Rollout.
