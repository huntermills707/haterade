# 10. Automated v2 retrain and promotion

Date: 2026-07-14
Status: Accepted

## Context

Milestone M5 closes the loop between training and serving: a retrained model
must automatically pass a quality gate, be staged for canary deployment, and
optionally be promoted to Production after the canary succeeds.

M4 already built the canary delivery mechanism (Argo Rollouts + Istio traffic
split + Prometheus analysis), but the "v2" artifact was a placeholder — the
same v1 ONNX placed under Triton version `2`.

M5 must:

1. Actually retrain a model.
2. Register it in MLflow and compare its eval `auroc_macro` against the
current Production threshold.
3. If it passes, stage it as the candidate for canary deployment.
4. Build a real version-2 Triton model repository from that MLflow run.
5. Run the canary and, on success, promote it to stable / Production.

## Options considered

1. **Manual promotion.** After a retrain, a human exports the ONNX, copies it
to the canary PVC, applies `rollout-v2.yaml`, watches the canary, and manually
runs `kubectl argo rollouts promote`.

2. **Automated training gate + orchestration script.** The training run itself
validates the candidate against the production AUROC threshold and stages the
model version. A shell orchestrator then builds the canary artifact, starts
the Rollout, waits for Healthy/Degraded, and optionally completes the
promotion.

## Decision

Use **Option 2: automated training gate + orchestration script**.

### Training gate (`training/src/promotion.py`)

- `validate_candidate(run_id)` compares the run's `auroc_macro` to the
  production threshold. The threshold is configurable via
  `MLFLOW_PRODUCTION_AUROC_THRESHOLD`; if unset, the current Production
  version's `auroc_macro` is used.
- `stage_candidate(run_id)` validates and transitions the model version to
  `Staging`.
- `promote_to_production(run_id)` transitions the model version to
  `Production`.

`training/train.py` wires the gate behind `MLFLOW_PROMOTE_MODEL=true`:

- Registers the model under `distilbert-toxicity`.
- Resolves the production threshold.
- Fails fast with exit code 1 if the candidate does not beat the threshold.
- Calls `stage_candidate()` and tags the run with `milestone=M5` on success.

### Canary orchestrator (`serving/cpu/canary/promote-and-canary.sh`)

Takes an MLflow `run_id` and an optional `--promote` flag:

1. Runs the training gate (`python -m training.src.promotion validate`).
2. Builds the canary model repository with
   `serving/cpu/canary/build-canary-from-run.sh`, which places the ONNX under
   Triton version `2` and pins the config to serve only version `2`.
3. Applies the canary ServiceMonitor and `rollout-v2.yaml`.
4. Waits for the Rollout to reach `Healthy` or `Degraded`.
5. With `--promote`, completes the rollout (`kubectl argo rollouts promote`)
   and marks the MLflow version as `Production`.

### Rollback (`serving/cpu/rollback.sh`)

Reverts a bad canary by aborting any in-progress rollout and re-applying the
stable `rollout.yaml` (v1 model repository).

## Consequences

### Positive

- **Closed-loop promotion.** A passing retrain can move to canary and then to
  Production without handoffs or copy-paste.
- **Fail-fast quality gate.** Bad retrains stop at training time, before any
  serving artifact is built or traffic is shifted.
- **Reproducible handoff.** The canary artifact is built from the same MLflow
  run that passed the gate, not from an ad-hoc local directory.
- **Safe by default.** The orchestrator waits for the canary to report
  `Healthy` before promotion; `--promote` is opt-in so a human can still
  inspect the canary.

### Negative

- **Promotion is CPU-path only for now.** The GPU predictor is still a KServe
  InferenceService; the same automated gate could be reused, but the canary
  orchestrator is scoped to the CPU Argo Rollout.
- **Production promotion does not copy PVCs.** After `kubectl argo rollouts
  promote`, the new stable ReplicaSet still mounts the canary PVC
  (`triton-cpu-canary-model-repo`). The stable `rollout.yaml` continues to
  point at `triton-cpu-model-repo`, so a later `kubectl apply -f
  serving/cpu/rollout.yaml` would roll back to v1. This matches the M4/M5 demo
  semantics; long-term, the stable PVC should be updated to the new v2
  artifact as part of promotion.
- **GPU verification is deferred.** M5 was implemented and verified on the CPU
  cluster; the GPU cluster will run the same training gate and canary pattern
  once hardware is available.

## References

- `training/src/promotion.py`
- `training/src/tracking.py`
- `training/train.py`
- `training/.env.example`
- `serving/cpu/canary/build-canary-from-run.sh`
- `serving/cpu/canary/promote-and-canary.sh`
- `serving/cpu/rollback.sh`
- `serving/cpu/canary/rollout-v2.yaml`
- `serving/cpu/rollout.yaml`
- `docs/adr/0009-placeholder-v2-artifact-for-m4.md` — placeholder v4 approach
  superseded by this ADR.
