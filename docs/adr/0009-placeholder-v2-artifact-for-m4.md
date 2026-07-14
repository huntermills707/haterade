# 9. Use a placeholder v2 artifact for M4 canary

Date: 2026-07-14
Status: Accepted

## Context

Milestone M4 is about canary delivery mechanics, not about model
retraining. To demonstrate an Argo Rollouts canary we need two distinct
serving artifacts (stable v1 and candidate v2), but the genuinely
retrained model is the scope of M5.

Options considered:

1. **Placeholder v2.** Copy the verified v1 ONNX into a new PVC, place it
   under Triton version `2`, and pin the canary config to serve only that
   version. The model weights are identical to v1; only the Triton
   version label and PVC differ.

2. **Wait for M5 retrain.** Don't start M4 until a real v2 model exists.

## Decision

Use a **placeholder v2** for M4.

The placeholder is built by `serving/cpu/canary/build-canary-placeholder.sh`:

- Copies the v1 model repository from PVC `triton-cpu-model-repo` into a
  new PVC `triton-cpu-canary-model-repo`.
- Copies the version-1 ONNX into a version-2 directory:
  `distilbert-toxicity/2/model.onnx`.
- Adds `version_policy: { specific: { versions: [2] } }` to the canary
  `config.pbtxt` so the canary predictor serves only version 2.

The inference contract (input names, shapes, output names) is unchanged.
The placeholder is sufficient to exercise the Argo Rollouts canary,
including version-filtered Prometheus analysis.

## Consequences

### Positive

- **M4 is decoupled from M5.** Canary plumbing can be built and verified
  now, without waiting for a retrain.
- **No regression risk.** The placeholder v2 has the same model weights
  as the verified v1, so analysis should pass barring infra issues.
- **Observable difference.** Triton metrics carry a `version="2"` label,
  so the AnalysisTemplate can measure canary-specific success rate and
  latency.

### Negative

- **Placeholder is not a real model upgrade.** The canary demo does not
  prove that a new model version is safe; it only proves that the
  delivery mechanism works.
- **M5 must replace the placeholder.** When the real retrained model is
  ready, the canary PVC contents must be regenerated from the new
  MLflow run.

## Replacement path (M5)

When M5 produces a new MLflow run:

1. Run `serving/cpu/build-model-repo.sh` with the new `MLFLOW_RUN_ID`.
2. Copy the resulting model repo into PVC
   `triton-cpu-canary-model-repo` instead of running the placeholder
   script.
3. Apply `serving/cpu/canary/rollout-v2.yaml` to run the canary again.

## References

- `serving/cpu/canary/build-canary-placeholder.sh`
- `serving/cpu/canary/model-pvc.yaml`
- `serving/cpu/canary/rollout-v2.yaml`
- `serving/cpu/analysis-template.yaml`
- `serving/cpu/build-model-repo.sh`
- `docs/adr/0008-argo-rollouts-canary-with-kserve-rawdeployment.md`
