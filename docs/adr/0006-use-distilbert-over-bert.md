# 6. Use DistilBERT over full BERT for the Jigsaw toxicity model

Date: 2026-07-13
Status: Accepted

## Context

M1 trains a text classifier on the Jigsaw Toxic Comment Classification
Challenge (2018) — 160k English comments labeled across six non-mutually
exclusive toxicity categories (`toxic`, `severe_toxic`, `obscene`,
`threat`, `insult`, `identity_hate`). Multi-label, sigmoid output,
BCEWithLogitsLoss.

The choice of backbone had to satisfy three hard constraints:

1. **CPU-only training on the laptop, with short runs.** The GPU cluster
   is pending hardware (`README.md` M0 status). Per-step latency on a
   laptop CPU at batch 16, seq 128, fp32 is the budget — measured at
   ~1.8 s/step on this host (CachyOS laptop, no AVX-512). The first
   end-to-end run finished 2k rows / 1 epoch in **236 s** train + 6 s
   eval (MLflow run `18c785f7036143869547d97fc2476c40`).
2. **The same artifact must flow into the GPU path (M2 Triton +
   TensorRT).** A backbone that doesn't trace cleanly to ONNX→TRT would
   block the headline CPU-vs-GPU cost story.
3. **Cold-start budget for the portfolio demo.** This is not a Kaggle
   entry; it's a platform showcase. State-of-the-art accuracy is not the
   deliverable. A model that reaches AUROC ≈ 0.97 on a small holdout in
   one epoch is more than enough to drive KServe/KEDA/Argo Rollouts
   demos.

## Decision

Use **`distilbert-base-uncased`** as the backbone for M1.

- 6-layer, 66M parameters (≈ 60% the size of `bert-base-uncased`).
- HuggingFace `transformers` `AutoModelForSequenceClassification` with
  `problem_type="multi_label_classification"` and `num_labels=6`.
- Pretrained weights from the HF Hub. Classifier head trained from
  scratch on Jigsaw.
- Max sequence length: 128 (matches the Triton `SEQ_LEN` plan baked in
  `serving/gpu/`).
- fp32 on CPU; fp16 plan on Turing (sm_75) for the GPU path.

### First measured result

| metric             | value   |
|--------------------|---------|
| `auroc_macro`      | 0.9795  |
| `auroc_toxic`      | 0.9730  |
| `auroc_severe_toxic` | 0.9849 |
| `auroc_obscene`    | 0.9811  |
| `auroc_threat`     | 0.9849  |
| `auroc_insult`     | 0.9800  |
| `auroc_identity_hate` | 0.9729 |
| train_seconds      | 236     |
| eval_seconds       | 6.1     |

2000 train rows / 200 eval rows / 1 epoch / batch 16 / lr 2e-5.
Eval sample is small and Jigsaw is heavily imbalanced toward the
negative class — the macro AUROC is generous, not robust. It is more
than enough for the platform-level demos that consume this artifact
(KServe RawDeployment, KEDA scaling, Argo Rollouts canary).

## Consequences

### Positive

- **Fits the CPU short-run budget.** 1 epoch ≈ 4 min on the laptop.
  Full BERT-base at the same batch/seq would be ~3× slower (12-layer
  encoder vs 6).
- **Artifacts are tractable.** `model.pth` is 256 MiB in the MLflow
  run; full BERT would be ~420 MiB. Matters for the Triton PVC copy
  step and for `kubectl cp` speeds.
- **ONNX/TRT export is well-trodden.** `optimum.exporters` and TensorRT
  both have first-class DistilBERT paths. No custom op handling needed.
- **Per-class AUROC > 0.97 on a smoke holdout** is more than enough
  signal for the platform demos (the model exists to be *served*, not
  to win Kaggle).

### Negative

- **Not state of the art on Jigsaw.** Ensembled BERT-large + auxiliary
  data hit ~0.99 on the public LB; DistilBERT tops out lower (~0.98
  with tuning). Acceptable for the project framing.
- **6 layers → smaller margin for pruning/quantization tricks later.**
  If a stretch goal is INT8 distillation for edge-style serving, the
  teacher needs to be bigger than DistilBERT. Out of scope for v1.
- **Uncased vocabulary.** Loses signal on toxicity markers that ride on
  casing (e.g., "F*&K" vs "fuck"). Cased DistilBERT exists; not used
  here because the uncased vocab is what every downstream tool
  (TRT engine plans, M5 retrain) expects. Worth revisiting if M5
  retrain surfaces casing-driven false negatives.

## Alternatives considered

### `bert-base-uncased` (full BERT)

12-layer, 110M params. ~3× CPU train time per step at the same batch
and seq length. The accuracy delta on Jigsaw is real (~+0.005 macro
AUROC) but irrelevant to the platform narrative. Rejected on the
short-run budget.

### `roberta-base`

Stronger pretraining recipe, generally beats BERT on downstream tasks.
But RoBERTa has no TokenTypeEmbeddings and its tokenizer differs from
the BERT family — would force a separate export path for the GPU TRT
engine and complicate the "same model, two runtimes" framing.
Rejected for symmetry cost.

### `distilbert-base-multilingual-cased`

Considered for a brief moment because the Jigsaw dataset has
non-English comments in the wild. Rejected: the official competition
train set is English-only, and the multilingual variant is 1.7 GB on
the Hub (vs 268 MB for the English base), which would dominate the
artifact size.

### TinyBERT / MobileBERT / quantized variants

Real options for an edge-deployment story. Rejected here because the
project is explicitly *not* an edge deployment — the GPU cluster has
real RTX 2060 Super hardware and the CPU cluster is a beefy laptop.
The bottleneck for v1 is platform wiring, not inference cost.

## References

- DistilBERT paper: Sanh et al., 2019. *DistilBERT, a distilled version
  of BERT: smaller, faster, cheaper and lighter.*
  https://arxiv.org/abs/1910.01108
- Jigsaw Toxic Comment Classification Challenge (Kaggle, 2018):
  https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge
- M1 implementation: `training/` (entrypoint
  `python -m training.train`, smoke-tested 2026-07-13).
- Verified run: MLflow run_id `18c785f7036143869547d97fc2476c40`,
  experiment `toxicity-distilbert` on the CPU cluster.
