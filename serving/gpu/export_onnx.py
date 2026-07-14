"""Export a DistilBERT toxicity classifier to ONNX with a dynamic batch axis.

Bridge between the trained PyTorch checkpoint (in MLflow) and the TensorRT
plan Triton serves. We leave the batch dimension dynamic so TRT can build an
optimization profile over a batch range (see build-engine.sh).

Usage:
    python export_onnx.py \\
        --model-uri ./local-checkpoint-dir \\
        --seq-len 128 \\
        --out ./model.onnx

For an MLflow-registered model, resolve the URI to a local directory first:
    mlflow artifacts download \\
        --artifact-uri models:/distilbert-toxicity/Production \\
        --dst-path ./local-checkpoint-dir
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-uri", required=True,
                   help="Local path or HF hub id of the trained model")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--num-labels", type=int, default=6,
                   help="Jigsaw multi-label output dim")
    p.add_argument("--out", required=True, help="Output .onnx path")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_uri)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_uri,
        num_labels=args.num_labels,
        torchscript=True,
    )
    model.eval()

    dummy = tokenizer(
        "this is a dummy input for tracing",
        padding="max_length",
        truncation=True,
        max_length=args.seq_len,
        return_tensors="pt",
    )

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy["input_ids"], dummy["attention_mask"]),
            str(out_path),
            opset_version=17,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids":      {0: "batch"},
                "attention_mask": {0: "batch"},
                "logits":         {0: "batch"},
            },
        )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
