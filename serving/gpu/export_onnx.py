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
    p.add_argument("--tokenizer-uri", default=None,
                   help="Optional separate tokenizer path (defaults to model-uri)")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--num-labels", type=int, default=6,
                   help="Jigsaw multi-label output dim")
    p.add_argument("--int32-inputs", action="store_true",
                   help="Export ONNX inputs as INT32 (required for TensorRT). "
                        "Default is INT64 (for CPU ONNX Runtime).")
    p.add_argument("--out", required=True, help="Output .onnx path")
    return p.parse_args()


class Int32InputWrapper(torch.nn.Module):
    """Wrap a transformer model so ONNX inputs are int32.

    TensorRT (at least through Triton 23.05 / TensorRT 8.6) does not support
    int64 inputs. PyTorch embedding layers need int64 indices, so we accept
    int32 from Triton and cast to int64 right before the original model.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.model(
            input_ids.to(torch.long),
            attention_mask=attention_mask.to(torch.long),
        )


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer_uri = args.tokenizer_uri or args.model_uri
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_uri)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_uri,
        num_labels=args.num_labels,
        torchscript=True,
    )
    base_model.eval()

    tokenizer_out = tokenizer(
        "this is a dummy input for tracing",
        padding="max_length",
        truncation=True,
        max_length=args.seq_len,
        return_tensors="pt",
    )

    if args.int32_inputs:
        # TensorRT does not support INT64 inputs; wrap model to accept INT32
        # and cast internally to the Long indices PyTorch embeddings require.
        model = Int32InputWrapper(base_model)
        dummy_input_ids = tokenizer_out["input_ids"].to(torch.int32)
        dummy_attention_mask = tokenizer_out["attention_mask"].to(torch.int32)
    else:
        # CPU ONNX Runtime path uses native INT64 inputs.
        model = base_model
        dummy_input_ids = tokenizer_out["input_ids"]
        dummy_attention_mask = tokenizer_out["attention_mask"]

    model.eval()

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask),
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
