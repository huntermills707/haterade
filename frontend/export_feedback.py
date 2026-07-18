#!/usr/bin/env python3
"""Export logged user feedback into Jigsaw-schema train/test CSVs.

Reads predictions.jsonl (id -> text) and feedback.jsonl (id -> user labels)
written by app.py, joins them on id, shuffles with a fixed seed, and splits
into feedback_train.csv / feedback_test.csv. Point training at the output
directory with FEEDBACK_CSV_DIR to fold this data into the next run.

The data lives on the toxicity-ui pod's PVC; copy it out first, e.g.:
    kubectl cp default/$(kubectl get pod -l app=toxicity-ui -o name | cut -d/ -f2):/data ./data
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

# Label contract — must match training/src/env.py:label_columns exactly.
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
COLUMNS = ["id", "comment_text", *LABELS]


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default="data",
                    help="dir containing predictions.jsonl + feedback.jsonl")
    ap.add_argument("--out-dir", default=None,
                    help="where to write the CSVs (default: same as --data-dir)")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir

    preds = {r["id"]: r["text"] for r in load_jsonl(data_dir / "predictions.jsonl")}
    # Last feedback wins if a user resubmits for the same id.
    fb = {r["id"]: r["labels"] for r in load_jsonl(data_dir / "feedback.jsonl")}

    rows, skipped = [], 0
    for fid, labels in fb.items():
        text = preds.get(fid)
        if text is None:
            skipped += 1
            continue
        rows.append({
            "id": fid,
            "comment_text": text,
            **{label: int(bool(labels.get(label, False))) for label in LABELS},
        })

    if not rows:
        print("no feedback rows to export (feedback.jsonl missing or empty)",
              file=sys.stderr)
        return 1

    random.Random(args.seed).shuffle(rows)
    n_test = max(1, round(len(rows) * args.test_frac)) if len(rows) > 1 else 0
    test_rows, train_rows = rows[:n_test], rows[n_test:]

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("feedback_train.csv", train_rows),
                        ("feedback_test.csv", test_rows)):
        with (out_dir / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(split)

    print(f"exported {len(rows)} rows ({skipped} skipped, no matching prediction): "
          f"{len(train_rows)} train / {len(test_rows)} test -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
