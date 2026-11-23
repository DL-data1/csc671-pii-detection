"""Full DistilBERT PII training run with sliding windows (issue #5).

Trains distilbert-base-cased on the training split, evaluates on validation
after every epoch with document-level, entity-level F1 (seqeval), and keeps the
checkpoint with the best validation F1.

Long documents are split into overlapping windows (src/windows.py: 512
subwords, stride 128). For evaluation, each word's logits are averaged over the
windows it appears in, so every word of every document is scored exactly once.
That makes the validation numbers document-level, not per-window.

The test split is never touched here; model selection uses validation only.

Outputs:
  <output>/                     best model + tokenizer (from_pretrained-ready),
                                train_info.json, val_report.txt
  <results-dir>/<run-name>.json small tracked summary for the experiment table

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 5 --lr 5e-5 --run-name full_lr5e-5
    python scripts/train.py --train-docs 200 --val-docs 50 --run-name smoke   # quick check
"""

import argparse
import ast
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from seqeval.metrics import classification_report
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.labels import ENTITY_TYPES, LABELS, id2label, label2id  # noqa: E402
from src.windows import MAX_LENGTH, STRIDE, aggregate_word_logits, encode_windows  # noqa: E402

DEFAULT_MODEL = "distilbert-base-cased"


def load_split(path, n_docs, seed, exclude_prompts=(), only_prompts=None):
    """Read a raw CSV and return (tokens, labels) docs; n_docs=0 keeps all.

    exclude_prompts / only_prompts filter rows by prompt_id (template).
    """
    df = pd.read_csv(path, usecols=["document", "tokens", "labels", "prompt_id"])
    if exclude_prompts:
        df = df[~df["prompt_id"].isin(exclude_prompts)]
    if only_prompts is not None:
        df = df[df["prompt_id"].isin(only_prompts)]
    if n_docs and n_docs < len(df):
        df = df.sample(n=n_docs, random_state=seed)
    docs = []
    for _, row in df.iterrows():
        tokens = ast.literal_eval(row["tokens"])
        labels = ast.literal_eval(row["labels"])
        assert len(tokens) == len(labels), row["document"]
        docs.append({"document": row["document"], "tokens": tokens, "labels": labels})
    return docs


def encode_docs(docs, tokenizer, max_length=MAX_LENGTH, stride=STRIDE):
    """Window-encode every doc. Returns features, per-window metas, per-window doc index."""
    features, metas, doc_index = [], [], []
    for i, doc in enumerate(docs):
        f, m = encode_windows(tokenizer, doc["tokens"], doc["labels"], label2id, max_length, stride)
        features.extend(f)
        metas.extend(m)
        doc_index.extend([i] * len(f))
    return features, metas, doc_index


def doc_level_sequences(logits, docs, metas, doc_index):
    """Turn per-window logits into per-document (gold, predicted) BIO sequences."""
    by_doc = {}
    for w, d in enumerate(doc_index):
        by_doc.setdefault(d, []).append(w)
    true_seqs, pred_seqs = [], []
    for d, doc in enumerate(docs):
        windows = by_doc[d]
        word_logits, missing = aggregate_word_logits(
            len(doc["tokens"]), [logits[w] for w in windows], [metas[w] for w in windows]
        )
        assert not missing, f"doc {doc['document']}: words without subwords {missing[:5]}"
        true_seqs.append(list(doc["labels"]))
        pred_seqs.append([id2label[int(i)] for i in word_logits.argmax(-1)])
    return true_seqs, pred_seqs


def class_weights(features, scheme, cap):
    """Per-label loss weights from first-subword label counts in the training windows.

    "sqrt": w = sqrt(count_O / count_label), capped at `cap`, O = 1. Labels that
    never occur in training (I-EMAIL, I-URL_PERSONAL, I-USERNAME) get 1.
    """
    if scheme == "none":
        return None
    counts = np.zeros(len(LABELS))
    for f in features:
        for lab in f["labels"]:
            if lab != -100:
                counts[lab] += 1
    w = np.ones(len(LABELS))
    seen = counts > 0
    w[seen] = np.minimum(np.sqrt(counts[label2id["O"]] / counts[seen]), cap)
    return w


class WeightedTrainer(Trainer):
    """Trainer with class-weighted token cross-entropy (ignore_index -100)."""

    def __init__(self, *a, label_weights=None, **kw):
        super().__init__(*a, **kw)
        self.label_weights = torch.tensor(label_weights, dtype=torch.float32)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(),
            labels.view(-1),
            weight=self.label_weights.to(logits.device),
            ignore_index=-100,
        )
        return (loss, outputs) if return_outputs else loss


def make_compute_metrics(docs, metas, doc_index):
    """compute_metrics closure: document-level entity P/R/F1, overall and per type.

    Relies on Trainer returning eval predictions in dataset order (it does:
    sequential sampler; do not enable group_by_length).
    """

    def compute_metrics(eval_pred):
        logits = eval_pred.predictions
        assert len(logits) == len(metas), (len(logits), len(metas))
        true_seqs, pred_seqs = doc_level_sequences(logits, docs, metas, doc_index)
        rep = classification_report(true_seqs, pred_seqs, output_dict=True, zero_division=0)
        out = {
            "precision": rep["micro avg"]["precision"],
            "recall": rep["micro avg"]["recall"],
            "f1": rep["micro avg"]["f1-score"],
            "macro_f1": rep["macro avg"]["f1-score"],
        }
        for t in ENTITY_TYPES:
            out[f"f1_{t}"] = rep.get(t, {}).get("f1-score", 0.0)
        return out

    return compute_metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "raw"))
    ap.add_argument("--train-docs", type=int, default=0, help="0 = all")
    ap.add_argument("--val-docs", type=int, default=0, help="0 = all")
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.1, help="fraction of total steps")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-name", default="full_baseline")
    ap.add_argument("--output", default=None, help="default: models/<run-name>")
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "docs" / "results"))
    ap.add_argument("--model-name", default=DEFAULT_MODEL, help="e.g. distilbert-base-uncased")
    ap.add_argument("--class-weights", choices=["none", "sqrt"], default="none")
    ap.add_argument("--weight-cap", type=float, default=10.0, help="max class weight relative to O")
    ap.add_argument("--max-length", type=int, default=MAX_LENGTH)
    ap.add_argument("--stride", type=int, default=STRIDE, help="overlap between windows, in subwords")
    ap.add_argument(
        "--holdout-prompt-ids", type=int, nargs="*", default=[],
        help="generalization check: drop these prompt templates from train AND val, then score the best "
        "checkpoint once on their train+val documents (never test). Not used for model selection.",
    )
    ap.add_argument("--keep-checkpoints", action="store_true", help="keep per-epoch checkpoints (large)")
    args = ap.parse_args()

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output) if args.output else REPO_ROOT / "models" / args.run_name
    results_dir = Path(args.results_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    hold = args.holdout_prompt_ids
    train_docs = load_split(data_dir / "train.csv", args.train_docs, args.seed, exclude_prompts=hold)
    val_docs = load_split(data_dir / "validation.csv", args.val_docs, args.seed, exclude_prompts=hold)
    holdout_docs = []
    if hold:
        for split in ("train.csv", "validation.csv"):
            holdout_docs += load_split(data_dir / split, 0, args.seed, only_prompts=hold)
        print(f"holdout prompt_ids {hold}: {len(holdout_docs)} docs (train+val), excluded from training")
    train_ds, _, _ = encode_docs(train_docs, tokenizer, args.max_length, args.stride)
    val_ds, val_metas, val_doc_index = encode_docs(val_docs, tokenizer, args.max_length, args.stride)
    print(f"train: {len(train_docs)} docs -> {len(train_ds)} windows (max_length {args.max_length}, stride {args.stride})")
    label_weights = class_weights(train_ds, args.class_weights, args.weight_cap)
    if label_weights is not None:
        print("class weights:", {LABELS[i]: round(float(x), 2) for i, x in enumerate(label_weights)})
    print(f"val:   {len(val_docs)} docs -> {len(val_ds)} windows")

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, num_labels=len(LABELS), id2label=id2label, label2id=label2id
    )

    steps_per_epoch = -(-len(train_ds) // args.batch_size)
    total_steps = int(steps_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=50,
        fp16=torch.cuda.is_available(),  # RTX 20xx: fp16 yes, bf16 no
        report_to="none",
        dataloader_num_workers=0,
        seed=args.seed,
    )

    trainer_cls = Trainer
    extra = {}
    if label_weights is not None:
        trainer_cls, extra = WeightedTrainer, {"label_weights": label_weights}
    trainer = trainer_cls(
        **extra,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        processing_class=tokenizer,
        compute_metrics=make_compute_metrics(val_docs, val_metas, val_doc_index),
    )

    start = time.time()
    trainer.train()
    runtime = time.time() - start

    holdout = None
    if holdout_docs:
        h_ds, h_metas, h_index = encode_docs(holdout_docs, tokenizer, args.max_length, args.stride)
        val_metrics_fn, trainer.compute_metrics = trainer.compute_metrics, None  # metrics fn is tied to val docs
        h_out = trainer.predict(h_ds)
        trainer.compute_metrics = val_metrics_fn
        h_true, h_pred = doc_level_sequences(h_out.predictions, holdout_docs, h_metas, h_index)
        h_rep = classification_report(h_true, h_pred, output_dict=True, zero_division=0)
        print("HOLDOUT (unseen prompt templates):")
        print(classification_report(h_true, h_pred, digits=4, zero_division=0))
        holdout = {
            "prompt_ids": hold,
            "docs": len(holdout_docs),
            "precision": round(float(h_rep["micro avg"]["precision"]), 4),
            "recall": round(float(h_rep["micro avg"]["recall"]), 4),
            "f1": round(float(h_rep["micro avg"]["f1-score"]), 4),
            "macro_f1": round(float(h_rep["macro avg"]["f1-score"]), 4),
            "per_type": {
                t: {k: round(float(h_rep[t][k]), 4) for k in ("precision", "recall", "f1-score", "support")}
                for t in ENTITY_TYPES if t in h_rep
            },
        }

    # Best checkpoint is loaded at this point; score it once more for the report.
    pred_out = trainer.predict(val_ds)
    true_seqs, pred_seqs = doc_level_sequences(pred_out.predictions, val_docs, val_metas, val_doc_index)
    report = classification_report(true_seqs, pred_seqs, digits=4, zero_division=0)
    report_dict = classification_report(true_seqs, pred_seqs, output_dict=True, zero_division=0)
    print(report)

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    history = trainer.state.log_history
    epochs = [
        {k: v for k, v in e.items() if k == "epoch" or k.startswith("eval_")}
        for e in history
        if "eval_f1" in e
    ]
    loss_log = [{k: e[k] for k in ("step", "epoch", "loss", "learning_rate") if k in e} for e in history if "loss" in e]
    per_type = {
        t: {k: round(float(report_dict[t][k]), 4) for k in ("precision", "recall", "f1-score", "support")}
        for t in ENTITY_TYPES
        if t in report_dict
    }
    best_val = {
        "precision": round(float(report_dict["micro avg"]["precision"]), 4),
        "recall": round(float(report_dict["micro avg"]["recall"]), 4),
        "f1": round(float(report_dict["micro avg"]["f1-score"]), 4),
        "macro_f1": round(float(report_dict["macro avg"]["f1-score"]), 4),
    }
    info = {
        "run_name": args.run_name,
        "base_model": args.model_name,
        "args": {k: v for k, v in vars(args).items() if k not in ("data_dir", "output", "results_dir")},
        "max_length": args.max_length,
        "stride": args.stride,
        "class_weight_vector": None if label_weights is None else [round(float(x), 4) for x in label_weights],
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "train_docs": len(train_docs),
        "train_windows": len(train_ds),
        "val_docs": len(val_docs),
        "val_windows": len(val_ds),
        "best_checkpoint": Path(trainer.state.best_model_checkpoint).name,
        "best_epoch": max(epochs, key=lambda e: e["eval_f1"])["epoch"] if epochs else None,
        "best_val": best_val,
        "best_val_per_type": per_type,
        "holdout": holdout,
        "per_epoch_val": epochs,
        "runtime_sec": round(runtime, 1),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "train_loss_log": loss_log,
    }
    (out_dir / "train_info.json").write_text(json.dumps(info, indent=2))
    (out_dir / "val_report.txt").write_text(report)

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{args.run_name}.json").write_text(json.dumps(info, indent=2))
    (results_dir / f"{args.run_name}_val_report.txt").write_text(report)

    if not args.keep_checkpoints:
        shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)
    print(f"Best val F1 {best_val['f1']:.4f} ({info['best_checkpoint']})")
    print(f"Saved model + tokenizer to {out_dir}")


if __name__ == "__main__":
    main()
