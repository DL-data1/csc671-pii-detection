"""Small end-to-end DistilBERT token-classification run (issue #2).

Trains distilbert-base-cased on a seeded random subset of the raw CSVs and
saves model + tokenizer to a directory that any inference script can load
with `AutoModelForTokenClassification.from_pretrained(<dir>)`.

This is the quick version: long documents are simply truncated at 512
subwords. The final preprocessing (sliding window, final splits, evaluation
functions) is James's work in issue #4.

Usage:
    python scripts/train_small.py
    python scripts/train_small.py --train-docs 200 --val-docs 50 --epochs 4 --output models/distilbert-small

Input note: the dataset's `tokens` are whitespace-split words with punctuation
attached (e.g. "Feng,"). At inference, split raw text on whitespace the same
way before tokenizing with `is_split_into_words=True`.
"""

import argparse
import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
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
from src.labels import LABELS, id2label, label2id  # noqa: E402

MODEL_NAME = "distilbert-base-cased"
MAX_LENGTH = 512


def load_split(path, n_docs, seed):
    """Read a raw CSV and return a seeded random sample of (tokens, labels) docs."""
    df = pd.read_csv(path, usecols=["document", "tokens", "labels"])
    if n_docs and n_docs < len(df):
        df = df.sample(n=n_docs, random_state=seed)
    docs = []
    for _, row in df.iterrows():
        tokens = ast.literal_eval(row["tokens"])
        labels = ast.literal_eval(row["labels"])
        assert len(tokens) == len(labels), row["document"]
        docs.append({"document": row["document"], "tokens": tokens, "labels": labels})
    return docs


def encode(docs, tokenizer):
    """Tokenize word lists and align word-level BIO labels to subwords.

    The first subword of each word gets the word's label; later subwords and
    special tokens get -100 so the loss and metrics ignore them.
    Returns the encoded features and the number of truncated documents.
    """
    features, truncated = [], 0
    for doc in docs:
        enc = tokenizer(doc["tokens"], is_split_into_words=True, truncation=True, max_length=MAX_LENGTH)
        word_ids = enc.word_ids()
        label_ids, prev = [], None
        for wid in word_ids:
            if wid is None or wid == prev:
                label_ids.append(-100)
            else:
                label_ids.append(label2id[doc["labels"][wid]])
            prev = wid
        last_word = max(w for w in word_ids if w is not None)
        if last_word < len(doc["tokens"]) - 1:
            truncated += 1
        features.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": label_ids})
    return features, truncated


def compute_metrics(eval_pred):
    """Entity-level P/R/F1 (seqeval) over first-subword predictions."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    true_seqs, pred_seqs = [], []
    for p_row, l_row in zip(preds, labels):
        mask = l_row != -100
        true_seqs.append([id2label[int(i)] for i in l_row[mask]])
        pred_seqs.append([id2label[int(i)] for i in p_row[mask]])
    return {
        "precision": precision_score(true_seqs, pred_seqs, zero_division=0),
        "recall": recall_score(true_seqs, pred_seqs, zero_division=0),
        "f1": f1_score(true_seqs, pred_seqs, zero_division=0),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "raw"))
    ap.add_argument("--train-docs", type=int, default=200, help="0 = all")
    ap.add_argument("--val-docs", type=int, default=50, help="0 = all")
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=str(REPO_ROOT / "models" / "distilbert-small"))
    args = ap.parse_args()

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_docs = load_split(data_dir / "train.csv", args.train_docs, args.seed)
    val_docs = load_split(data_dir / "validation.csv", args.val_docs, args.seed)
    train_ds, train_trunc = encode(train_docs, tokenizer)
    val_ds, val_trunc = encode(val_docs, tokenizer)
    print(f"train docs: {len(train_ds)} ({train_trunc} truncated at {MAX_LENGTH} subwords)")
    print(f"val docs:   {len(val_ds)} ({val_trunc} truncated at {MAX_LENGTH} subwords)")

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=str(out_dir / "trainer_tmp"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        warmup_steps=10,
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=5,
        save_strategy="no",
        fp16=torch.cuda.is_available(),  # RTX 20xx: fp16 yes, bf16 no
        report_to="none",
        dataloader_num_workers=0,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    start = time.time()
    trainer.train()
    runtime = time.time() - start
    metrics = trainer.evaluate()

    # Per-category report on the val subset.
    pred_out = trainer.predict(val_ds)
    preds = np.argmax(pred_out.predictions, axis=-1)
    true_seqs, pred_seqs = [], []
    for p_row, l_row in zip(preds, pred_out.label_ids):
        mask = l_row != -100
        true_seqs.append([id2label[int(i)] for i in l_row[mask]])
        pred_seqs.append([id2label[int(i)] for i in p_row[mask]])
    report = classification_report(true_seqs, pred_seqs, zero_division=0)
    print(report)

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    loss_log = [
        {k: e[k] for k in ("step", "epoch", "loss") if k in e} for e in trainer.state.log_history if "loss" in e
    ]
    info = {
        "base_model": MODEL_NAME,
        "args": vars(args),
        "max_length": MAX_LENGTH,
        "train_docs": len(train_ds),
        "val_docs": len(val_ds),
        "train_truncated": train_trunc,
        "val_truncated": val_trunc,
        "runtime_sec": round(runtime, 1),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "val_metrics": metrics,
        "train_loss_log": loss_log,
    }
    (out_dir / "train_info.json").write_text(json.dumps(info, indent=2))
    (out_dir / "val_report.txt").write_text(report)
    print(f"Saved model + tokenizer to {out_dir}")


if __name__ == "__main__":
    main()
