"""O-threshold sweep for a trained checkpoint (issue #8).

Subtracts an offset from the "O" logit before argmax, which makes the model
more willing to predict PII (higher recall, lower precision). To avoid tuning
and reporting on the same documents, validation is split into two halves
(seeded): the offset is chosen on one half and scored on the other, and vice
versa (2-fold). The test split is not used.

Usage:
    python scripts/tune_threshold.py --model models/full_baseline
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from seqeval.metrics import classification_report
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from src.labels import ENTITY_TYPES, id2label, label2id  # noqa: E402
from src.windows import aggregate_word_logits  # noqa: E402
from train import encode_docs, load_split  # noqa: E402

OFFSETS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]


def window_logits(model, tokenizer, features, batch_size=16):
    collate = DataCollatorForTokenClassification(tokenizer)
    device = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch = collate([{k: v for k, v in f.items() if k != "labels"} for f in features[i : i + batch_size]])
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits.float().cpu().numpy()
            for row, f in zip(logits, features[i : i + batch_size]):
                out.append(row[: len(f["input_ids"])])
    return out


def score(word_logits, docs, idx, offset):
    o = label2id["O"]
    true_seqs, pred_seqs = [], []
    for d in idx:
        lg = word_logits[d].copy()
        lg[:, o] -= offset
        true_seqs.append(list(docs[d]["labels"]))
        pred_seqs.append([id2label[int(i)] for i in lg.argmax(-1)])
    rep = classification_report(true_seqs, pred_seqs, output_dict=True, zero_division=0)
    res = {k: rep["micro avg"][m] for k, m in (("precision", "precision"), ("recall", "recall"), ("f1", "f1-score"))}
    res.update({f"f1_{t}": rep.get(t, {}).get("f1-score", 0.0) for t in ENTITY_TYPES})
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "raw"))
    ap.add_argument("--seed", type=int, default=0, help="seed for the validation half split")
    ap.add_argument("--out", default=None, help="JSON output (default: docs/results/threshold_<model>.json)")
    args = ap.parse_args()

    model_dir = Path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    info = json.loads((model_dir / "train_info.json").read_text())

    docs = load_split(Path(args.data_dir) / "validation.csv", 0, 0)
    feats, metas, doc_index = encode_docs(docs, tokenizer, info.get("max_length", 512), info.get("stride", 128))
    wl = window_logits(model, tokenizer, feats)
    per_doc = {}
    for w, d in enumerate(doc_index):
        per_doc.setdefault(d, []).append(w)
    word_logits = []
    for d, doc in enumerate(docs):
        agg, missing = aggregate_word_logits(len(doc["tokens"]), [wl[w] for w in per_doc[d]], [metas[w] for w in per_doc[d]])
        assert not missing
        word_logits.append(agg)

    perm = np.random.default_rng(args.seed).permutation(len(docs))
    halves = [perm[: len(docs) // 2], perm[len(docs) // 2 :]]

    sweep = {str(o): score(word_logits, docs, range(len(docs)), o) for o in OFFSETS}
    folds = []
    for tune, held in ((0, 1), (1, 0)):
        best = max(OFFSETS, key=lambda o: score(word_logits, docs, halves[tune], o)["f1"])
        folds.append(
            {
                "chosen_offset": best,
                "heldout_f1_offset0": score(word_logits, docs, halves[held], 0.0)["f1"],
                "heldout_f1_chosen": score(word_logits, docs, halves[held], best)["f1"],
            }
        )

    print(f"{'offset':>6} {'P':>6} {'R':>6} {'F1':>6}  (full val, for reference only)")
    for o, r in sweep.items():
        print(f"{o:>6} {r['precision']:.4f} {r['recall']:.4f} {r['f1']:.4f}")
    for i, f in enumerate(folds):
        print(f"fold {i}: chosen offset {f['chosen_offset']}, held-out F1 {f['heldout_f1_offset0']:.4f} -> {f['heldout_f1_chosen']:.4f}")

    out = Path(args.out) if args.out else REPO_ROOT / "docs" / "results" / f"threshold_{model_dir.name}.json"
    out.write_text(json.dumps({"model": model_dir.name, "offsets": OFFSETS, "full_val_sweep": sweep, "two_fold": folds}, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
