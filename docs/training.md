# Training results (issue #5)

## Setup

| | |
|---|---|
| Base model | `distilbert-base-cased` (66M params), token classification head, 13 BIO tags (`src/labels.py`) |
| Data | provided splits: train 3,547 docs, validation 443 docs (test not used) |
| Long documents | sliding window, 512 subwords, 128-subword overlap (`src/windows.py`): train 3,547 docs -> 4,043 windows, val 443 -> 499 |
| Label alignment | word's BIO label on its first subword (found via per-word offset 0, so windows starting mid-word are handled), `-100` elsewhere |
| Optimizer | AdamW, lr 5e-5, weight decay 0.01, linear schedule, 10% warmup (253 of 2,530 steps) |
| Batch / epochs | 8 / 5, fp16, seed 42 |
| Model selection | best validation entity-level micro F1 (`load_best_model_at_end`) |
| Hardware / time | RTX 2080 SUPER (8 GB), 202 s of training, torch 2.10 + transformers 5.23 |

**Evaluation is document-level.** Each word's logits are averaged over every window that contains it, then whole documents are scored with seqeval (an entity counts only on an exact span-and-type match; seqeval default conlleval-style chunking). Overlapping windows are therefore not double-counted and nothing past 512 subwords is dropped.

Reproduce: `python scripts/train.py --run-name full_baseline`. Full numbers, loss curve and per-epoch metrics: `docs/results/full_baseline.json`.

## Validation results (best checkpoint = epoch 5)

| type | precision | recall | F1 | support |
|---|---|---|---|---|
| NAME_STUDENT | 0.991 | 0.993 | **0.992** | 1048 |
| EMAIL | 0.951 | 0.992 | **0.971** | 372 |
| URL_PERSONAL | 0.984 | 0.938 | **0.960** | 64 |
| PHONE_NUM | 0.956 | 0.944 | **0.950** | 252 |
| STREET_ADDRESS | 0.934 | 0.960 | **0.947** | 351 |
| USERNAME | 0.944 | 0.882 | **0.912** | 76 |
| **micro avg** | 0.969 | 0.976 | **0.973** | 2163 |
| macro avg | 0.960 | 0.952 | 0.955 | 2163 |

For comparison, the issue #2 small run (200 train docs, truncated at 512) reached 0.78 F1 with 0.00 on URL and USERNAME.

## Per-epoch validation F1

| epoch | val loss | P | R | F1 | NAME | ADDRESS | PHONE | EMAIL | URL | USERNAME |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0055 | 0.927 | 0.960 | 0.943 | 0.982 | 0.876 | 0.909 | 0.939 | 0.921 | 0.877 |
| 2 | 0.0032 | 0.960 | 0.968 | 0.964 | 0.990 | 0.914 | 0.931 | 0.967 | 0.960 | 0.918 |
| 3 | 0.0033 | 0.957 | 0.971 | 0.964 | 0.992 | 0.905 | 0.943 | 0.970 | 0.952 | 0.912 |
| 4 | 0.0030 | 0.962 | 0.971 | 0.967 | 0.991 | 0.919 | 0.945 | 0.972 | 0.961 | 0.905 |
| 5 | 0.0030 | 0.969 | 0.976 | **0.973** | 0.992 | 0.947 | 0.950 | 0.971 | 0.960 | 0.912 |

Training loss: 1.61 (step 50) -> 0.001 (step 2,500). Validation loss flattens after epoch 2 and F1 is still creeping up at epoch 5, so there is no sign of overfitting yet; longer training is one thing to try in #8.

## Observations / caveats (for #7, #8 and the report)

- **Weakest categories:** USERNAME (0.91, lowest recall 0.88) and STREET_ADDRESS (0.95). USERNAME and URL have only 64-76 validation entities, so their scores move by ~1.5 points per entity.
- **Template leakage:** all 8 prompt templates (`prompt_id` 0-7) appear in train, validation *and* test in roughly the same proportions. These validation (and future test) scores measure in-template performance, not generalization to new kinds of text. See #1 / #4.
- **Unseen formats fail:** the training data has 9 phone formats (e.g. `(999) 999-9999`, `+99 99 999 9999`, `999-9999-9999`) but never the common US `999-999-9999`. On `tests/examples.txt`, 3 of the 5 phone numbers (`415-555-1234`, `408-555-4321`, `650-555-2468`) are tagged `STREET_ADDRESS` instead of `PHONE_NUM`. Every other example entity is found with the right type, and the no-PII sentence produces nothing. This is a concrete generalization-gap example for #7.
- **Trailing punctuation:** words keep attached punctuation (`john@example.com,`), so predicted spans include it. That matches the gold labels, so it does not affect scores, but the demo output (#6) may want to strip it.
- **Inference consistency:** `scripts/predict.py` takes the label from the first window a word appears in; training evaluation averages logits across windows. `src/windows.aggregate_word_logits` can be reused in `predict.py` so the demo matches the reported numbers.

## Checkpoint

Saved locally to `models/full_baseline/` (250 MB, gitignored). Loads with `AutoModelForTokenClassification.from_pretrained("models/full_baseline")` and works with `scripts/predict.py --model models/full_baseline`.
