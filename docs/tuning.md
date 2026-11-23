# Tuning and final model (issue #8)

**Final model: `distilbert-base-uncased`, otherwise the #5 baseline recipe** (lr 5e-5, 5 epochs, batch 8, 10% warmup, 512/128 sliding windows, no class weights, seed 42). Best epoch 2. Validation entity-level F1 **0.985** (cased baseline: 0.973). Saved as `models/uncased/`.

All decisions below use the **validation** split only; the test split was never loaded. The full table, with every run and per-category F1, is in [`experiments.md`](experiments.md) (regenerate with `python scripts/experiment_table.py`). Run summaries are in `docs/results/*.json`.

## How the model was chosen

1. **Noise floor first.** The baseline config was run with 3 seeds (42, 43, 44): F1 **0.971 ± 0.002**. A single entity is ~0.05 points of micro F1, and USERNAME / URL F1 move ~1.5 points per entity (64-76 val entities), so differences below a few tenths of a point are noise.
2. **One-change-at-a-time sweep** (`scripts/tune_sweep.sh`), each run picking its best epoch on validation.
3. **Rule fixed before looking:** a change counts only if it beats the baseline by more than the seed spread; finalists are re-run on 3 seeds.
4. **Generalization tie-break** (`scripts/holdout_check.sh`): because all prompt templates appear in every split, validation only measures in-template performance. The finalists were retrained with one whole prompt style removed and scored on it (below). The rule, again fixed in advance: prefer cased only if it beats uncased there by more than ~1 point.

## Sweep results (validation, best epoch)

| change from baseline | F1 | macro F1 | vs. baseline 0.971 ± 0.002 |
|---|---|---|---|
| baseline (3 seeds) | 0.971 ± 0.002 | 0.956 | – |
| lr 3e-5 | 0.972 | 0.958 | noise |
| lr 8e-5 | 0.971 | 0.956 | noise |
| batch 16 | 0.971 | 0.957 | noise |
| 8 epochs | 0.977 | 0.964 | better (best epoch 7) |
| class weights, sqrt-inverse, cap 10 (all PII labels ×10 vs `O`) | 0.962 | 0.943 | **worse** |
| class weights, sqrt-inverse, cap 50 (URL/USERNAME ×43-46) | 0.963 | 0.935 | **worse** |
| **uncased** (3 seeds) | **0.985 ± 0.001** | **0.976** | **much better** |

Per category, uncased (3-seed mean) vs cased: NAME 0.993 vs 0.992, ADDRESS 0.979 vs 0.931, PHONE 0.972 vs 0.951, EMAIL 0.984 vs 0.975, URL 0.982 vs 0.966, USERNAME 0.946 vs 0.921. It reaches 0.977 after a single epoch and plateaus from epoch 2; the cased model is still improving at epoch 5-8.

A plausible (untested) explanation: the uncased tokenizer lowercases and strips accents, so names, usernames and URL parts split into fewer, more familiar pieces, and capitalization in these generated texts is not a reliable PII signal. For example, `AnaPérez` becomes `Ana ##P ##é ##rez` (cased) vs `ana ##per ##ez` (uncased).

## Class imbalance

`O` is 97.1% of word labels; `B-URL_PERSONAL` (490) and `B-USERNAME` (574) are the rarest. What was tried:

- **Class-weighted cross-entropy** (two strengths): recall barely moves (0.976 → 0.976 / 0.978) but precision drops (0.969 → 0.950 / 0.948), so F1 falls ~1 point. The model already finds almost all entities; weighting mainly adds false positives. Not used.
- **Lowering the `O` decision threshold** (`scripts/tune_threshold.py`, final model): offset chosen on one half of validation and scored on the other, both ways. The best offset is **0 on both halves**; any offset trades more precision than it gains recall. Not used, so inference needs no change.
- **Down-sampling documents with no PII:** not applicable. Only 9 of 3,547 training documents contain no PII.

## Generalization check: unseen prompt style

The 8 prompt templates are 4 prompt styles × 2 variants (0/4, 1/5, 2/6, 3/7 share the same instructions), so holding out one template is not really unseen. Style 0/4 was removed from train and validation, and the best checkpoint was scored once on its 628 train+val documents. Templates 6/7 (suggested on #1 as a possible team test set) were not touched.

| config | val F1 (other styles) | **unseen-style F1** | ADDRESS | USERNAME |
|---|---|---|---|---|
| cased, 5 epochs | 0.972 | 0.942 | 0.858 | 0.918 |
| cased, 8 epochs | 0.976 | 0.944 | 0.856 | 0.945 |
| **uncased, 5 epochs** | 0.985 | **0.967** | 0.922 | 0.953 |

Uncased stays ahead on unseen text, so it is the final model. Both models drop when the prompt style is new (uncased 0.985 → 0.967, cased 0.972 → 0.942), mostly on STREET_ADDRESS. That gap is the in-template vs. new-template generalization effect the report should discuss (#7).

## Known weaknesses of the final model (observations, not used for selection)

On the 11 hand-written prompts in `tests/examples.txt`, the uncased model finds every name, address and URL, but:

- **US phone format `415-555-1234` is not recognized**: 4 of 5 are missed and 1 is tagged `STREET_ADDRESS`. (The cased model tagged 3 of 5 as `STREET_ADDRESS`.) This format never occurs in the training data, which has 9 other phone formats.
- It misses the short username `dk123` and the email `test@example.net` in one sentence.

So the higher validation score does not cover formats the data never shows. Fixing this would need training data with those formats (augmentation), which would also contaminate the unfamiliar-PII test in #4/#7, so it was not done here. It is a good example for the error analysis (#9) and limitations section.

## Reproduce

```bash
bash scripts/tune_sweep.sh                     # round 1 (9 runs, ~40 min on an RTX 2080 SUPER)
python scripts/train.py --run-name uncased_s43 --model-name distilbert-base-uncased --seed 43   # (and s44)
bash scripts/holdout_check.sh                  # unseen-style check (3 runs, ~12 min)
python scripts/tune_threshold.py --model models/uncased
python scripts/experiment_table.py
# final model:
python scripts/train.py --run-name uncased --model-name distilbert-base-uncased
```
