# CSC 671 PII Detection

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Small training run (issue #2)

Trains `distilbert-base-cased` on a 200-doc train / 50-doc val sample of `data/raw/` and saves the model and tokenizer to `models/distilbert-small/`:

```bash
python scripts/train_small.py
```

Options: `--train-docs`, `--val-docs` (0 = all), `--epochs`, `--lr`, `--batch-size`, `--output`.

## Full training run (issue #5)

Trains `distilbert-base-cased` on the full train split, evaluates on validation every epoch, and keeps the checkpoint with the best validation entity-level F1:

```bash
python scripts/train.py                          # defaults: 5 epochs, lr 5e-5, batch 8 -> models/full_baseline/
python scripts/train.py --lr 3e-5 --run-name lr3e-5
```

- **Long documents:** split into overlapping windows of 512 subwords with a 128-subword overlap (`src/windows.py`). Training uses every window.
- **Validation is document-level:** each word's logits are averaged over all windows that contain it, then seqeval scores whole documents. Nothing is truncated or double-counted.
- **Outputs:** the best model + tokenizer, `train_info.json` (hyperparameters, per-epoch metrics, loss curve, runtime, GPU) and `val_report.txt` in `models/<run-name>/`. A copy of the summary goes to `docs/results/<run-name>.json` (tracked) for the experiment table. Results are written up in `docs/training.md`.
- The test split is never used for training or model selection.

The label set lives in `src/labels.py` (13 BIO tags). The checkpoint's `config.json` stores `id2label`, so any checkpoint directory loads with `AutoModelForTokenClassification.from_pretrained(<dir>)`.

**Inference input:** the training `tokens` are whitespace-split words with punctuation attached (e.g. `"Feng,"`). Split raw text with `text.split()` and tokenize with `is_split_into_words=True`; read each word's label from its first subword.


## Inference

Run PII detection with a compatible checkpoint:

```bash
python scripts/predict.py --model models/distilbert-small "My name is John Smith and my email is john@example.com."
```

Optional redaction mode:

```bash
python scripts/predict.py --model models/distilbert-small --redact "My name is John Smith and my email is john@example.com."
```

The inference script supports:

- highlighted PII spans
- JSON output with `type`, `text`, `start`, and `end`
- optional PII redaction
- sliding-window inference for inputs longer than 512 subword tokens

Example prompts are available in `tests/examples.txt`.

The script loads the tokenizer and model from the checkpoint directory, merges BIO predictions into complete PII spans, and prints the detected entity type, character offsets, and highlighted text.

Example output:

```text
Detected PII:
----------------------------------------------------------------------
'John Smith'                   type=NAME_STUDENT         offset=(11, 21)
'john@example.com.'            type=EMAIL                offset=(38, 55)

Highlighted text:
----------------------------------------------------------------------
My name is [[NAME_STUDENT: John Smith]] and my email is [[EMAIL: john@example.com.]]
```

Long inputs are handled with sliding-window inference so text beyond the first 512 subword tokens can also be processed.

## Web UI

A simple Gradio demo is available in scripts/app.py.

Run it from the project root:

python scripts/app.py

Then open the local URL shown in the terminal (usually http://127.0.0.1:7860) in a browser.