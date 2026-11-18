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

The label set lives in `src/labels.py` (13 BIO tags). The checkpoint's `config.json` stores `id2label`, so any checkpoint directory loads with `AutoModelForTokenClassification.from_pretrained(<dir>)`.

**Inference input:** the training `tokens` are whitespace-split words with punctuation attached (e.g. `"Feng,"`). Split raw text with `text.split()` and tokenize with `is_split_into_words=True`; read each word's label from its first subword.
