"""Sliding-window encoding and window -> word aggregation for long documents.

DistilBERT sees at most 512 subwords, and ~16% of documents are longer. Each
document is split into overlapping windows (max_length 512, stride 128 subwords
of overlap). Training uses every window. At evaluation/inference time the
logits for a word that appears in several windows are averaged, and the word's
label is the argmax of that average.

Only the first subword of each word carries a label. A window can start in the
middle of a word, so "first subword" is decided from the tokenizer's per-word
offsets (offset start == 0), not from "first time this word id appears in the
window".

Use the same MAX_LENGTH / STRIDE / aggregation at inference so reported scores
match the demo.
"""

import numpy as np

MAX_LENGTH = 512
STRIDE = 128


def encode_windows(tokenizer, words, word_labels=None, label2id=None, max_length=MAX_LENGTH, stride=STRIDE):
    """Encode one document (list of words) into overlapping windows.

    Returns (features, metas):
      features: list of {"input_ids", "attention_mask"[, "labels"]} per window,
                ready for DataCollatorForTokenClassification.
      metas:    list of {"word_ids": [...], "first": [...]} per window, where
                first[j] is True when token j is the first subword of its word.
    """
    enc = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    features, metas = [], []
    for w in range(len(enc["input_ids"])):
        word_ids = enc.word_ids(w)
        first = [wid is not None and off[0] == 0 for wid, off in zip(word_ids, enc["offset_mapping"][w])]
        feat = {"input_ids": enc["input_ids"][w], "attention_mask": enc["attention_mask"][w]}
        if word_labels is not None:
            feat["labels"] = [label2id[word_labels[wid]] if f else -100 for wid, f in zip(word_ids, first)]
        features.append(feat)
        metas.append({"word_ids": word_ids, "first": first})
    return features, metas


def aggregate_word_logits(n_words, window_logits, metas):
    """Average first-subword logits per word across all windows of one document.

    window_logits: sequence of [seq_len, num_labels] arrays (padding allowed
    past each window's real length). Returns [n_words, num_labels]; words that
    got no subword (should not happen with this data) stay all-zero and are
    reported in the second return value.
    """
    num_labels = window_logits[0].shape[-1]
    sums = np.zeros((n_words, num_labels), dtype=np.float64)
    counts = np.zeros(n_words, dtype=np.int64)
    for logits, meta in zip(window_logits, metas):
        for j, (wid, f) in enumerate(zip(meta["word_ids"], meta["first"])):
            if f:
                sums[wid] += logits[j]
                counts[wid] += 1
    missing = np.flatnonzero(counts == 0).tolist()
    counts[counts == 0] = 1
    return sums / counts[:, None], missing
