import argparse
import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

"""
DistilBERT PII inference script.

Loads a trained DistilBERT token-classification checkpoint and detects
personally identifiable information (PII) in raw input text.

The script is designed to work with DistilBERT checkpoint,
but it can load any compatible checkpoint directory using:

    AutoTokenizer.from_pretrained(<dir>)
    AutoModelForTokenClassification.from_pretrained(<dir>)

The label mapping is read directly from the checkpoint's config.json through
`model.config.id2label`, so no separate label file is required.

Important preprocessing:
    The training data uses whitespace-split words with punctuation attached,
    for example:

        "Feng,"
        "Drive."
        "maria@example.com."

    To match training behavior, inference also splits input text on whitespace
    before tokenizing with:

        is_split_into_words=True

Prediction handling:
    The model was trained using the label on only the first subword of each
    original word. Other subwords were ignored with label -100.

    During inference, this script:
        1. Splits raw text into whitespace-separated words.
        2. Tokenizes the words with the DistilBERT tokenizer.
        3. Runs the token-classification model.
        4. Keeps the prediction from the first subword of each word.
        5. Merges B-/I- BIO predictions into complete PII entity spans.
        6. Reconstructs character offsets in the original input text.
        7. Prints detected PII spans, entity types, offsets, and highlighted text.

Long input:
    Input is currently truncated at 512 subword tokens to match the current
    training setup. Sliding-window inference can be added later for longer
    documents.

Usage:
    python scripts/predict.py --model models/distilbert-small \
        "My name is John Smith and my email is john@example.com."

Example output:
    'John Smith'         type=NAME_STUDENT
    'john@example.com.'  type=EMAIL

    Highlighted text:
    My name is [[NAME_STUDENT: John Smith]] and my email is
    [[EMAIL: john@example.com.]]
"""


def split_words_with_offsets(text):
    words = []
    offsets = []

    for match in re.finditer(r"\S+", text):
        words.append(match.group())
        offsets.append((match.start(), match.end()))

    return words, offsets


def merge_bio_predictions(words, offsets, labels):
    entities = []
    current = None

    for word, (start, end), label in zip(words, offsets, labels):

        if label == "O":
            if current is not None:
                entities.append(current)
                current = None
            continue

        if "-" not in label:
            continue

        prefix, entity_type = label.split("-", 1)

        if prefix == "B":
            if current is not None:
                entities.append(current)

            current = {"type": entity_type, "start": start, "end": end}

        elif prefix == "I":
            if current is not None and current["type"] == entity_type:
                current["end"] = end

            else:
                if current is not None:
                    entities.append(current)

                current = {"type": entity_type, "start": start, "end": end}

    if current is not None:
        entities.append(current)

    return entities


def highlight_text(text, entities):
    result = ""
    last_end = 0

    for entity in entities:
        start = entity["start"]
        end = entity["end"]
        entity_type = entity["type"]

        result += text[last_end:start]

        result += f"[[{entity_type}: " f"{text[start:end]}]]"

        last_end = end

    result += text[last_end:]

    return result


def redact_text(text, entities):
    result = ""
    last_end = 0

    for entity in entities:
        start = entity["start"]
        end = entity["end"]
        entity_type = entity["type"]

        result += text[last_end:start]
        result += f"[{entity_type}]"

        last_end = end

    result += text[last_end:]

    return result


def predict(model_path, text):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    model = AutoModelForTokenClassification.from_pretrained(model_path)

    model.to(device)
    model.eval()

    words, offsets = split_words_with_offsets(text)

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        padding="max_length",
        return_tensors="pt",
    )

    model_inputs = {
        key: value.to(device)
        for key, value in encoding.items()
        if key in tokenizer.model_input_names
    }

    with torch.no_grad():
        outputs = model(**model_inputs)

    predicted_ids = torch.argmax(outputs.logits, dim=-1).cpu().tolist()

    word_predictions = {}

    num_windows = len(predicted_ids)

    for window_index in range(num_windows):

        word_ids = encoding.word_ids(batch_index=window_index)

        seen_in_window = set()

        for token_index, word_id in enumerate(word_ids):

            if word_id is None:
                continue

            if word_id in seen_in_window:
                continue

            seen_in_window.add(word_id)


            if word_id in word_predictions:
                continue

            label_id = predicted_ids[window_index][token_index]

            label = model.config.id2label[label_id]

            word_predictions[word_id] = label

    word_labels = []

    for word_id in range(len(words)):
        label = word_predictions.get(word_id, "O")
        word_labels.append(label)

    entities = merge_bio_predictions(words, offsets, word_labels)

    print("Windows processed:", num_windows)

    return entities, len(words), len(words)


def main():

    parser = argparse.ArgumentParser(
        description="Detect PII using a DistilBERT checkpoint"
    )

    parser.add_argument("--model", required=True, help="Path to checkpoint directory")

    parser.add_argument("text", help="Text to scan for PII")
    parser.add_argument(
        "--redact",
        action="store_true",
        help="Replace detected PII with entity type labels",
    )

    args = parser.parse_args()

    entities, processed, total = predict(args.model, args.text)

    print("\nDetected PII:")
    print("-" * 70)

    if not entities:
        print("No PII detected.")

    else:
        for entity in entities:

            span = args.text[entity["start"] : entity["end"]]

            print(
                f"{span!r:30} "
                f"type={entity['type']:20} "
                f"offset=({entity['start']}, {entity['end']})"
            )

    print("\nHighlighted text:")
    print("-" * 70)
    print(highlight_text(args.text, entities))

    json_entities = []

    for entity in entities:
        json_entities.append(
            {
                "type": entity["type"],
                "text": args.text[entity["start"] : entity["end"]],
                "start": entity["start"],
                "end": entity["end"],
            }
        )

    print("\nJSON output:")
    print("-" * 70)
    print(json.dumps(json_entities, indent=2, ensure_ascii=False))

    if args.redact:
        print("\nRedacted text:")
        print("-" * 70)
        print(redact_text(args.text, entities))

    if processed < total:
        print(
            f"\nWarning: input was truncated " f"({processed}/{total} words processed)."
        )


if __name__ == "__main__":
    main()
