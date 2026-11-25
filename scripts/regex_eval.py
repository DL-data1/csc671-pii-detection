import argparse
import ast
import re
import subprocess

import pandas as pd
from seqeval.metrics import classification_report


TYPES = {
    "EMAIL",
    "PHONE_NUM",
    "URL_PERSONAL",
}


def run_regex(text):
    result = subprocess.run(
        ["./regex_cli.exe"],
        input=text,
        text=True,
        capture_output=True,
        encoding="utf-8",
        check=True,
    )

    entities = []

    for line in result.stdout.splitlines():
        if not line:
            continue

        entity_type, start, end = line.split("\t")

        entities.append({
            "type": entity_type,
            "start": int(start),
            "end": int(end),
        })

    return entities


def get_offsets(text, tokens):
    offsets = []
    pos = 0

    for token in tokens:
        start = text.find(token, pos)

        if start == -1:
            raise ValueError(f"Cannot find token: {token}")

        end = start + len(token)

        offsets.append((start, end))
        pos = end

    return offsets


def filter_gold(labels):
    result = []

    for label in labels:
        if label == "O":
            result.append("O")
            continue

        prefix, entity_type = label.split("-", 1)

        if entity_type in TYPES:
            result.append(label)
        else:
            result.append("O")

    return result


def spans_to_bio(entities, offsets):
    labels = ["O"] * len(offsets)

    for entity in entities:
        if entity["type"] not in TYPES:
            continue

        indexes = []

        for i, (start, end) in enumerate(offsets):
            if (
                max(start, entity["start"])
                <
                min(end, entity["end"])
            ):
                indexes.append(i)

        for j, i in enumerate(indexes):
            prefix = "B" if j == 0 else "I"

            labels[i] = (
                f"{prefix}-{entity['type']}"
            )

    return labels


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/raw/test.csv"
    )

    args = parser.parse_args()

    df = pd.read_csv(args.data)

    gold_sequences = []
    pred_sequences = []

    for index, row in df.iterrows():
        text = row["text"]

        tokens = ast.literal_eval(
            row["tokens"]
        )

        gold = ast.literal_eval(
            row["labels"]
        )

        offsets = get_offsets(
            text,
            tokens
        )

        entities = run_regex(text)

        gold_sequences.append(
            filter_gold(gold)
        )

        pred_sequences.append(
            spans_to_bio(
                entities,
                offsets
            )
        )

        if (index + 1) % 100 == 0:
            print(
                f"Processed {index + 1} documents"
            )

    report = classification_report(
        gold_sequences,
        pred_sequences,
        output_dict=True,
        zero_division=0,
    )
    

    print("\nRegex baseline")
    print("=" * 40)

    overall = report["micro avg"]

    print("\nOVERALL")
    print(
        f"Precision: {overall['precision']:.4f}"
    )
    print(
        f"Recall:    {overall['recall']:.4f}"
    )
    print(
        f"F1:        {overall['f1-score']:.4f}"
    )

    for entity_type in TYPES:
        score = report.get(
            entity_type,
            {}
        )

        print(f"\n{entity_type}")

        print(
            f"Precision: "
            f"{score.get('precision', 0):.4f}"
        )
        print(
            f"Recall:    "
            f"{score.get('recall', 0):.4f}"
        )
        print(
            f"F1:        "
            f"{score.get('f1-score', 0):.4f}"
        )


if __name__ == "__main__":
    main()