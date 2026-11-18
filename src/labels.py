"""Shared BIO label set for PII token classification.

The order of LABELS is fixed: the model's output ids depend on it, so do not
reorder. Every checkpoint also stores these maps in its config.json
(`id2label` / `label2id`), so inference can decode predictions from the
checkpoint alone.
"""

ENTITY_TYPES = [
    "NAME_STUDENT",
    "STREET_ADDRESS",
    "PHONE_NUM",
    "EMAIL",
    "URL_PERSONAL",
    "USERNAME",
]

# "O" + B-/I- for each entity type = 13 tags. I-EMAIL, I-URL_PERSONAL and
# I-USERNAME never occur in the current data but are kept for a standard BIO set.
LABELS = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")]

label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}
