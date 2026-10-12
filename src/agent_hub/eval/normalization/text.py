"""Explicit experiment transforms. No stemming, stop words, or Unicode folding."""

import unicodedata

PRESETS = [
    {"id": "raw", "label": "Raw", "description": "No character transformations."},
    {"id": "diacritics", "label": "Diacritics", "description": "Remove U+064B–U+0652 only."},
    {"id": "tatweel", "label": "Tatweel", "description": "Remove U+0640 only."},
    {"id": "alef", "label": "Alef", "description": "Map أ, إ, آ to ا only."},
    {
        "id": "conservative",
        "label": "Conservative",
        "description": "Diacritics + tatweel + Alef; hamza combining marks remain.",
    },
    {
        "id": "aggressive",
        "label": "Aggressive",
        "description": "Conservative + ى→ي + ة→ه; deliberately conflates distinctions.",
    },
]
TOKENIZER = (
    "unicode-category-v1: contiguous Unicode L/M/N characters or underscore; "
    "combining marks and tatweel preserved; no lowercase, NFC/NFKC, stemming, "
    "stopwords, or other normalization. BM25 uses unique query terms."
)


def normalize(text: str, preset: str) -> str:
    if preset not in {p["id"] for p in PRESETS}:
        raise ValueError(f"Unknown normalization preset: {preset}")
    if preset in ("diacritics", "conservative", "aggressive"):
        text = "".join(c for c in text if not "\u064b" <= c <= "\u0652")
    if preset in ("tatweel", "conservative", "aggressive"):
        text = text.replace("\u0640", "")
    if preset in ("alef", "conservative", "aggressive"):
        text = text.translate(str.maketrans("أإآ", "ااا"))
    if preset == "aggressive":
        text = text.translate(str.maketrans("ىة", "يه"))
    return text


def tokenize(text: str) -> list[str]:
    tokens, current = [], []
    for char in text:
        if unicodedata.category(char)[0] in "LMN" or char == "_":
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens
