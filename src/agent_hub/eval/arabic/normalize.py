"""Arabic orthographic normalisation, one toggle per transform.

Why one toggle per transform: the interesting question is never "does
normalisation help?" (it does) but "which normalisation, and at what cost?".
Every transform here destroys information in exchange for recall, and the
trade is different for each one. Bundling them behind a single boolean makes
that trade invisible, so each is independently switchable and the harness
measures them one at a time.

The retrieval consequence, concretely: Arabic writers are inconsistent about
hamza seating, taa marbuta, and diacritics. The SAME word routinely appears as
several distinct byte sequences across a corpus. Lexical matching sees these as
unrelated tokens, and embedding models — trained on text with the same
inconsistency — give them merely similar, not identical, vectors. Normalising
both sides of the comparison collapses the variants onto one form.

Reference for the code points below: Unicode Arabic block U+0600–U+06FF.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace

__all__ = [
    "NormalizeConfig",
    "OFF",
    "CONSERVATIVE",
    "AGGRESSIVE",
    "normalize",
    "per_transform_variants",
    "TRANSFORMS",
]

# --- code points -------------------------------------------------------------

# Harakat (short-vowel marks) + shadda + sukun. Optional in writing; usually
# absent in body text, but present in Qur'anic, poetic, legal, and pedagogical
# text — which means a corpus can be internally inconsistent about them.
_DIACRITICS = re.compile(r"[ً-ْٓ-ٰٕ]")

# Tatweel / kashida U+0640 — a pure justification glyph with no phonetic value.
# Stretches a word for typographic alignment: الإجـــراء == الإجراء.
_TATWEEL = re.compile(r"ـ")

# Alef with hamza above/below, madda, wasla → bare alef.
_ALEF_VARIANTS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})

# Hamza on a carrier → the bare carrier letter.
_HAMZA_CARRIERS = str.maketrans({"ؤ": "و", "ئ": "ي"})

# Standalone hamza U+0621 — removed only under AGGRESSIVE.
_BARE_HAMZA = re.compile(r"ء")

_TAA_MARBUTA = str.maketrans({"ة": "ه"})       # ة → ه
_ALEF_MAQSURA = str.maketrans({"ى": "ي"})      # ى → ي

# Arabic-Indic (U+0660-0669) and Extended/Persian (U+06F0-06F9) → ASCII.
_DIGITS = str.maketrans(
    {chr(0x0660 + i): str(i) for i in range(10)} | {chr(0x06F0 + i): str(i) for i in range(10)}
)

# Arabic punctuation → ASCII equivalents. Comma, semicolon, question mark,
# and the full stop used in some regional typography.
_PUNCT = str.maketrans({"،": ",", "؛": ";", "؟": "?", "۔": "."})

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class NormalizeConfig:
    """Which transforms to apply. Every field defaults to off."""

    nfc: bool = False
    strip_diacritics: bool = False
    strip_tatweel: bool = False
    unify_alef: bool = False
    unify_hamza: bool = False
    strip_bare_hamza: bool = False
    unify_taa_marbuta: bool = False
    unify_alef_maqsura: bool = False
    unify_digits: bool = False
    unify_punctuation: bool = False
    collapse_whitespace: bool = False

    def label(self) -> str:
        on = [f.replace("_", "-") for f, v in vars(self).items() if v]
        if not on:
            return "off"
        return "+".join(on)


OFF = NormalizeConfig()

#: Safe, near-lossless. Every transform here removes a mark that carries no
#: lexical distinction in Modern Standard Arabic prose, so nothing that
#: distinguishes two real words is destroyed.
CONSERVATIVE = NormalizeConfig(
    nfc=True,
    strip_diacritics=True,
    strip_tatweel=True,
    unify_alef=True,
    unify_digits=True,
    unify_punctuation=True,
    collapse_whitespace=True,
)

#: Maximum recall, accepted information loss. Adds the transforms that CAN
#: merge genuinely distinct words — see the cost notes in TRANSFORMS below.
#: Use when recall matters more than precision, and measure the precision cost
#: rather than assuming it is small.
AGGRESSIVE = replace(
    CONSERVATIVE,
    unify_hamza=True,
    strip_bare_hamza=True,
    unify_taa_marbuta=True,
    unify_alef_maqsura=True,
)


#: name → (what it does, what it costs). Used to render the per-transform table
#: in the eval report, so the cost is visible next to the recall gain.
TRANSFORMS: dict[str, tuple[str, str]] = {
    "nfc": (
        "Unicode NFC composition",
        "None. Pure byte-level canonicalisation.",
    ),
    "strip_diacritics": (
        "Remove harakat, shadda, sukun, superscript alef",
        "Merges words distinguished only by vowelling (عِلْم/عَلَم). Rare in prose, "
        "common in religious and pedagogical text.",
    ),
    "strip_tatweel": (
        "Remove U+0640 kashida",
        "None. Tatweel is purely typographic.",
    ),
    "unify_alef": (
        "أ إ آ ٱ → ا",
        "Low. Hamza seating on alef is inconsistently written by most authors, "
        "so the distinction is already unreliable in real corpora.",
    ),
    "unify_hamza": (
        "ؤ → و,  ئ → ي",
        "Moderate. Can merge distinct roots.",
    ),
    "strip_bare_hamza": (
        "Remove standalone ء",
        "Moderate. Deletes a real consonant.",
    ),
    "unify_taa_marbuta": (
        "ة → ه",
        "HIGH. Destroys the feminine-ending distinction; merges pairs like "
        "مدرسة/مدرسه but also unrelated words. The single lossiest transform here.",
    ),
    "unify_alef_maqsura": (
        "ى → ي",
        "Moderate. Merges على/علي — a preposition and a proper name.",
    ),
    "unify_digits": (
        "٠-٩ and ۰-۹ → 0-9",
        "None for retrieval; keep the original if you display the text back.",
    ),
    "unify_punctuation": (
        "، ؛ ؟ ۔ → , ; ? .",
        "None for retrieval.",
    ),
    "collapse_whitespace": (
        "Collapse runs of whitespace to one space",
        "None.",
    ),
}


def normalize(text: str, config: NormalizeConfig = CONSERVATIVE) -> str:
    """Apply the enabled transforms, in an order chosen so they compose safely.

    Order matters: NFC first (so later rules see composed forms), diacritics and
    tatweel before letter mapping (so marks don't sit between a letter and its
    replacement), and whitespace last (so earlier deletions can't leave doubled
    spaces behind).
    """
    if not text:
        return text

    if config.nfc:
        text = unicodedata.normalize("NFC", text)
    if config.strip_diacritics:
        text = _DIACRITICS.sub("", text)
    if config.strip_tatweel:
        text = _TATWEEL.sub("", text)
    if config.unify_alef:
        text = text.translate(_ALEF_VARIANTS)
    if config.unify_hamza:
        text = text.translate(_HAMZA_CARRIERS)
    if config.strip_bare_hamza:
        text = _BARE_HAMZA.sub("", text)
    if config.unify_taa_marbuta:
        text = text.translate(_TAA_MARBUTA)
    if config.unify_alef_maqsura:
        text = text.translate(_ALEF_MAQSURA)
    if config.unify_digits:
        text = text.translate(_DIGITS)
    if config.unify_punctuation:
        text = text.translate(_PUNCT)
    if config.collapse_whitespace:
        text = _WHITESPACE.sub(" ", text).strip()

    return text


def per_transform_variants(base: NormalizeConfig = OFF) -> dict[str, NormalizeConfig]:
    """One config per transform, each enabling exactly that transform over `base`.

    This is what makes the ablation possible: run retrieval once per variant and
    the difference between each result and `base` is that transform's isolated
    contribution, uncontaminated by the others.
    """
    return {name: replace(base, **{name: True}) for name in vars(base)}
