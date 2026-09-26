"""Arabic normalisation unit tests — pure functions, no stack, no network.

Arabic strings are built from explicit \\u escapes wherever a diacritic or an
invisible character is involved. Pasted Arabic looks identical on screen whether
or not it carries marks, so a literal would make these tests unreadable and
un-reviewable.
"""

from dataclasses import replace

from agent_hub.eval.arabic.normalize import (
    AGGRESSIVE,
    CONSERVATIVE,
    OFF,
    TRANSFORMS,
    NormalizeConfig,
    normalize,
    per_transform_variants,
)

# "الإجراء" — the bare, unmarked form every variant below should collapse onto.
BARE = "الإجراء"

# Same word carrying harakat: sukun on lam, kasra on alef-hamza, sukun on jim,
# fatha on ra.
VOWELLED = (
    "الْإِجْرَاء"
)

# Same word stretched with four tatweel (kashida) characters.
TATWEELED = "الإج" + "ـ" * 4 + "راء"


def test_off_is_identity():
    assert normalize(VOWELLED, OFF) == VOWELLED
    assert normalize(TATWEELED, OFF) == TATWEELED


def test_strip_diacritics_recovers_bare_form():
    cfg = NormalizeConfig(strip_diacritics=True)
    assert normalize(VOWELLED, cfg) == BARE


def test_strip_tatweel_recovers_bare_form():
    cfg = NormalizeConfig(strip_tatweel=True)
    assert normalize(TATWEELED, cfg) == BARE


def test_unify_alef_collapses_all_four_variants():
    cfg = NormalizeConfig(unify_alef=True)
    plain = "ا"  # ا
    for variant in ("أ", "إ", "آ", "ٱ"):  # أ إ آ ٱ
        assert normalize(variant, cfg) == plain


def test_unify_hamza_maps_carriers_to_base_letters():
    cfg = NormalizeConfig(unify_hamza=True)
    assert normalize("ؤ", cfg) == "و"  # ؤ → و
    assert normalize("ئ", cfg) == "ي"  # ئ → ي


def test_unify_taa_marbuta():
    cfg = NormalizeConfig(unify_taa_marbuta=True)
    # اللائحة → اللائحه
    assert normalize("اللائحة", cfg) == (
        "اللائحه"
    )


def test_unify_alef_maqsura():
    cfg = NormalizeConfig(unify_alef_maqsura=True)
    assert normalize("على", cfg) == "علي"  # على → علي


def test_unify_digits_handles_both_arabic_indic_ranges():
    cfg = NormalizeConfig(unify_digits=True)
    assert normalize("٢٠٢٦", cfg) == "2026"  # ٢٠٢٦
    assert normalize("۲۰۲۶", cfg) == "2026"  # ۲۰۲۶ (extended)


def test_unify_punctuation():
    cfg = NormalizeConfig(unify_punctuation=True)
    assert normalize("،؛؟", cfg) == ",;?"  # ، ؛ ؟


def test_collapse_whitespace():
    cfg = NormalizeConfig(collapse_whitespace=True)
    assert normalize("  a \t\n b  ", cfg) == "a b"


# --- the property that actually matters for retrieval ------------------------


def test_conservative_collapses_orthographic_variants_onto_one_form():
    """The whole point: spelling variants of one word must become one string.

    If this fails, lexical matching sees unrelated tokens and the embedding
    model sees merely-similar vectors — which is exactly the silent Arabic
    retrieval failure the harness exists to measure.
    """
    # "إجراءات" written with hamza-under-alef, and with bare alef.
    with_hamza = "إجراءات"
    without_hamza = "اجراءات"
    assert normalize(with_hamza, CONSERVATIVE) == normalize(without_hamza, CONSERVATIVE)


def test_conservative_collapses_vowelled_and_tatweeled_onto_bare():
    forms = [BARE, VOWELLED, TATWEELED]
    normalized = {normalize(f, CONSERVATIVE) for f in forms}
    assert len(normalized) == 1, f"expected one form, got {normalized}"


def test_conservative_preserves_taa_marbuta_but_aggressive_does_not():
    """CONSERVATIVE is documented as near-lossless; taa marbuta is the lossy one."""
    word = "مدرسة"  # مدرسة
    assert "ة" in normalize(word, CONSERVATIVE)
    assert "ة" not in normalize(word, AGGRESSIVE)


def test_aggressive_is_a_superset_of_conservative():
    for field, enabled in vars(CONSERVATIVE).items():
        if enabled:
            assert getattr(AGGRESSIVE, field), f"AGGRESSIVE dropped {field}"


# --- ablation plumbing -------------------------------------------------------


def test_per_transform_variants_enables_exactly_one_each():
    variants = per_transform_variants(OFF)
    assert set(variants) == set(vars(OFF))
    for name, cfg in variants.items():
        on = [f for f, v in vars(cfg).items() if v]
        assert on == [name], f"{name} variant enabled {on}"


def test_per_transform_variants_layers_over_a_base():
    base = NormalizeConfig(nfc=True)
    variants = per_transform_variants(base)
    assert variants["strip_tatweel"].nfc is True
    assert variants["strip_tatweel"].strip_tatweel is True


def test_every_config_field_is_documented_in_transforms():
    """A new toggle without a documented cost would silently skip the report."""
    assert set(TRANSFORMS) == set(vars(OFF))


def test_label_describes_enabled_transforms():
    assert OFF.label() == "off"
    assert replace(OFF, strip_tatweel=True).label() == "strip-tatweel"


def test_empty_string_is_safe():
    assert normalize("", CONSERVATIVE) == ""
    assert normalize("", AGGRESSIVE) == ""
