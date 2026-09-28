"""Integration tests verifying real extracted corpus quality."""

import json
from pathlib import Path
import pytest
from legal_rag.extraction.validate import validate_corpus

CORPUS_JSON_PATH = Path("data/processed/corpus.json")


@pytest.mark.skipif(
    not CORPUS_JSON_PATH.exists(),
    reason="data/processed/corpus.json not found. Run 'dvc repro' first.",
)
def test_extracted_corpus_integrity():
    with open(CORPUS_JSON_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    # 1. Structural Validation (Strict contiguity across all articles)
    validate_corpus(records)

    # 2. Spot-check first 20 articles for valid Arabic script and non-empty text
    spot_check_records = records[:20]
    assert len(spot_check_records) == 20, "Corpus contains fewer than 20 articles."

    for rec in spot_check_records:
        art_num = rec["article_number"]
        ar_text = rec["text_ar"]

        assert isinstance(art_num, int), f"Article {art_num} is not an integer."
        assert len(ar_text.strip()) > 5, f"Article {art_num} Arabic text is suspiciously short."
        assert any(
            "\u0600" <= char <= "\u06ff" for char in ar_text
        ), f"Article {art_num} does not contain valid Arabic Unicode characters."


@pytest.mark.skipif(
    not CORPUS_JSON_PATH.exists(),
    reason="data/processed/corpus.json not found. Run 'dvc repro' first.",
)
def test_extracted_corpus_parser_bug_fixes():
    """Regression test for specific parser bug fixes."""
    with open(CORPUS_JSON_PATH, "r", encoding="utf-8") as f:
        records_by_num = {r["article_number"]: r for r in json.load(f)}

    # Fix 1: Repealed ranges generated correctly (Articles 54-80 and 389-417)
    for art_num in list(range(54, 81)) + list(range(389, 418)):
        assert art_num in records_by_num, f"Repealed Article {art_num} was not generated."
        assert records_by_num[art_num]["is_repealed"] is True, f"Article {art_num} is_repealed should be True."

    # Article 388 should NOT be flagged as repealed
    assert records_by_num[388]["is_repealed"] is False, "Article 388 misflagged as repealed."

    # Fix 2: Tashkeel / Tanween diacritics preserved
    tanween_found = any("\u064b" in r["text_ar"] for r in records_by_num.values())
    assert tanween_found, "Tashkeel/Tanween diacritics were not preserved in Arabic text."

    # Fix 3: Arabic numbers not reversed
    art_389_ar = records_by_num[389]["text_ar"]
    assert "389" in art_389_ar or "٣٨٩" in art_389_ar, "Arabic numbers were not formatted in LTR order."
    assert "983" not in art_389_ar, "Arabic numbers are still reversed in text_ar."

    # Fix 4: Topic hierarchy preserved sub-topic and parent topic
    art_10 = records_by_num[10]
    assert "Conflicts of law as to place" in art_10["topic"] or "تنازع القوانين من حيث المكان" in art_10["topic"]
    assert "Application of Laws" in art_10["topic"] or "تطبيق القانون" in art_10["topic"]