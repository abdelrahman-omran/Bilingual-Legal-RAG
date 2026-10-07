"""Unit tests for corpus validation bounds and assertions."""

import pytest

from legal_rag.extraction.validate import MAX_ARTICLE_LENGTH_CHARS, validate_corpus


def test_validate_corpus_success():
    valid_records = [
        {
            "article_number": 1,
            "text_ar": "المادة الأولى ينفذ القانون",
            "text_en": "Article 1 shall take effect",
            "is_repealed": False,
        },
        {
            "article_number": 2,
            "text_ar": "المادة ألغيت بموجب قانون",
            "text_en": "Article 2 was repealed by law",
            "is_repealed": True,
        },
    ]
    validate_corpus(valid_records)


def test_validate_corpus_contiguity_gap_fails():
    gap_records = [
        {
            "article_number": 1,
            "text_ar": "نص المادة 1",
            "text_en": "Text 1",
            "is_repealed": False,
        },
        {
            "article_number": 3,  # Missing Article 2
            "text_ar": "نص المادة 3",
            "text_en": "Text 3",
            "is_repealed": False,
        },
    ]
    with pytest.raises(ValueError, match="Contiguity gap detected"):
        validate_corpus(gap_records)


def test_validate_corpus_empty_arabic_fails():
    empty_ar_records = [
        {
            "article_number": 1,
            "text_ar": "   ",
            "text_en": "Text 1",
            "is_repealed": False,
        }
    ]
    with pytest.raises(ValueError, match="empty text_ar"):
        validate_corpus(empty_ar_records)


def test_validate_corpus_exceeds_max_length_fails():
    huge_records = [
        {
            "article_number": 1,
            "text_ar": "أ" * (MAX_ARTICLE_LENGTH_CHARS + 100),
            "text_en": "Text",
            "is_repealed": False,
        }
    ]
    with pytest.raises(ValueError, match="exceeds sane length threshold"):
        validate_corpus(huge_records)
