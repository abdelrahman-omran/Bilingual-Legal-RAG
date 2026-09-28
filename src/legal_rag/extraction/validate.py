"""Validation rules for extracted legal corpus records."""

MAX_ARTICLE_LENGTH_CHARS = 10_000


def validate_corpus(records: list[dict]) -> None:
    """Validate extracted records for contiguity, non-empty text, length caps, and repealed flags.

    Raises:
        ValueError: If any validation condition is violated.
    """
    if not records:
        raise ValueError("Corpus validation failed: Record list is empty.")

    previous_art_num = None

    for idx, rec in enumerate(records):
        art_num = rec.get("article_number")
        text_ar = rec.get("text_ar", "")
        text_en = rec.get("text_en", "")
        is_repealed = rec.get("is_repealed", False)

        # 1. Integer Article Number & Contiguity Check
        if not isinstance(art_num, int):
            raise ValueError(f"Record index {idx} has non-integer article_number: {art_num}")

        if previous_art_num is not None:
            # Asserts monotonic incrementing (no missing articles or unexpected sequence jumps)
            if art_num != previous_art_num + 1:
                raise ValueError(
                    f"Contiguity gap detected between article {previous_art_num} and {art_num}"
                )
        previous_art_num = art_num

        # 2. Non-empty Arabic Text Check
        if not text_ar or not text_ar.strip():
            raise ValueError(f"Article {art_num} has empty text_ar.")

        # 3. Maximum Length Cap Check (Prevents merged split failures)
        combined_len = len(text_ar) + len(text_en)
        if combined_len > MAX_ARTICLE_LENGTH_CHARS:
            raise ValueError(
                f"Article {art_num} exceeds sane length threshold ({combined_len} > {MAX_ARTICLE_LENGTH_CHARS} chars). "
                "Possible split failure."
            )

        # 4. Repealed Flag Logic Check
        has_repealed_keyword = (
            "repealed" in text_en.lower() or "ملغاة" in text_ar or "ملغى" in text_ar
        )
        if has_repealed_keyword and not is_repealed:
            raise ValueError(
                f"Article {art_num} contains repealed wording but is_repealed is set to False."
            )