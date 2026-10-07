"""CLI runner for DVC pipeline stage: extract."""

import argparse
import json
from pathlib import Path

from legal_rag.extraction.pdf_reader import parse_pdf


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract clean bilingual Egyptian Civil Code corpus."
    )
    parser.add_argument(
        "--pdf", required=True, type=Path, help="Path to input bilingual PDF file"
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="Path to output corpus JSON file"
    )
    args = parser.parse_args()

    if not args.pdf.is_file():
        parser.error(f"PDF file does not exist: {args.pdf}")

    print(f"Parsing bilingual PDF from {args.pdf}...")
    records = parse_pdf(str(args.pdf))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Successfully extracted {len(records)} articles -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
