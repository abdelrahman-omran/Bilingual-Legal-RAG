"""Core Egyptian Civil Code extraction logic."""
import json
import re
import unicodedata
from pathlib import Path
import pdfplumber
import pyarabic.araby as araby

AR_DIGITS = "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵٦٧٨٩"
AR2EN = str.maketrans(AR_DIGITS, "01234567890123456789")
RE_AR_LETTER = re.compile(r"[\u0621-\u064A]")

def ar_to_int(s: str) -> int:
    digits = re.sub(r"\D", "", s.translate(AR2EN))
    return int(digits) if digits else 0

def _is_weak_digit_run(text: str) -> bool:
    stripped = text.strip("()., ")
    return bool(stripped) and all(ch.isdigit() or ch in "()., " for ch in text)

def _word_glyphs(word: dict, chars: list[dict], rtl: bool) -> list[str]:
    matched = [
        c for c in chars
        if word["x0"] - 0.5 <= c["x0"] <= word["x1"] + 0.5 and abs(c["top"] - word["top"]) < 3
    ]
    matched.sort(key=lambda c: c["x0"], reverse=rtl)
    return [c["text"] for c in matched]

def _ordered_word_text(word: dict, chars: list[dict], rtl: bool) -> str:
    glyphs = _word_glyphs(word, chars, rtl)
    if not rtl or _is_weak_digit_run("".join(glyphs)):
        return "".join(glyphs)
    return "".join(glyphs)

def normalize_arabic(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(AR2EN)
    text = re.sub(r"\(\s*(\d+)\s*\(", r"(\1)", text)
    return re.sub(r"\s+", " ", text).strip()

def fix_arabic_output_artifacts(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\)([٠-٩0-9]+)\(', r'(\1)', text)
    text = re.sub(r'\s+،', '،', text)
    text = re.sub(r'\s+؛', '؛', text)
    text = re.sub(r'\s+:', ':', text)
    text = re.sub(r'\s+\.', '.', text)
    return text

def find_dynamic_split_x(words, page_width, band=(0.15, 0.85)):
    if not words:
        return page_width / 2
    n_bins = int(page_width) + 1
    occupied = [False] * n_bins
    for w in words:
        for i in range(max(0, int(w["x0"])), min(n_bins - 1, int(w["x1"])) + 1):
            occupied[i] = True
    lo_bound, hi_bound = band[0] * page_width, band[1] * page_width
    best_start, best_len = None, 0
    cur_start, cur_len = None, 0
    for i in range(n_bins):
        if lo_bound <= i <= hi_bound and not occupied[i]:
            cur_start = i if cur_start is None else cur_start
            cur_len += 1
        else:
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
            cur_start, cur_len = None, 0
    if cur_len > best_len:
        best_len, best_start = cur_len, cur_start
    return (best_start + best_len / 2) if best_start is not None else (page_width / 2)

def group_words_by_row(words, tol=3.0):
    words_sorted = sorted(words, key=lambda w: w["top"])
    rows, current_row, anchor_top = [], [], None
    for w in words_sorted:
        if anchor_top is None or abs(w["top"] - anchor_top) <= tol:
            current_row.append(w)
            anchor_top = w["top"] if anchor_top is None else anchor_top
        else:
            rows.append(current_row)
            current_row, anchor_top = [w], w["top"]
    if current_row:
        rows.append(current_row)
    return rows

def parse_pdf(pdf_path: str) -> list[dict]:
    articles_dict = {}
    hierarchy = {
        "book": "Preliminary Chapter (الباب التمهيدي)",
        "chapter": "General Provisions (أحكام عامة)",
        "section": "1. Laws and Rights",
        "topic": "General"
    }
    current_art = None
    seen_heading = False
    
    with pdfplumber.open(pdf_path) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(use_text_flow=False)
            chars = page.chars
            if not words:
                continue
            split_x = find_dynamic_split_x(words, page.width)
            rows = group_words_by_row(words)
            for row in rows:
                en_words = [w for w in row if w["x0"] < split_x and not RE_AR_LETTER.search(w["text"])]
                ar_words = [w for w in row if w["x0"] >= split_x]
                en_str = " ".join(w["text"] for w in sorted(en_words, key=lambda x: x["x0"])).strip()
                ar_sorted = sorted(ar_words, key=lambda x: x["x0"], reverse=True)
                ar_tokens = [_ordered_word_text(w, chars, rtl=True) for w in ar_sorted]
                ar_str = " ".join(ar_tokens).strip()
                ar_str = normalize_arabic(ar_str)
                ar_str = araby.strip_tashkeel(ar_str)
                ar_str = araby.strip_tatweel(ar_str)
                ar_str = fix_arabic_output_artifacts(ar_str)
                
                if re.search(r'SECTION|الفصل|الباب|باب|أحكام عامة', f"{en_str} {ar_str}", re.IGNORECASE):
                    seen_heading = True
                if not seen_heading:
                    continue
                
                en_match = re.search(r'\bArticle\s+([0-9]+)\b', en_str, re.IGNORECASE)
                ar_match = re.search(r'مادة\s*\(?\s*([٠-٩0-9]+)\s*\)?', ar_str)
                
                if en_match or ar_match:
                    art_num = int(en_match.group(1)) if en_match else ar_to_int(ar_match.group(1))
                    current_art = art_num
                    en_body = en_str[en_match.end():].strip() if en_match else ""
                    ar_body = re.sub(r'^(?:مادة\s*\(?\s*[٠-٩0-9]+\s*\)?\s*)', '', ar_str).strip() if ar_match else ar_str
                    if art_num not in articles_dict:
                        articles_dict[art_num] = {
                            "article_number": art_num,
                            "book": hierarchy["book"],
                            "chapter": hierarchy["chapter"],
                            "section": hierarchy["section"],
                            "topic": hierarchy["topic"],
                            "text_ar": ar_body,
                            "text_en": en_body,
                            "is_repealed": False,
                            "source_page": pno,
                            "citation": f"Egyptian Civil Code, Article {art_num}"
                        }
                elif current_art is not None and current_art in articles_dict:
                    if re.search(r'SECTION\s+[I|V|X]+', en_str, re.IGNORECASE):
                        hierarchy["chapter"] = en_str
                    elif re.search(r'^\d+[\.\-]\s+', en_str):
                        hierarchy["section"] = en_str
                    else:
                        if en_str:
                            articles_dict[current_art]["text_en"] = (articles_dict[current_art]["text_en"] + " " + en_str).strip()
                        if ar_str:
                            articles_dict[current_art]["text_ar"] = (articles_dict[current_art]["text_ar"] + " " + ar_str).strip()

    articles_list = sorted(articles_dict.values(), key=lambda x: x["article_number"])
    for a in articles_list:
        a["is_repealed"] = bool(re.search(r'\brepealed\b', a["text_en"].lower()) or "ملغاة" in a["text_ar"] or "ملغى" in a["text_ar"])
        a["text_en"] = ' '.join(a["text_en"].split())
        a["text_ar"] = ' '.join(a["text_ar"].split())

    return articles_list