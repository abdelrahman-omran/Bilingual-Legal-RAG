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
    # Added '-' and '/' to ensure number ranges (like 389-417) stay Left-to-Right
    stripped = text.strip("()., -/")
    return bool(stripped) and all(ch.isdigit() or ch in "()., -/" for ch in text)

def _ordered_word_text(word: dict, chars: list[dict], rtl: bool) -> str:
    # Expanded vertical tolerance (8.0) to capture floating Tashkeel/Tanween diacritics
    matched = [
        c for c in chars
        if word["x0"] - 1.0 <= c["x0"] <= word["x1"] + 1.0 and abs(c["top"] - word["top"]) < 8.0
    ]
    
    # Sort Left-to-Right first (Standard for English and Numbers)
    matched.sort(key=lambda c: c["x0"])
    ltr_text = "".join(c["text"] for c in matched)
    
    # If it is RTL (Arabic) AND it is NOT a number, reverse the sort for Arabic letters
    if rtl and not _is_weak_digit_run(ltr_text):
        matched.sort(key=lambda c: c["x0"], reverse=True)
        return "".join(c["text"] for c in matched)
        
    return ltr_text

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
    pending_title_for = None
    current_main_topic = ""  
    
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
                ar_str = araby.strip_tatweel(ar_str)
                ar_str = fix_arabic_output_artifacts(ar_str)
                
                if re.search(r'BOOK|SECTION|الفصل|الباب|باب|أحكام عامة', f"{en_str} {ar_str}", re.IGNORECASE):
                    seen_heading = True
                if not seen_heading:
                    continue
                
                en_match = re.search(r'^\bArticle\s+([0-9]+)\b', en_str, re.IGNORECASE)
                ar_match = re.search(r'^مادة\s*\(?\s*([٠-٩0-9]+)\s*\)?', ar_str)
                is_article_start = bool(en_match or ar_match)
                
                # 1. Intercept Repealed Ranges
# 1. Intercept Repealed Ranges
                repeal_match = re.search(r'Articles?\s+(\d+)(?:\s*(?:-|to|and|–)\s*(\d+))?\s+.*?\brepealed\b', en_str, re.IGNORECASE)
                if repeal_match:
                    start_rep = int(repeal_match.group(1))
                    end_rep = int(repeal_match.group(2)) if repeal_match.group(2) else start_rep
                    for r_num in range(start_rep, end_rep + 1):
                        articles_dict[r_num] = {
                            "article_number": r_num,
                            "book": hierarchy.get("book", ""),
                            "chapter": hierarchy.get("chapter", ""),
                            "section": hierarchy.get("section", ""),
                            "topic": hierarchy.get("topic", ""),
                            "text_ar": ar_str,
                            "text_en": en_str,
                            "is_repealed": True,
                            "source_page": pno,
                            "citation": f"Egyptian Civil Code, Article {r_num}"
                        }
                    current_art = end_rep
                    pending_title_for = None
                    continue

                # 2. Intercept Hierarchy Headers (Book, Chapter, Section, Topic, Sub-topic)
                header_type = None
                if re.search(r'^BOOK\s+[IVX]+', en_str, re.IGNORECASE):
                    header_type = "book"
                elif re.search(r'^Chapter\s+[IVX]+', en_str, re.IGNORECASE) or re.search(r'^الباب\s+', ar_str):
                    header_type = "chapter"
                elif re.search(r'^Section\s+[IVX]+', en_str, re.IGNORECASE) or re.search(r'^الفصل\s+', ar_str):
                    header_type = "section"
                elif re.search(r'^\d{1,2}\.\s+[A-Z]', en_str):
                    # شرط صارم للعنوان: يجب أن يبدأ برقم مكون من منزلين بحد أقصى متبوعاً بنقطة ثم بحرف كبير، 
                    # وألا يكون مجرد بند داخل مادة (مثل بنود الأشخاص الاعتبارية 3 و 4)
                    if not re.search(r'(Wakfs|corporations|state|persons)', en_str, re.IGNORECASE):
                        text_part = re.sub(r'^\d+[\.\-]\s*', '', en_str)
                        words = re.findall(r'[a-zA-Z]+', text_part)
                        minor_words = {"and", "of", "the", "in", "on", "to", "as", "by", "for", "with", "a", "an", "or"}
                        major_words = [w for w in words if w.lower() not in minor_words]
                        
                        if major_words and all(w[0].isupper() for w in major_words):
                            header_type = "topic"
                        
                elif re.search(r'^(?:Conflicts of law|تنازع القوانين|Associations|الجمعيات)', f"{en_str} {ar_str}", re.IGNORECASE):
                    header_type = "sub_topic"
                    
                if header_type:
                    val = f"{en_str} | {ar_str}".strip()
                    if header_type == "book":
                        hierarchy["book"] = val
                        hierarchy["chapter"] = hierarchy["section"] = hierarchy["topic"] = ""
                        current_main_topic = ""
                    elif header_type == "chapter":
                        hierarchy["chapter"] = val
                        hierarchy["section"] = hierarchy["topic"] = ""
                        current_main_topic = ""
                    elif header_type == "section":
                        hierarchy["section"] = val
                        hierarchy["topic"] = ""
                        current_main_topic = ""
                    elif header_type == "topic":
                        hierarchy["topic"] = val
                        current_main_topic = val  
                    elif header_type == "sub_topic":
                        if current_main_topic:
                            hierarchy["topic"] = f"{current_main_topic} - {val}"
                        else:
                            hierarchy["topic"] = val
                    
                    if header_type != "sub_topic":
                        pending_title_for = header_type
                    continue
                    
                # 3. Intercept Secondary Title Lines
                if pending_title_for and not is_article_start and not repeal_match:
                    current_val = hierarchy[pending_title_for]
                    separator = " - " if current_val else ""
                    appended_val = f"{current_val}{separator}{en_str} | {ar_str}".strip()
                    hierarchy[pending_title_for] = appended_val
                    
                    if pending_title_for == "topic":
                        current_main_topic = appended_val
                        
                    pending_title_for = None
                    continue

                # 4. Handle Article Starts
                if is_article_start:
                    pending_title_for = None
                    art_num = int(en_match.group(1)) if en_match else ar_to_int(ar_match.group(1))
                    current_art = art_num
                    en_body = en_str[en_match.end():].strip() if en_match else ""
                    ar_body = re.sub(r'^(?:مادة\s*\(?\s*[٠-٩0-9]+\s*\)?\s*)', '', ar_str).strip() if ar_match else ar_str
                    if art_num not in articles_dict:
                        articles_dict[art_num] = {
                            "article_number": art_num,
                            "book": hierarchy.get("book", ""),
                            "chapter": hierarchy.get("chapter", ""),
                            "section": hierarchy.get("section", ""),
                            "topic": hierarchy.get("topic", ""),
                            "text_ar": ar_body,
                            "text_en": en_body,
                            "is_repealed": False,
                            "source_page": pno,
                            "citation": f"Egyptian Civil Code, Article {art_num}"
                        }
                    continue
                    
                # 5. Regular text append
                if current_art is not None and current_art in articles_dict:
                    if en_str:
                        articles_dict[current_art]["text_en"] = (articles_dict[current_art]["text_en"] + " " + en_str).strip()
                    if ar_str:
                        articles_dict[current_art]["text_ar"] = (articles_dict[current_art]["text_ar"] + " " + ar_str).strip()

    articles_list = sorted(articles_dict.values(), key=lambda x: x["article_number"])
    articles_list = fix_bilingual_alignment_drift(articles_list)
    
    # 6. Final cleaning and fallback repealed check
    for a in articles_list:
        if not a["is_repealed"]:
            a["is_repealed"] = bool(re.search(r'\brepealed\b', a["text_en"].lower()) or "ملغاة" in a["text_ar"] or "ملغى" in a["text_ar"])
        a["text_en"] = ' '.join(a["text_en"].split())
        a["text_ar"] = ' '.join(a["text_ar"].split())

    return articles_list

#Fix the drift problem between article 1021 and 1022
def fix_bilingual_alignment_drift(articles_list: list[dict]) -> list[dict]:
    """Fix bilingual column split drift by programmatically splitting and creating missing articles."""
    articles_map = {a["article_number"]: a for a in articles_list}
    
    # Specific fix for Article 1021/1022 drift observed in the document
    if 1021 in articles_map and 1022 not in articles_map:
        art_1021 = articles_map[1021]
        
        # Check if Article 1021's English text contains the text intended for 1022
        split_marker = "cost of the necessary works"
        if split_marker in art_1021["text_en"]:
            parts_en = art_1021["text_en"].split(split_marker, 1)
            
            # Refine Article 1021 English text
            art_1021["text_en"] = parts_en[0].strip()
            
            # Construct Article 1022 programmatically inheriting metadata from 1021
            art_1022_text_en = (split_marker + parts_en[1]).strip()
            
            # Arabic text for 1022 corresponds to the second paragraph of Article 1021
# Arabic text split using a flexible regex to catch (2) in any digit format
            ar_parts = re.split(r'(?=\(\s*[٢2]\s*\))', art_1021["text_ar"])
            
            if len(ar_parts) >= 2:
                art_1021_ar = ar_parts[0].strip()
                art_1022_ar = "".join(ar_parts[1:]).strip()
            else:
                art_1021_ar = art_1021["text_ar"]
                art_1022_ar = ""

            art_1021["text_ar"] = art_1021_ar
            new_art_1022 = {
                "article_number": 1022,
                "book": art_1021["book"],
                "chapter": art_1021["chapter"],
                "section": art_1021["section"],
                "topic": art_1021["topic"],
                "text_ar": art_1022_ar,
                "text_en": art_1022_text_en,  # Fixed variable name here
                "is_repealed": False,
                "source_page": art_1021["source_page"],
                "citation": "Egyptian Civil Code, Article 1022"
            }
            
            # Insert 1022 into the list
            articles_list.append(new_art_1022)
            articles_list.sort(key=lambda x: x["article_number"])
            
    return articles_list