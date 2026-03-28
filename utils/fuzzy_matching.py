from rapidfuzz import process, fuzz
import easyocr
import numpy as np
from typing import List, Tuple, Dict
import textdistance
import re

reader = easyocr.Reader(['en'])  # language list


def combine_easyocr_text_ordered(results, sep=" "):
    # Sort by top-left corner (y, then x)
    sorted_results = sorted(results, key=lambda x: (x[0][1][1], x[0][0][0]))
    return sep.join([text for (_, text, _) in sorted_results])

def get_ocr_image(image):
    ocr_result = reader.readtext(np.array(image))
    ocr_result = combine_easyocr_text_ordered(ocr_result).strip().lower()
    return ocr_result


def smart_fuzzy_brand_match(ocr_text, brand_list, base_threshold=70):
    clean_ocr = ocr_text.strip()
    is_short = len(clean_ocr) < 4

    threshold = base_threshold + 15 if is_short else base_threshold

    match, score, _ = process.extractOne(clean_ocr, brand_list, scorer=fuzz.WRatio)
    
    # if is_short and clean_ocr.lower() not in [b.lower() for b in brand_list]:
    #     return "UNKNOWN", 0

    return (match, score) if score >= threshold else ("UNKNOWN", score)

def robust_fuzzy_match(ocr_text: str, db_text: str, base_threshold=75, min_text_len=3) -> str:
    ocr_clean = ocr_text.strip()
    if len(ocr_clean) < min_text_len:
        return "UNKNOWN"

    score = fuzz.WRatio(ocr_clean, db_text)
    partial_score = fuzz.partial_ratio(ocr_clean, db_text)

    if len(ocr_clean) <= 3 and score < base_threshold + 10:
        return "UNKNOWN"

    if score >= base_threshold and partial_score >= 80:
        return db_text

    return "UNKNOWN"


def smart_fuzzy_brand_match_batch(
    texts: List[str],
    brand_list: List[str],
    base_threshold: int = 70
) -> List[Tuple[str, int]]:
    """
    Quickly match each OCR text against known brands using a vectorized C++ backend.
    Applies a higher threshold for very short texts to avoid spurious matches.

    Args:
        texts: List of OCR-extracted strings.
        brand_list: List of known brand names (already lowercased).
        base_threshold: Base fuzzy-match threshold (0-100). Short strings use base+15.

    Returns:
        List of (best_match, score) tuples, one per input text.
        If no match meets the per-text threshold, returns ("UNKNOWN", best_score_or_0).
    """
    # Preprocess queries
    queries = [t.strip().lower() for t in texts]
    # Determine per-query thresholds"
    thresholds = [base_threshold + 15 if len(q) < 4 else base_threshold for q in queries]

    # Bulk compute pairwise scores >= base_threshold
    raw_matches = process.cdist(
        queries,
        brand_list,
        scorer=fuzz.WRatio,
        processor=lambda s: s,
        score_cutoff=base_threshold
    )

    best_match_score = np.max(raw_matches, axis=1)
    best_match_index = np.argmax(raw_matches, axis=1)
    is_matched = best_match_score > thresholds
    
    results = []
    for i, is_match in enumerate(is_matched):
        if is_match:
            score = best_match_score[i]
            brand = brand_list[best_match_index[i]]
            results.append((brand, score))
        else:
            results.append(("UNKNOWN", best_match_score[i]))
    return results


# --- Batched fuzzy matcher ---
def robust_fuzzy_match_batch(
    ocr_texts: List[str],
    db_texts: List[str],
    base_threshold:    int = 75,
    min_text_len:      int = 3,
    partial_threshold: int = 80,
    workers:           int = -1
) -> List[str]:
    """
    Batchwise fuzzy matching of each OCR string against a single DB string.
    Returns either the DB string or "UNKNOWN".
    """
    # Clean and length‐filter
    ocr_clean = [t.strip() for t in ocr_texts]
    lengths   = np.array([len(t) for t in ocr_clean])
    results   = ["UNKNOWN"] * len(ocr_clean)

    # Only match those long enough
    mask = lengths >= min_text_len
    if not mask.any():
        return results

    queries = [ocr_clean[i] for i in np.nonzero(mask)[0]]

    # Compute WRatio & partial_ratio in parallel
    full_scores    = process.cdist(queries, db_texts,    scorer=fuzz.WRatio,        workers=workers)
    partial_scores = process.cdist(queries, db_texts,    scorer=fuzz.partial_ratio, workers=workers)

    # For each query, pick best db index
    best_idx     = full_scores.argmax(axis=1)
    best_full    = full_scores[np.arange(full_scores.shape[0]), best_idx]
    best_partial = partial_scores[np.arange(full_scores.shape[0]), best_idx]

    # Apply thresholds
    accept = (best_full >= base_threshold) & (best_partial >= partial_threshold)
    short  = np.array([len(q) <= 3 for q in queries])
    accept |= (short & (best_full >= base_threshold + 10))

    valid_indices = np.nonzero(mask)[0]
    for qi, orig_i in enumerate(valid_indices):
        if accept[qi]:
            results[orig_i] = db_texts[best_idx[qi]]
    return results


def normalize_digits_letters(s: str) -> str:
    """
    Insert spaces between digits and letters in either direction:
      "2hd" -> "2 hd",  "hd2" -> "hd 2"
    """
    # (?<=\d)(?=[A-Za-z])  = position between digit and letter
    # (?<=[A-Za-z])(?=\d)  = position between letter and digit
    return re.sub(r'(?<=\d)(?=[A-Za-z])|(?<=[A-Za-z])(?=\d)', ' ', s)

def smart_score(a: str, b: str) -> float:
    # print(a, b)
    """
    Combined similarity:
    1) token_set_ratio for multi‑word vs multi‑word
    2) Jaro–Winkler for short single tokens
    3) partial_ratio otherwise
    Returns 0–100.
    """
    a = normalize_digits_letters(a.lower().strip())
    b = normalize_digits_letters(b.lower().strip())

    # if b in ['rai', 'omega', 'yakult']:
    #     print("a, b: ", (a, b))

    # 1) Multi‑word case: ignore extra tokens
    if len(a.split()) > 1 and len(b.split()) > 1:
        # return fuzz.token_set_ratio(a, b)
        return fuzz.WRatio(a, b)

    # 2) Very short single tokens: use Jaro–Winkler
    if max(len(a), len(b)) <= 6:
        return textdistance.jaro_winkler.normalized_similarity(a, b) * 100

    # 3) Fallback: allow substring matches
    return fuzz.partial_ratio(a, b)