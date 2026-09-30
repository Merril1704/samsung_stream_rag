"""Contradiction screening for retrieved evidence chunks.

Step 1 — Deterministic rule-based pre-filter (no LLM call):
  Filters out pairs from the same doc_id or pairs lacking substantive topical intersection.
  Selects pairs displaying concrete conflict indicators:
    - Conflicting dimensioned quantities (time, money, percentages, capacity, dates)
    - Grounded asymmetric negation contrast
    - Explicit policy conflict / override markers
Step 2 — LLM confirmation for suspect pairs passing Step 1:
  Calls prompt and parses JSON using the existing brace-depth parser.
  On confirmation, sets contradiction_flag=True and cross-references contradicts_chunk_id
  on both chunks without removing either chunk from the result.
"""
import re
import time
from typing import Any, Optional
from .types import EvidenceChunk
from decomposer.llm_verifier import _extract_first_json_object

STOPWORDS = {
    "i", "need", "to", "a", "an", "the", "and", "also", "plus", "for",
    "in", "of", "on", "with", "my", "me", "please", "can", "you", "is",
    "are", "was", "were", "about", "your", "it", "that", "this", "as",
    "well", "be", "do", "will", "would", "should", "could", "may", "might",
    "must", "shall", "has", "have", "had", "having", "been", "being",
    "policy", "section", "doc", "per", "under", "all", "any", "each",
    "such", "other", "than", "more", "less", "before", "after", "time",
    "within", "made", "make", "at", "by", "from", "into", "through",
    "during", "above", "below", "out", "off", "over", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "both", "few", "some", "no", "nor", "not",
    "only", "own", "same", "so", "too", "very", "s", "t",
    "just", "don", "now",
}

NEGATION_PATTERNS = [
    r"\bnot\b", r"\bno\b", r"\bcannot\b", r"\bnever\b", r"\bn't\b",
    r"\bnone\b", r"\bneither\b", r"\bnon-refundable\b", r"\bineligible\b",
    r"\bprohibited\b", r"\bdisallowed\b", r"\bunauthorized\b", r"\bexempt\b",
]

DIMENSION_PATTERNS = {
    "days": r"\b(\d+)\s*(?:-|–)?\s*(?:days?|business\s+days?)\b",
    "hours": r"\b(\d+)\s*(?:-|–)?\s*hours?\b",
    "weeks": r"\b(\d+)\s*(?:-|–)?\s*weeks?\b",
    "money": r"(?:₹|rs\.?|inr|\$)\s*(\d+(?:,\d+)*)|\b(\d+(?:,\d+)*)\s*(?:rupees|inr|usd|dollars)\b",
    "percent": r"\b(\d+)\s*%",
    "capacity": r"\b(\d+)\s*(?:guests?|people|attendees?|seats?|persons?)\b",
    "dates": r"\b((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?|\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b",
}

CONFLICT_MARKERS = [
    r"\bdiffers?\s+from\b",
    r"\bsupersedes?\b",
    r"\boverrides?\b",
    r"\bnotwithstanding\b",
    r"\bin\s+contrast\b",
    r"\bdiscrepancy\b",
    r"\bexceptions?\s+to\b",
    r"\bconflicts?\b",
]

CONTRADICTION_SYSTEM_PROMPT = """\
You are checking whether two retrieved passages genuinely contradict each other on the same specific claim.

CRITICAL RULES:
1. Two passages are a GENUINE contradiction ONLY if they state mutually exclusive, conflicting rules for the EXACT SAME condition, situation, and scope.
2. Different rules that apply to DIFFERENT scopes, categories, regions, or conditions (e.g., domestic travel vs. international travel, standard vs. emergency, employee vs. contractor) are NOT contradictions. They are distinct rules for distinct situations.

Respond with ONLY compact JSON, no prose, no markdown fences, no repeated output:
{"is_contradiction": true|false, "reason": "..."}

Example:
Passage A: "Cancellations made within 48 hours of the event are non-refundable."
Passage B: "Cancellations made within 48 hours are eligible for a 50% refund."
{"is_contradiction": true, "reason": "different refund outcomes stated for the same 48-hour cancellation window"}

Example:
Passage A: "Domestic travel requires receipts for expenses exceeding ₹500."
Passage B: "International travel requires original currency receipts and conversion forms."
{"is_contradiction": false, "reason": "different policy scopes (domestic vs. international travel), not a conflict"}

Example:
Passage A: "The venue holds up to 200 guests."
Passage B: "Catering is available for groups of up to 150."
{"is_contradiction": false, "reason": "different subjects (venue capacity vs. catering group size), not a genuine conflict"}\
"""


def _content_tokens(text: str) -> set[str]:
    raw = re.findall(r"[a-z0-9']+", text.lower())
    return {t for t in raw if t not in STOPWORDS and len(t) > 2 and not t.isdigit()}


def extract_dimension_values(text: str) -> dict[str, set[str]]:
    """Extract normalized values keyed by measurement dimension."""
    t = text.lower()
    res: dict[str, set[str]] = {}
    for dim, pat in DIMENSION_PATTERNS.items():
        matches: set[str] = set()
        for m in re.finditer(pat, t):
            val = next((g for g in m.groups() if g is not None), None)
            if val:
                matches.add(val.replace(",", "").strip())
        if matches:
            res[dim] = matches
    return res


def has_conflicting_dimensions(text_a: str, text_b: str) -> bool:
    """Return True if both chunks share a dimension but specify conflicting values."""
    dims_a = extract_dimension_values(text_a)
    dims_b = extract_dimension_values(text_b)
    shared_dims = set(dims_a.keys()) & set(dims_b.keys())
    for d in shared_dims:
        if dims_a[d] != dims_b[d]:
            return True
    return False


def has_negation_contrast(text_a: str, text_b: str) -> bool:
    """Return True if one chunk contains negation markers and the other does not."""
    t_a = text_a.lower()
    t_b = text_b.lower()
    neg_a = any(re.search(pat, t_a) for pat in NEGATION_PATTERNS)
    neg_b = any(re.search(pat, t_b) for pat in NEGATION_PATTERNS)
    return neg_a != neg_b


def has_explicit_conflict_marker(text_a: str, text_b: str) -> bool:
    """Return True if either chunk contains explicit divergence/override language."""
    for pat in CONFLICT_MARKERS:
        if re.search(pat, text_a.lower()) or re.search(pat, text_b.lower()):
            return True
    return False


def _has_differing_numbers(text_a: str, text_b: str) -> bool:
    """Legacy helper maintained for backward compatibility."""
    return has_conflicting_dimensions(text_a, text_b)


def _has_asymmetric_negation(text_a: str, text_b: str) -> bool:
    """Legacy helper maintained for backward compatibility."""
    return has_negation_contrast(text_a, text_b)


def is_suspect_pair(chunk_a: EvidenceChunk, chunk_b: EvidenceChunk) -> bool:
    """High-precision Step 1 rule screen:
    1. Different doc_id.
    2. Substantive topical overlap (>=3 content tokens or Jaccard >= 0.12).
    3. Concrete conflict indicator:
       - Conflicting dimensioned quantities (days, hours, money, percent, capacity, dates)
       - Explicit policy override/divergence marker
       - Negation contrast on grounded topic (>=4 tokens or Jaccard >= 0.15)
    """
    if chunk_a.doc_id == chunk_b.doc_id:
        return False

    tokens_a = _content_tokens(chunk_a.text)
    tokens_b = _content_tokens(chunk_b.text)
    shared = tokens_a & tokens_b

    jaccard = len(shared) / max(len(tokens_a | tokens_b), 1)
    if len(shared) < 3 and jaccard < 0.12:
        return False

    if has_conflicting_dimensions(chunk_a.text, chunk_b.text):
        return True

    if has_explicit_conflict_marker(chunk_a.text, chunk_b.text):
        return True

    if has_negation_contrast(chunk_a.text, chunk_b.text) and (len(shared) >= 4 or jaccard >= 0.15):
        return True

    return False


# Backward-compatibility alias
_is_suspect_pair = is_suspect_pair


class ContradictionScreen:
    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.last_screen_stats: dict[str, Any] = {
            "total_candidate_pairs": 0,
            "prefiltered_pairs": 0,
            "llm_calls": 0,
            "prefilter_rate": 0.0,
            "latency_s": 0.0,
        }

    def screen(self, sub_query: str, chunks: list[EvidenceChunk]) -> list[tuple[str, str]]:
        """
        Returns list of (chunk_id_a, chunk_id_b) pairs confirmed as genuine contradictions.
        Runs deterministic Step 1 rule pre-filter first; calls LLM only for suspect pairs.
        If no llm_client provided, returns Step-1 suspect pairs as-is.
        On confirmation, sets contradiction_flag=True and cross-references contradicts_chunk_id
        on both chunks without removing either chunk.
        """
        t_start = time.perf_counter()
        total_candidate_pairs = 0
        prefiltered_pairs = 0
        llm_calls = 0

        suspect_pairs: list[tuple[EvidenceChunk, EvidenceChunk]] = []

        # Find all suspect pairs via Step 1 pre-filter
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                total_candidate_pairs += 1
                if is_suspect_pair(chunks[i], chunks[j]):
                    suspect_pairs.append((chunks[i], chunks[j]))
                else:
                    prefiltered_pairs += 1

        confirmed_pairs: list[tuple[str, str]] = []

        for chunk_a, chunk_b in suspect_pairs:
            is_contradiction = False

            if self.llm is None:
                # Lower-confidence flag without LLM: surface suspect pair per 'never silently resolve'
                is_contradiction = True
            else:
                llm_calls += 1
                user_prompt = (
                    f'Sub-query: "{sub_query}"\n'
                    f'Passage A: "{chunk_a.text}"\n'
                    f'Passage B: "{chunk_b.text}"'
                )
                raw = self.llm.complete(CONTRADICTION_SYSTEM_PROMPT, user_prompt)
                data = _extract_first_json_object(raw)
                if data is not None:
                    if "is_contradiction" in data:
                        is_contradiction = bool(data.get("is_contradiction", False))
                    else:
                        # Fallback for offline mock client
                        is_contradiction = True
                else:
                    # Parse failure: keep unconfirmed suspect pair
                    is_contradiction = True

            if is_contradiction:
                chunk_a.contradiction_flag = True
                chunk_a.contradicts_chunk_id = chunk_b.chunk_id
                chunk_b.contradiction_flag = True
                chunk_b.contradicts_chunk_id = chunk_a.chunk_id
                confirmed_pairs.append((chunk_a.chunk_id, chunk_b.chunk_id))

        elapsed_s = time.perf_counter() - t_start
        rate = (prefiltered_pairs / total_candidate_pairs) if total_candidate_pairs > 0 else 0.0
        self.last_screen_stats = {
            "total_candidate_pairs": total_candidate_pairs,
            "prefiltered_pairs": prefiltered_pairs,
            "llm_calls": llm_calls,
            "prefilter_rate": rate,
            "latency_s": elapsed_s,
        }

        return confirmed_pairs

