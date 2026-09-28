"""Contradiction screening for retrieved evidence chunks.

Step 1 — Rule-based screen (no LLM call) over pairs from different doc_ids:
  Checks for >=2 shared content tokens AND (differing numbers OR asymmetric negation).
Step 2 — LLM confirmation for suspect pairs from Step 1:
  Calls prompt and parses JSON using the existing brace-depth parser.
  On confirmation, sets contradiction_flag=True and cross-references contradicts_chunk_id
  on both chunks without removing either chunk from the result.
"""
import re
from typing import Optional
from .types import EvidenceChunk
from decomposer.llm_verifier import _extract_first_json_object

STOPWORDS = {
    "i", "need", "to", "a", "an", "the", "and", "also", "plus", "for",
    "in", "of", "on", "with", "my", "me", "please", "can", "you", "is",
    "are", "was", "were", "about", "your", "it", "that", "this", "as",
    "well", "be", "do",
}

NEGATION_PATTERNS = [r"\bnot\b", r"\bno\b", r"\bcannot\b", r"\bnever\b", r"\bn't\b"]

CONTRADICTION_SYSTEM_PROMPT = """\
You are checking whether two retrieved passages genuinely contradict each other on the same specific claim.

Respond with ONLY compact JSON, no prose, no markdown fences, no repeated output:
{"is_contradiction": true|false, "reason": "..."}

Example:
Passage A: "Cancellations made within 48 hours of the event are non-refundable."
Passage B: "Cancellations made within 48 hours are eligible for a 50% refund."
{"is_contradiction": true, "reason": "different refund outcomes stated for the same 48-hour cancellation window"}

Example:
Passage A: "The venue holds up to 200 guests."
Passage B: "Catering is available for groups of up to 150."
{"is_contradiction": false, "reason": "different subjects (venue capacity vs. catering group size), not a genuine conflict"}\
"""


def _content_tokens(text: str) -> set[str]:
    raw = re.findall(r"[a-z0-9']+", text.lower())
    return {t for t in raw if t not in STOPWORDS and len(t) > 1}


def _has_differing_numbers(text_a: str, text_b: str) -> bool:
    nums_a = set(re.findall(r"\b\d+\b", text_a))
    nums_b = set(re.findall(r"\b\d+\b", text_b))
    # Differing numbers: each has numbers and their sets differ
    if nums_a and nums_b and nums_a != nums_b:
        return True
    return False


def _has_asymmetric_negation(text_a: str, text_b: str) -> bool:
    has_neg_a = any(re.search(pat, text_a.lower()) for pat in NEGATION_PATTERNS)
    has_neg_b = any(re.search(pat, text_b.lower()) for pat in NEGATION_PATTERNS)
    return has_neg_a != has_neg_b


def _is_suspect_pair(chunk_a: EvidenceChunk, chunk_b: EvidenceChunk) -> bool:
    """Step 1 rule screen: different doc_id, >=2 shared content tokens, differing numbers or negations."""
    if chunk_a.doc_id == chunk_b.doc_id:
        return False

    tokens_a = _content_tokens(chunk_a.text)
    tokens_b = _content_tokens(chunk_b.text)
    shared = tokens_a & tokens_b

    if len(shared) < 2:
        return False

    diff_nums = _has_differing_numbers(chunk_a.text, chunk_b.text)
    asym_neg = _has_asymmetric_negation(chunk_a.text, chunk_b.text)

    return diff_nums or asym_neg


class ContradictionScreen:
    def __init__(self, llm_client=None):
        self.llm = llm_client

    def screen(self, sub_query: str, chunks: list[EvidenceChunk]) -> list[tuple[str, str]]:
        """
        Returns list of (chunk_id_a, chunk_id_b) pairs confirmed as genuine contradictions.
        Runs Step 1 rule screen first; calls LLM for pairs passing Step 1.
        If no llm_client provided, returns Step-1 suspect pairs as-is.
        On confirmation, sets contradiction_flag=True and cross-references contradicts_chunk_id
        on both chunks without removing either chunk.
        """
        suspect_pairs: list[tuple[EvidenceChunk, EvidenceChunk]] = []

        # Find all suspect pairs via Step 1 rule screen
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                if _is_suspect_pair(chunks[i], chunks[j]):
                    suspect_pairs.append((chunks[i], chunks[j]))

        confirmed_pairs: list[tuple[str, str]] = []

        chunk_map = {c.chunk_id: c for c in chunks}

        for chunk_a, chunk_b in suspect_pairs:
            is_contradiction = False

            if self.llm is None:
                # Lower-confidence flag without LLM: surface suspect pair per 'never silently resolve'
                is_contradiction = True
            else:
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

        return confirmed_pairs
