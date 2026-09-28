"""Rule-based Retrieval Controller.

Dependency-free by design (no spaCy/NER model) for G1 reproducibility on a
clean machine. This trades some precision for zero setup cost — if G2/G3
accuracy proves insufficient against held-out data, the entity/completeness
signal functions below are the swap point for a real NER/dependency parser.
"""
import re
import hashlib
from .types import SessionState, RetrievalDecision

# Config-style pattern lists kept separate from decision logic so they can be
# tuned without touching control flow. This is still a heuristic limitation —
# it won't generalize to unseen phrasing the way the model-based controller
# can, which is precisely what the required ablation is meant to surface.
SUPPRESSION_PATTERNS = [
    r"\brepeat\b", r"\bshorter\b", r"\bbullet(s)?\b", r"\brephrase\b",
    r"\breformat\b", r"\bsummari[sz]e (that|it|your (last )?answer)\b",
    r"\bagain\b", r"\bin (a )?table\b", r"\bmore concise\b",
]
ANAPHORA_PATTERNS = [r"\bthat\b", r"\bit\b", r"\byour (last )?answer\b", r"\bthis\b"]
SUPPRESSION_NOISE_WORDS = {
    "repeat", "bullet", "bullets", "point", "points", "answer", "last",
    "your", "please", "short", "shorter", "that", "it", "again", "table",
}

STOPWORDS = {
    "i", "need", "to", "a", "the", "and", "for", "in", "of", "on", "with",
    "my", "me", "please", "can", "you", "is", "are", "was", "were", "about",
    "your", "it", "that", "this",
}
REQUEST_VERBS = {"need", "want", "plan", "tell", "find", "get", "show", "give", "book"}
PRONOUNS = {"i", "we", "you", "he", "she", "they"}


def _raw_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def _content_tokens(text: str) -> set[str]:
    return {t for t in _raw_tokens(text) if t not in STOPWORDS and len(t) > 1}


def _entity_signal(text: str) -> set[str]:
    """Proxy for stable retrievable content: capitalized words + numbers + content tokens."""
    caps = set(re.findall(r"\b[A-Z][a-zA-Z]+\b", text))
    nums = set(re.findall(r"\b\d+\b", text))
    return {c.lower() for c in caps} | nums | _content_tokens(text)


def _looks_complete(text: str) -> bool:
    request_verb_present = bool(_raw_tokens(text) & REQUEST_VERBS)
    enough_tokens = len(_content_tokens(text)) >= 4
    return request_verb_present and enough_tokens


def _is_topic_conjunction(chunk_text: str) -> bool:
    """True if 'and' in the newly-arrived chunk joins two topics rather than
    two clauses (e.g. 'cancellation policy AND catering options' vs 'Pune AND I need')."""
    for m in re.finditer(r"\band\s+(\w+)", chunk_text.lower()):
        following = m.group(1)
        if following not in PRONOUNS and following not in REQUEST_VERBS:
            return True
    return False


class RuleBasedController:
    mode = "rule_based"

    def __init__(self, wait_threshold: int = 4):
        self.wait_threshold = wait_threshold

    def decide(self, state: SessionState, new_chunk: str, timestamp_s: float) -> RetrievalDecision:
        state.append_chunk(new_chunk)
        full = state.transcript_so_far
        lower = full.lower()

        # --- Suppression check first: cheapest, highest precedence ---
        has_suppression_cue = any(re.search(p, lower) for p in SUPPRESSION_PATTERNS)
        has_anaphor = any(re.search(p, lower) for p in ANAPHORA_PATTERNS)
        new_since_last_answer = True
        if state.last_answer_topic:
            topic_tokens = _content_tokens(state.last_answer_topic)
            current_tokens = _content_tokens(new_chunk) - SUPPRESSION_NOISE_WORDS
            diff = current_tokens - topic_tokens
            # Require >=2 genuinely new content words, not just any nonzero
            # difference — a single incidental token shouldn't override an
            # otherwise clear suppression cue (found via Scenario 3 testing).
            new_since_last_answer = len(diff) >= 2

        if has_suppression_cue and has_anaphor and not new_since_last_answer:
            return RetrievalDecision("SUPPRESS", "suppression", 0.85,
                                      "presentation-only cue, no new entities vs. prior answer")

        entities = _entity_signal(full)
        entity_hash = hashlib.sha1(" ".join(sorted(entities)).encode()).hexdigest()[:10]

        if entity_hash == state.last_retrieved_entity_hash:
            return RetrievalDecision("WAIT", "none", 0.4,
                                      "no new stable entities since last retrieval (debounce)")

        if not _looks_complete(full) or len(entities) < self.wait_threshold:
            return RetrievalDecision("WAIT", "none", 0.5,
                                      f"intent unstable: {len(entities)} stable entities "
                                      f"(threshold {self.wait_threshold})")

        new_chunk_entities = _entity_signal(new_chunk)
        multi_intent = _is_topic_conjunction(new_chunk) and len(new_chunk_entities) >= 2
        state.last_retrieved_entity_hash = entity_hash

        if multi_intent:
            return RetrievalDecision("RETRIEVE", "multi_intent", 0.75,
                                      f"topic-joining conjunction + {len(new_chunk_entities)} "
                                      f"new entities in latest chunk", possible_multi_intent=True)

        return RetrievalDecision("RETRIEVE", "provisional", 0.65,
                                  f"stable entities reached ({len(entities)} >= {self.wait_threshold})")
