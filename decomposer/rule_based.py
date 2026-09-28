"""Rule-based Multi-Intent Decomposer.

Dependency-free by design (no spaCy/NER model), consistent with the style of
controller/rule_based.py.  The four steps follow the spec precisely:
  A – connector detection + strategy selection
  B – entity-protection capitalization heuristic
  C – standalone-completeness check per candidate part
  D – high-confidence vs. ambiguous confidence assignment
"""
import re
from .types import DecompositionResult

# ── connector phrases (domain-agnostic) ─────────────────────────────────────
CONNECTORS = [" and ", " also ", " plus ", " as well as ", " in addition "]

# ── stopwords (mirrors controller/rule_based.py STOPWORDS) ──────────────────
STOPWORDS = {
    "i", "need", "to", "a", "an", "the", "and", "also", "plus", "for",
    "in", "of", "on", "with", "my", "me", "please", "can", "you", "is",
    "are", "was", "were", "about", "your", "it", "that", "this", "as",
    "well", "be", "do", "addition",
}

# leading connector noise removed when assessing standalone-completeness.
# Note: "in" is excluded from connector noise so that prepositional fragments
# (e.g. "in neural network training") have their preposition preserved for detection.
LEADING_CONNECTOR_NOISE = {
    "and", "also", "plus", "as", "well", "addition",
}

# prepositions that signal a prepositional-fragment start (over-fragmentation)
PREPOSITIONS = {"in", "at", "for", "with", "on", "of", "from", "by", "into"}


# ── helpers ──────────────────────────────────────────────────────────────────

def _content_tokens(text: str) -> list[str]:
    """Lower-cased content tokens after removing stopwords (order preserved)."""
    raw = re.findall(r"[a-z0-9']+", text.lower())
    return [t for t in raw if t not in STOPWORDS and len(t) > 1]


def _is_standalone(part: str) -> tuple[bool, str]:
    """
    Return (is_standalone, reason) for *part*.

    Failure modes:
      • The candidate part starts with a preposition (e.g. "in", "at", "for", "with", "on").
      • After stripping leading connector noise and stopwords, fewer than 2 content words remain.
    """
    stripped = part.strip()
    raw_tokens = re.findall(r"[a-z0-9']+", stripped.lower())

    # Advance past leading connector noise (e.g. "also", "and")
    start = 0
    while start < len(raw_tokens) and raw_tokens[start] in LEADING_CONNECTOR_NOISE:
        start += 1

    remaining = raw_tokens[start:]
    if not remaining:
        return False, "empty fragment after stripping noise"

    # Preposition check on the initial token of the candidate fragment
    if remaining[0] in PREPOSITIONS:
        return False, f"prepositional fragment starting with '{remaining[0]}'"

    content = _content_tokens(stripped)
    if len(content) < 2:
        return False, f"insufficient content words ({len(content)} < 2)"

    return True, ""


# ── split utilities ───────────────────────────────────────────────────────────

def _split_on_connector(text: str, connector: str) -> list[str]:
    """Case-insensitive split on a single connector phrase."""
    pattern = re.compile(re.escape(connector), re.IGNORECASE)
    return [p for p in pattern.split(text) if p.strip()]


def _split_on_all_connectors(text: str) -> list[str]:
    """Split simultaneously on all connector phrases (combined strategy)."""
    pattern = re.compile(
        "|".join(re.escape(c) for c in CONNECTORS), re.IGNORECASE
    )
    return [p for p in pattern.split(text) if p.strip()]


def _connectors_present(text: str) -> list[str]:
    """Return connector phrases found in *text* (in order of first occurrence)."""
    found = []
    lower = text.lower()
    for c in CONNECTORS:
        if c in lower:
            found.append(c)
    return found


# ── entity-protection heuristic ──────────────────────────────────────────────

def _is_capitalized_word(word: str) -> bool:
    return bool(word) and word[0].isupper() and word.isalpha()


def _is_entity_bridge(text: str, connector: str) -> tuple[bool, str]:
    """
    Return (True, span_description) if *connector* sits between two
    capitalized tokens, signalling it may be a multi-word proper noun/entity.
    Sentence-initial position is excluded to avoid false positives.
    """
    lower_conn = connector.strip().lower()
    pattern = re.compile(re.escape(connector), re.IGNORECASE)
    for m in pattern.finditer(text):
        before = text[:m.start()].rstrip()
        after = text[m.end():].lstrip()
        before_tokens = before.split()
        after_tokens = after.split()
        if not before_tokens or not after_tokens:
            continue
        before_word = before_tokens[-1]
        after_word = after_tokens[0]

        # Sentence-initial guard: if before_word is the very first word in the
        # utterance, skip (its capitalisation is trivial / expected).
        utterance_start_words = text.lstrip().split()
        is_sentence_initial = utterance_start_words and before_word == utterance_start_words[0]

        if (
            _is_capitalized_word(before_word)
            and _is_capitalized_word(after_word)
            and not is_sentence_initial
        ):
            span = f"{before_word} {connector.strip()} {after_word}"
            return True, span
    return False, ""


# ── main class ────────────────────────────────────────────────────────────────

class RuleBasedDecomposer:
    """
    Implements Steps A–D from the Stage 2 spec for the rule-based decomposition
    layer.  Returns a high-confidence DecompositionResult directly, or one
    with method="ambiguous" when the LLM verifier should weigh in.
    """

    def decide(self, utterance: str) -> DecompositionResult:
        present = _connectors_present(utterance)

        # ── Step A ── early exit if no connector present ─────────────────────
        if not present:
            return DecompositionResult(
                is_compound=False,
                sub_queries=[utterance],
                method="single",
                reason="no connector phrase found",
            )

        # ── Enumerate both strategies ─────────────────────────────────────────
        # Strategy 1: best single-connector split (most parts)
        best_s1_parts: list[str] = []
        best_s1_connector: str = ""
        for conn in present:
            parts = _split_on_connector(utterance, conn)
            if len(parts) > len(best_s1_parts):
                best_s1_parts = parts
                best_s1_connector = conn

        # Strategy 2: combined split on all connectors
        s2_parts = _split_on_all_connectors(utterance)

        # Pick whichever yields more parts; tie → S1
        if len(s2_parts) > len(best_s1_parts):
            candidate_parts = s2_parts
            strategy_label = "combined"
            active_connectors = present
        else:
            candidate_parts = best_s1_parts
            strategy_label = "single"
            active_connectors = [best_s1_connector] if best_s1_connector else present

        rejected_splits: list[str] = []

        # ── Step B ── entity-protection per split point ───────────────────────
        # For each connector in the active strategy, check if it bridges
        # two capitalized tokens.  If so, merge those parts and log the span.
        merged_parts = list(candidate_parts)
        entity_protected = False

        for conn in active_connectors:
            suspect, span = _is_entity_bridge(utterance, conn)
            if suspect:
                entity_protected = True
                rejected_splits.append(
                    f"entity-protection: '{span}' — connector '{conn.strip()}' "
                    f"likely inside a multi-word proper noun/entity"
                )
                # Rebuild merged_parts: join the segments that bracket this connector
                pattern = re.compile(re.escape(conn), re.IGNORECASE)
                # Find which segment pairs are affected and merge them
                new_parts: list[str] = []
                text_remaining = utterance
                segs = [p.strip() for p in pattern.split(text_remaining) if p.strip()]
                # Re-identify which consecutive pair is the suspect bridge
                # by checking capitalization in the rebuilt segments
                i = 0
                while i < len(segs):
                    seg = segs[i]
                    if (
                        i + 1 < len(segs)
                        and _is_capitalized_word(seg.split()[-1] if seg.split() else "")
                        and _is_capitalized_word(segs[i + 1].split()[0] if segs[i + 1].split() else "")
                    ):
                        merged = seg + conn + segs[i + 1]
                        new_parts.append(merged.strip())
                        i += 2
                    else:
                        new_parts.append(seg.strip())
                        i += 1
                merged_parts = new_parts

        if len(merged_parts) < 2:
            return DecompositionResult(
                is_compound=False,
                sub_queries=[utterance],
                method="single",
                rejected_splits=rejected_splits,
                reason="no valid split points remain after entity protection",
            )

        # ── Step C ── standalone-completeness check ───────────────────────────
        failing_parts: list[tuple[str, str]] = []
        for p in merged_parts:
            is_ok, reason = _is_standalone(p)
            if not is_ok:
                failing_parts.append((p, reason))

        if failing_parts:
            for fp, reason in failing_parts:
                rejected_splits.append(
                    f"standalone-check failed: '{fp.strip()}' — {reason}"
                )
            # The split fails; fall through to is_compound=False
            return DecompositionResult(
                is_compound=False,
                sub_queries=[utterance],
                method="single",
                rejected_splits=rejected_splits,
                reason="split rejected: one or more parts failed standalone-completeness check",
            )

        # ── Step D ── confidence assignment ───────────────────────────────────
        # High-confidence criteria:
        #   - exactly one connector phrase used
        #   - that connector appears exactly once in the utterance
        #   - no entity-protection merges occurred
        #   - each part has ≥3 content words
        exactly_one_connector = len(active_connectors) == 1
        conn_for_check = active_connectors[0] if active_connectors else ""
        appears_once = utterance.lower().count(conn_for_check.lower()) == 1 if conn_for_check else False
        all_parts_rich = all(len(_content_tokens(p)) >= 3 for p in merged_parts)

        if (
            exactly_one_connector
            and appears_once
            and not entity_protected
            and all_parts_rich
        ):
            return DecompositionResult(
                is_compound=True,
                sub_queries=[p.strip() for p in merged_parts],
                method="rule_split",
                rejected_splits=rejected_splits,
                reason="clean single-connector split, both parts standalone",
            )

        # Borderline / ambiguous → let LLMVerifier decide
        return DecompositionResult(
            is_compound=True,  # tentative; verifier may overturn
            sub_queries=[p.strip() for p in merged_parts],
            method="ambiguous",
            rejected_splits=rejected_splits,
            reason=(
                "ambiguous split: borderline content word count, "
                "multiple connector types, or entity-protection merge occurred"
            ),
        )
