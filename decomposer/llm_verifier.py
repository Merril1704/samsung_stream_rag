"""LLM Verifier for the Multi-Intent Decomposer.

Only called for ambiguous splits from RuleBasedDecomposer — the rule-based
layer stays authoritative for clear cases.  Reuses:
  - the LLMClient protocol from controller/model_based.py (no new factory)
  - the _extract_first_json_object brace-depth parser pattern from
    controller/model_based.py to handle markdown fences and repeated output.

Fail-safe: any parse failure → is_compound=False (same philosophy as Stage 1's
WAIT default — never fabricate a split on uncertainty).
"""
import json
from typing import Optional
from .types import DecompositionResult

VERIFIER_SYSTEM_PROMPT = """\
You are verifying whether a user utterance should be split into independent sub-queries for parallel search.

You will be given the original utterance and a CANDIDATE split into parts. Decide:
- Is each candidate part a genuinely independent, standalone request that could be answered separately from the others?
- Or does the split break up what is really a single request (e.g., splitting off a sentence fragment, or separating parts of one named entity)?

Respond with ONLY compact JSON, no prose, no markdown code fences, no repeated output:
{"confirm_split": true|false, "reason": "..."}

Example:
Utterance: "cancel my booking and tell me the refund policy"
Candidate parts: ["cancel my booking", "tell me the refund policy"]
{"confirm_split": true, "reason": "two independent actions: a cancellation request and a separate policy question"}

Example:
Utterance: "what is attention in transformers"
Candidate parts: ["what is attention", "in transformers"]
{"confirm_split": false, "reason": "the second part is a sentence fragment, not an independent request; this is one question"}\
"""


def _extract_first_json_object(raw: str) -> Optional[dict]:
    """Finds and parses the first balanced top-level JSON object in raw text.
    Handles markdown fences, leading/trailing prose, and duplicate blocks.
    Mirrors the implementation in controller/model_based.py exactly.
    """
    start = raw.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(raw)):
            char = raw[i]
            if escape:
                escape = False
                continue
            if in_string:
                if char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = raw[start: i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        break
                    break
        start = raw.find("{", start + 1)
    return None


class LLMVerifier:
    """Second-opinion LLM check, only for ambiguous rule-based results."""

    def __init__(self, llm_client):
        self.llm = llm_client

    def verify(self, utterance: str, candidate_sub_queries: list[str]) -> DecompositionResult:
        parts_repr = str(candidate_sub_queries)
        user_prompt = (
            f'Utterance: "{utterance}"\n'
            f"Candidate parts: {parts_repr}"
        )
        raw = self.llm.complete(VERIFIER_SYSTEM_PROMPT, user_prompt)
        data = _extract_first_json_object(raw)

        if data is None:
            # Fail-safe: parse failure → do not split
            return DecompositionResult(
                is_compound=False,
                sub_queries=[utterance],
                method="rule_split+llm_verified",
                rejected_splits=list(candidate_sub_queries),
                reason="LLM response parse failure — defaulting to no split",
            )

        if "confirm_split" not in data:
            # Fallback for offline MockLLMClient (built for Stage 1 controller):
            confirmed = (
                len(candidate_sub_queries) >= 2
                and all(len(p.split()) >= 2 for p in candidate_sub_queries)
            )
            reason = data.get("reason", "offline mock verification")
        else:
            confirmed = bool(data.get("confirm_split", False))
            reason = data.get("reason", "")

        if confirmed:
            return DecompositionResult(
                is_compound=True,
                sub_queries=list(candidate_sub_queries),
                method="rule_split+llm_verified",
                reason=reason,
            )
        else:
            return DecompositionResult(
                is_compound=False,
                sub_queries=[utterance],
                method="rule_split+llm_verified",
                rejected_splits=list(candidate_sub_queries),
                reason=reason,
            )
