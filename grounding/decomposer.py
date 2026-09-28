"""ClaimDecomposer: splits compound answers into atomic factual claims.

One LLM call per answer. Prevents multi-fact compound sentences from slipping
past entailment checking when only partially supported.
"""
import logging
from controller.llm_wrapper import ContextBudgetExceeded
from decomposer.llm_verifier import _extract_first_json_object

logger = logging.getLogger(__name__)

CLAIM_DECOMPOSER_SYSTEM_PROMPT = """You are splitting an answer into atomic factual claims.

Goal: Only split sentences that combine multiple independent facts joined by conjunctions like "and" or separate sentences.

Strict guidelines:
- If a sentence has only one main verb and no conjunctions (e.g. "Venue A allows external catering with no restrictions."), it is ALREADY atomic. Output it as a single claim.
- Never split prepositional phrases or modifiers (such as "with no restrictions") into a separate claim.
- Never rephrase the text. Always keep the exact original words.

Respond with ONLY compact JSON:
{"claims": ["claim 1", ...]}

Example 1:
Answer: "The venue is available with 5 business days notice and includes full AV support."
{"claims": ["The venue is available with 5 business days notice.", "The venue includes full AV support."]}

Example 2:
Answer: "Venue A allows external catering with no restrictions."
{"claims": ["Venue A allows external catering with no restrictions."]}

Example 3:
Answer: "Cancellations made more than 30 days before the event receive a full refund."
{"claims": ["Cancellations made more than 30 days before the event receive a full refund."]}"""


class ClaimDecomposer:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.last_fallback: bool = False

    def decompose(self, answer_text: str) -> list[str]:
        """
        Splits answer_text into a list of atomic claims — each claim must
        assert exactly ONE fact. A sentence bundling multiple facts
        must be split into separate claims, one per fact.
        Returns the list of claim strings. Fail-safe: if the LLM call fails
        to parse, fall back to treating the whole answer_text as a single
        claim (never crash, never silently drop the answer).
        """
        self.last_fallback = False
        stripped = answer_text.strip()
        if not stripped:
            return []

        user_prompt = f'Answer: "{stripped}"'
        try:
            raw = self.llm.complete(CLAIM_DECOMPOSER_SYSTEM_PROMPT, user_prompt)
        except ContextBudgetExceeded:
            raise
        except Exception as e:
            logger.warning("Claim decomposition LLM completion failed: %s", e, exc_info=True)
            self.last_fallback = True
            return [stripped]

        try:
            data = _extract_first_json_object(raw)
            if data and isinstance(data, dict):
                claims = data.get("claims")
                if isinstance(claims, list) and all(isinstance(c, str) for c in claims) and claims:
                    valid_claims = [c.strip() for c in claims if c.strip()]
                    if valid_claims:
                        return valid_claims
        except Exception as e:
            logger.warning("Claim decomposition JSON parsing failed: %s", e, exc_info=True)

        # Fail-safe: fall back to treating the whole answer_text as a single claim
        self.last_fallback = True
        return [stripped]
