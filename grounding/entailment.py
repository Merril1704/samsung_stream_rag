"""EntailmentChecker: checks whether a passage strictly entails an atomic claim.

Uses the validated system prompt from scratch/test_entailment_prompt.py,
including the strict language-strength rules and 3 worked examples.
"""
import logging
from controller.llm_wrapper import ContextBudgetExceeded
from decomposer.llm_verifier import _extract_first_json_object

logger = logging.getLogger(__name__)

ENTAILMENT_SYSTEM_PROMPT = """You are checking whether a specific claim is actually supported by a document passage.

You will be given a CLAIM and a PASSAGE. Decide:
- Is the claim fully supported by the passage — same facts, same numbers, same subject/entity?
- A claim is NOT supported if the passage discusses a different subject/entity than the claim
  (even if it mentions similar-sounding facts about something else).
- A claim is NOT supported if the passage doesn't contain the specific fact/number claimed.
- A claim is NOT supported if it uses stronger, more specific, or more absolute language than the passage — e.g., a passage saying 'standard' does not support a claim of 'full' or 'complete'; a passage saying 'available' does not support a claim of 'guaranteed'; a passage listing specific items does not support a claim of 'no restrictions' or 'all inclusive' unless it says so explicitly.

Respond with ONLY compact JSON, no prose, no markdown fences, no repeated output:
{"supported": true|false, "reason": "..."}

Example:
Claim: "Cancellations made more than 30 days before the event receive a full refund."
Passage: "Bookings cancelled more than 30 days before the event date receive a full refund of any deposit paid."
{"supported": true, "reason": "passage states the same 30-day window and full refund outcome"}

Example:
Claim: "Venue A holds up to 40 seated guests."
Passage: "Venue B — Grand Hall Conference Suite. Capacity: 30 seated theatre-style."
{"supported": false, "reason": "passage describes Venue B, not Venue A — wrong subject despite being about venue capacity"}

Example:
Claim: "The venue provides full AV support."
Passage: "Includes projector, whiteboard, and standard AV."
{"supported": false, "reason": "passage says 'standard AV', a claim of 'full AV support' is stronger/more absolute than what the passage states"}"""


class EntailmentChecker:
    def __init__(self, llm_client):
        self.llm = llm_client

    def check(self, claim: str, passage: str) -> tuple[bool, str]:
        """
        Returns (supported: bool, reason: str).
        Fail-safe: on parse failure, return (False, "entailment check parse failure").
        Never default to True on failure — an unverifiable claim is an unsupported claim.
        """
        user_prompt = f'Claim: "{claim.strip()}"\nPassage: "{passage.strip()}"'
        try:
            raw = self.llm.complete(ENTAILMENT_SYSTEM_PROMPT, user_prompt)
        except ContextBudgetExceeded:
            raise
        except Exception as e:
            logger.warning("Entailment LLM completion failed: %s", e, exc_info=True)
            return False, f"entailment client error: {type(e).__name__}"

        try:
            data = _extract_first_json_object(raw)
            if data and isinstance(data, dict):
                if "supported" in data:
                    return bool(data["supported"]), str(data.get("reason", ""))
                return False, "missing 'supported' key in response"
        except Exception as e:
            logger.warning("Entailment JSON parsing failed: %s", e, exc_info=True)

        return False, "entailment check parse failure"
