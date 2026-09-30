from __future__ import annotations
import logging
from controller.model_based import _extract_first_json_object
from controller.llm_wrapper import ContextBudgetExceeded
from grounding.types import ClaimEntry
from retrieval.types import EvidenceChunk

logger = logging.getLogger(__name__)

GENERATOR_SYSTEM_PROMPT = """You generate factual claims answering a topic based ONLY on provided document chunks.

SYNTACTIC CONSTRAINTS (follow strictly):
1. Output compact JSON only: {"claims": [{"claim": "...", "chunk_id": "..."}]}
2. ONE sentence per claim.
3. ONE fact per claim.
4. Do not join two facts with "and", "or", or ";".
5. Copy numbers, dates, and terms verbatim from the chunk.
6. For each claim, chunk_id MUST be the exact chunk that directly contains and proves that specific rule, threshold, or fact (e.g. if a chunk has approval thresholds, cite that chunk, not a general policy overview chunk).
7. State only what the chunks say. Do not extrapolate, generalize, or invent claims.
8. If any candidate chunk has '(contradiction_flag: True, conflicts with: CHUNK_ID)', you MUST generate at least one claim citing that chunk AND at least one separate claim citing the conflicting CHUNK_ID so that both conflicting perspectives are surfaced.
9. If the chunks do not answer the topic, return {"claims": []}."""


def format_candidate_chunks(chunks: list[EvidenceChunk]) -> str:
    parts = []
    for c in chunks:
        header = f"- chunk_id: {c.chunk_id}"
        if getattr(c, "contradiction_flag", False) and getattr(c, "contradicts_chunk_id", None):
            header += f" (contradiction_flag: True, conflicts with: {c.contradicts_chunk_id})"
        parts.append(f"{header}\n  text: {c.text}")
    return "\n".join(parts)


class AnswerGenerator:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.last_error: str | None = None

    def generate(
        self,
        topic: str,
        chunks: list[EvidenceChunk],
        detail: str | None = None,
    ) -> list[ClaimEntry]:
        self.last_error = None
        if not chunks:
            return []

        chunks_text = format_candidate_chunks(chunks)
        user_prompt = f"Topic: {topic}\n"
        if detail:
            user_prompt += f"Detail / Focus: {detail}\n"
        user_prompt += f"\nCandidate Chunks:\n{chunks_text}\n\nRespond with compact JSON only:"

        try:
            raw = self.llm_client.complete(GENERATOR_SYSTEM_PROMPT, user_prompt, max_tokens=1000)
        except ContextBudgetExceeded:
            raise
        except Exception as e:
            logger.warning("Answer generation failed: %s", e, exc_info=True)
            self.last_error = f"{type(e).__name__}: {e}"
            return []

        try:
            data = _extract_first_json_object(raw)
            if data and isinstance(data, dict):
                claims_data = data.get("claims")
                if isinstance(claims_data, list):
                    entries: list[ClaimEntry] = []
                    for item in claims_data:
                        if isinstance(item, dict) and "claim" in item and "chunk_id" in item:
                            claim_str = str(item["claim"]).strip()
                            cid_str = str(item["chunk_id"]).strip()
                            if claim_str:
                                entries.append(ClaimEntry(claim=claim_str, chunk_id=cid_str))
                    return entries
            self.last_error = "generator output parse failure"
            return []
        except Exception as e:
            logger.warning("Answer generation JSON parse error: %s", e, exc_info=True)
            self.last_error = f"Parse error: {e}"
            return []
