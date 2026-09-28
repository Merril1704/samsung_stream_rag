"""Decomposer orchestrator — Stage 2 of the streaming RAG pipeline.

Combines RuleBasedDecomposer (authoritative for high-confidence cases) with
LLMVerifier (consulted only for ambiguous splits), mirroring the hybrid
escalation pattern from Stage 1's HybridController.

Usage:
    from decomposer.decomposer import Decomposer
    from controller.llm_factory import get_llm_client

    decomposer = Decomposer(llm_client=get_llm_client())
    result = decomposer.decompose("book a room and tell me the parking options")
"""
from .types import DecompositionResult
from .rule_based import RuleBasedDecomposer
from .llm_verifier import LLMVerifier


class Decomposer:
    """
    Orchestrates rule-based decomposition with optional LLM verification.

    High-confidence rule results are returned directly.
    Ambiguous results are escalated to LLMVerifier if an llm_client is
    available; otherwise fail-safe to is_compound=False.
    """

    def __init__(self, llm_client=None):
        self.rule_decomposer = RuleBasedDecomposer()
        self.llm_verifier = LLMVerifier(llm_client) if llm_client else None

    def decompose(self, utterance: str) -> DecompositionResult:
        result = self.rule_decomposer.decide(utterance)

        # High-confidence paths — return immediately without LLM call:
        #   "single"      → no connector at all or standalone-check failure
        #   "rule_split"  → clean high-confidence split
        if result.method in ("single", "rule_split"):
            return result

        # Ambiguous path (method == "ambiguous")
        candidates = result.sub_queries

        if self.llm_verifier is not None:
            verified = self.llm_verifier.verify(utterance, candidates)
            return verified

        # No LLM client — fail-safe: do not guess
        return DecompositionResult(
            is_compound=False,
            sub_queries=[utterance],
            method="rule_split+llm_verified",
            rejected_splits=candidates,
            reason="ambiguous split, no LLM client available — defaulting to no split",
        )
