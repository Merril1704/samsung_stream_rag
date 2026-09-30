"""Tests for Streaming Live RAG: Speculative Pre-fetching and Zero-Latency Handoff."""
from __future__ import annotations

import json
import pytest
from controller.rule_based import RuleBasedController
from controller.types import SessionState
from decomposer.decomposer import Decomposer
from fusion.fusion_pipeline import FusionPipeline
from fusion.contradiction import ContradictionScreen
from grounding.verifier import GroundingVerifier
from session.generator import AnswerGenerator
from session.orchestrator import StreamRAGOrchestrator
from tests.test_session_4a import StubLLM


def test_speculative_prefetch_on_intermediate_chunk(indexed_dev_corpus):
    """When an intermediate chunk (is_final=False) reaches stable entities,
    orchestrator performs speculative pre-fetch without invoking generator LLM or committing to ledger.
    """
    stub_responses = {
        "Candidate Chunks:": json.dumps({"claims": [{"claim": "Cancellation policy applies.", "chunk_id": "DOC_04_§1"}]}),
        "Claim: ": json.dumps({"supported": True, "reason": "verified"}),
    }
    stub = StubLLM(responses=stub_responses)

    controller = RuleBasedController(wait_threshold=4)
    decomposer = Decomposer(llm_client=stub)
    fusion = FusionPipeline(contradiction_screen=ContradictionScreen(llm_client=stub))
    generator = AnswerGenerator(stub)
    verifier = GroundingVerifier(stub)

    orchestrator = StreamRAGOrchestrator(
        controller=controller,
        index=indexed_dev_corpus,
        fusion=fusion,
        generator=generator,
        verifier=verifier,
        decomposer=decomposer,
    )

    state = SessionState(session_id="test_stream_speculative")

    # Chunk 1: Unstable intent -> WAIT
    res1 = orchestrator.step(state, "I need to cancel", timestamp_s=0.0, is_final=False)
    assert res1.action == "WAIT"
    assert res1.turn_result is None
    assert len(state.prefetched_candidate_ids) == 0
    assert state.ledger is None

    # Chunk 2: Still incomplete -> WAIT
    res2 = orchestrator.step(state, "a corporate event", timestamp_s=0.5, is_final=False)
    assert res2.action == "WAIT"
    assert res2.turn_result is None
    assert len(state.prefetched_candidate_ids) == 0
    assert state.ledger is None

    # Chunk 3: Intent stable (>= 4 entities) -> PREFETCH
    res3 = orchestrator.step(state, "booked for next week", timestamp_s=1.0, is_final=False)
    assert res3.action == "PREFETCH"
    assert res3.turn_result is None  # Generator NOT called
    assert state.ledger is None      # Ledger NOT mutated
    assert len(state.prefetched_candidate_ids) == 10  # Pre-fetched into memory!
    assert state.prefetched_query is not None
    assert "cancel a corporate event booked for next week" in state.prefetched_query
    assert state.prefetched_latency_s >= 0.0
    # No LLM generator calls during pre-fetch
    assert len([call for call in stub.calls if "Candidate Chunks:" in call[1]]) == 0


def test_end_of_utterance_reuses_prefetched_candidates(indexed_dev_corpus):
    """When the final speech chunk (is_final=True) arrives, pre-warmed candidate
    chunks are reused directly for answer generation and committed to ledger.
    """
    stub_responses = {
        "Candidate Chunks:": json.dumps({"claims": [{"claim": "Cancellation policy applies under standard terms.", "chunk_id": "DOC_04_§1"}]}),
        "Claim: ": json.dumps({"supported": True, "reason": "verified"}),
        "Passage A:": json.dumps({"is_contradiction": False, "reason": "distinct"}),
    }
    stub = StubLLM(responses=stub_responses)

    controller = RuleBasedController(wait_threshold=4)
    decomposer = Decomposer(llm_client=stub)
    fusion = FusionPipeline(contradiction_screen=ContradictionScreen(llm_client=stub))
    generator = AnswerGenerator(stub)
    verifier = GroundingVerifier(stub)

    orchestrator = StreamRAGOrchestrator(
        controller=controller,
        index=indexed_dev_corpus,
        fusion=fusion,
        generator=generator,
        verifier=verifier,
        decomposer=decomposer,
    )

    state = SessionState(session_id="test_stream_handoff")

    # Intermediate chunk -> triggers pre-fetch
    res_intermediate = orchestrator.step(
        state,
        "I need to cancel a corporate event booked for next week",
        timestamp_s=1.0,
        is_final=False,
    )
    assert res_intermediate.action == "PREFETCH"
    prefetched_ids = list(state.prefetched_candidate_ids)
    assert len(prefetched_ids) > 0

    # Final chunk arrives -> is_final=True
    res_final = orchestrator.step(
        state,
        "under standard terms",
        timestamp_s=2.0,
        is_final=True,
    )
    assert res_final.action == "RETRIEVE"
    assert res_final.turn_result is not None
    assert state.ledger is not None
    assert "entry_1" in state.ledger.entries
    assert len(state.ledger.entries["entry_1"].claims) >= 1


def test_default_is_final_preserves_batch_orchestrator_behavior(indexed_dev_corpus):
    """Calling step() without is_final defaults to is_final=True, ensuring
    100% backward compatibility with all existing tests and benchmark runners.
    """
    stub_responses = {
        "Candidate Chunks:": json.dumps({"claims": [{"claim": "Full refund available.", "chunk_id": "DOC_04_§1"}]}),
        "Claim: ": json.dumps({"supported": True, "reason": "verified"}),
    }
    stub = StubLLM(responses=stub_responses)

    controller = RuleBasedController()
    decomposer = Decomposer(llm_client=stub)
    fusion = FusionPipeline(contradiction_screen=ContradictionScreen(llm_client=stub))
    generator = AnswerGenerator(stub)
    verifier = GroundingVerifier(stub)

    orchestrator = StreamRAGOrchestrator(
        controller=controller,
        index=indexed_dev_corpus,
        fusion=fusion,
        generator=generator,
        verifier=verifier,
        decomposer=decomposer,
    )

    state = SessionState(session_id="test_batch_compat")
    # Calling step() with default arguments (no is_final specified)
    res = orchestrator.step(state, "I need to plan a customer workshop in Pune for 30 people", timestamp_s=1.0)
    assert res.action == "RETRIEVE"
    assert res.turn_result is not None
    assert state.ledger is not None
