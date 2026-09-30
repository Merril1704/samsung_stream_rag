from __future__ import annotations

import pytest
from controller.rule_based import RuleBasedController
from controller.types import SessionState
from decomposer.decomposer import Decomposer
from eval.scenarios import get_scenario_by_id
from fusion.fusion_pipeline import FusionPipeline
from fusion.contradiction import ContradictionScreen
from grounding.verifier import GroundingVerifier
from session.generator import AnswerGenerator
from session.orchestrator import StreamRAGOrchestrator
from tests.test_session_4a import StubLLM


def test_eval_refine_01_two_turn_pipeline_refinement(indexed_dev_corpus):
    """Verify eval_refine_01 execution path:
    - Turn 1 creates a valid ledger entry with version=1.
    - Turn 2 routes via answer_refinement(), not answer_new_topic().
    - trace.path == 'REFINEMENT'.
    - version transitions from 1 to 2.
    - Rendered output preserves previous claims under Baseline and adds new constraint.
    """
    scen = get_scenario_by_id("eval_refine_01")
    t1 = scen.turns[0]
    t2 = scen.turns[1]

    # Deterministic stub responses reflecting the refined generator prompt output
    stub_responses = {
        # Contradiction screen (pairs from candidate chunks)
        "Passage A:": '{"is_contradiction": false, "reason": "distinct policy scopes"}',
        # Verifier
        "Claim: ": '{"supported": true, "reason": "explicitly supported by passage"}',
        # Turn 2 Generator (has Detail / Focus:)
        "Detail / Focus:": '{"claims": [{"claim": "Expense reports exceeding ₹25,000 require Senior Director approval in addition to manager sign-off.", "chunk_id": "DOC_09_§2"}]}',
        # Turn 1 Generator (candidate chunks without Detail / Focus)
        "Candidate Chunks:": '{"claims": [{"claim": "Expense reports under ₹25,000 require only direct manager approval.", "chunk_id": "DOC_09_§1"}]}',
    }
    stub = StubLLM(responses=stub_responses, default='{"claims": []}')

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
        llm_counter=None,
    )

    state = SessionState(session_id=scen.scenario_id)

    # Turn 1 Execution
    res1 = orchestrator.step(state=state, chunk=t1.user_input, timestamp_s=t1.timestamp_s)
    assert res1.action == "RETRIEVE"
    assert res1.trace.path == "NEW_TOPIC"
    assert state.ledger is not None
    assert "entry_1" in state.ledger.entries
    entry_v1 = state.ledger.entries["entry_1"]
    assert entry_v1.version == 1
    assert len(entry_v1.claims) >= 1
    assert "under ₹25,000 require only direct manager approval" in entry_v1.claims[0].claim
    assert "DOC_09_§1" in entry_v1.evidence

    # Turn 2 Execution
    res2 = orchestrator.step(state=state, chunk=t2.user_input, timestamp_s=t2.timestamp_s)
    assert res2.action == "RETRIEVE"
    assert res2.trace.path == "REFINEMENT"  # Must NOT fall back to NEW_TOPIC
    entry_v2 = state.ledger.entries["entry_1"]
    assert entry_v2.version == 2  # Version incremented from 1 to 2
    assert len(entry_v2.claims) == 2  # Baseline claim + refinement claim preserved
    assert res2.trace.version_transition is not None
    assert res2.trace.version_transition.old_version == 1
    assert res2.trace.version_transition.new_version == 2

    # Verify rendered text preserves baseline and includes refinement
    rendered = res2.text
    assert "Baseline" in rendered
    assert "Refinement" in rendered
    assert "under ₹25,000" in rendered
    assert "exceeding ₹25,000" in rendered


def test_eval_refine_02_two_turn_pipeline_refinement(indexed_dev_corpus):
    """Verify eval_refine_02 execution path:
    - Turn 1 creates a valid ledger entry with standard cancellation windows.
    - Turn 2 routes via answer_refinement(), not answer_new_topic().
    - trace.path == 'REFINEMENT'.
    - version transitions from 1 to 2.
    - Rendered output preserves standard cancellation policy and adds force majeure exception.
    """
    scen = get_scenario_by_id("eval_refine_02")
    t1 = scen.turns[0]
    t2 = scen.turns[1]

    stub_responses = {
        # Contradiction screen
        "Passage A:": '{"is_contradiction": false, "reason": "no conflict"}',
        # Verifier
        "Claim: ": '{"supported": true, "reason": "supported by DOC_04"}',
        # Turn 2 Generator (has Detail / Focus:)
        "Detail / Focus:": '{"claims": [{"claim": "Cancellations resulting from documented force majeure events are eligible for full refund regardless of notice period.", "chunk_id": "DOC_04_§3"}]}',
        # Turn 1 Generator
        "Candidate Chunks:": '{"claims": [{"claim": "Bookings cancelled more than 30 days before event receive a full refund.", "chunk_id": "DOC_04_§1"}]}',
    }
    stub = StubLLM(responses=stub_responses, default='{"claims": []}')

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
        llm_counter=None,
    )

    state = SessionState(session_id=scen.scenario_id)

    # Turn 1
    res1 = orchestrator.step(state=state, chunk=t1.user_input, timestamp_s=t1.timestamp_s)
    assert res1.action == "RETRIEVE"
    assert res1.trace.path == "NEW_TOPIC"
    assert "entry_1" in state.ledger.entries
    assert state.ledger.entries["entry_1"].version == 1

    # Turn 2
    res2 = orchestrator.step(state=state, chunk=t2.user_input, timestamp_s=t2.timestamp_s)
    assert res2.action == "RETRIEVE"
    assert res2.trace.path == "REFINEMENT"
    assert state.ledger.entries["entry_1"].version == 2
    assert len(state.ledger.entries["entry_1"].claims) == 2

    rendered = res2.text
    assert "Baseline" in rendered
    assert "Refinement" in rendered
    assert "30 days" in rendered
    assert "force majeure" in rendered
