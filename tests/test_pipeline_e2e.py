from __future__ import annotations

import json
from pathlib import Path
import pytest

from controller.llm_wrapper import CountingLLMClient
from controller.rule_based import RuleBasedController
from controller.types import SessionState
from decomposer.decomposer import Decomposer
from fusion.contradiction import ContradictionScreen
from fusion.fusion_pipeline import FusionPipeline
from grounding.verifier import GroundingVerifier
from session.generator import AnswerGenerator
from session.orchestrator import StreamRAGOrchestrator
from session.pipeline import answer_suppress
from session.ledger import active_claims
from telemetry.schema import TurnTrace


class StubLLM:
    def __init__(self, responses: dict[str, str] | None = None, default: str = ""):
        self.responses = responses or {}
        self.default = default
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, *, max_tokens: int | None = None) -> str:
        self.calls.append((system, user))
        for key, resp in self.responses.items():
            if key in user:
                return resp
        return self.default


class RaisingLLM:
    def __init__(self, exc: Exception):
        self.exc = exc

    def complete(self, system: str, user: str, *, max_tokens: int | None = None) -> str:
        raise self.exc


# ---------------------------------------------------------------------------
# A. Multi-Intent NEW_TOPIC (Scenario A)
# ---------------------------------------------------------------------------

def test_orchestrator_multi_intent_scenario_a(indexed_dev_corpus):
    scenario_a_prompt = (
        "I need to plan a customer workshop in Pune for 30 people, "
        "and I need the cancellation policy and the catering options."
    )

    claims_payload = json.dumps({"claims": [
        {"claim": "Silver Lake Conference Center capacity is 40 seated in Pune.", "chunk_id": "DOC_02_§1"},
        {"claim": "Cancellations made 30 days prior receive full refund.", "chunk_id": "DOC_04_§1"},
        {"claim": "Standard in-house catering includes three menu packages.", "chunk_id": "DOC_05_§1"},
    ]})

    stub = StubLLM(responses={
        "Candidate parts:": json.dumps({"confirm_split": True, "reason": "three independent requests"}),
        "Candidate Chunks:": claims_payload,
        "Claim: ": json.dumps({"supported": True, "reason": "grounded in text"}),
        "Passage A:": json.dumps({"is_contradiction": False, "reason": "different aspects"}),
    })

    client = CountingLLMClient(stub)
    controller = RuleBasedController()
    decomposer = Decomposer(llm_client=client)
    fusion = FusionPipeline(contradiction_screen=ContradictionScreen(llm_client=client))
    generator = AnswerGenerator(client)
    verifier = GroundingVerifier(client)

    orchestrator = StreamRAGOrchestrator(
        controller=controller,
        index=indexed_dev_corpus,
        fusion=fusion,
        generator=generator,
        verifier=verifier,
        decomposer=decomposer,
        llm_counter=client,
    )

    state = SessionState(session_id="test_scenario_a")
    res = orchestrator.step(state, scenario_a_prompt, timestamp_s=2.5)

    assert res.action == "RETRIEVE"
    assert res.turn_result is not None
    assert res.turn_result.path == "NEW_TOPIC"
    assert res.turn_result.verification.all_verified is True

    # Validate retrieved and cited doc_ids match Scenario A from gold_manifest.md (DOC_02, DOC_04, DOC_05)
    entry = state.ledger.entries["entry_1"]
    active = active_claims(entry)
    cited_chunk_ids = {c.chunk_id for c in active}
    cited_doc_ids = {cid.rsplit("_", 1)[0] for cid in cited_chunk_ids}
    assert "DOC_02" in cited_doc_ids, "Scenario A must cite DOC_02 (venue)"
    assert "DOC_04" in cited_doc_ids, "Scenario A must cite DOC_04 (cancellation)"
    assert "DOC_05" in cited_doc_ids, "Scenario A must cite DOC_05 (catering)"

    # Telemetry assertions
    trace = res.trace
    assert trace is not None
    assert trace.path == "NEW_TOPIC"
    stage_names = [s.stage for s in trace.stage_events]
    assert "decomposition" in stage_names
    assert "retrieval" in stage_names
    assert "fusion" in stage_names
    assert "generator" in stage_names
    assert "verifier" in stage_names
    assert "render" in stage_names

    # Must have recorded controller decision and multiple sub-query retrieval events
    controller_event = trace.retrieval_events[0]
    assert controller_event.action == "RETRIEVE"
    assert controller_event.trigger == "multi_intent"

    sub_retrievals = [e for e in trace.retrieval_events if e.reason == "sub_query_retrieval"]
    assert len(sub_retrievals) == 3, f"Expected 3 sub-query retrieval events, got {len(sub_retrievals)}"
    for r in sub_retrievals:
        assert len(r.chunk_ids_returned) > 0


# ---------------------------------------------------------------------------
# B. Single-Intent NEW_TOPIC
# ---------------------------------------------------------------------------

def test_orchestrator_single_intent_new_topic(indexed_dev_corpus):
    single_intent_prompt = "I need to find the travel reimbursement rule for employees"

    claims_payload = json.dumps({"claims": [
        {"claim": "Domestic meals are capped at standard rates.", "chunk_id": "DOC_06_§1"},
    ]})

    stub = StubLLM(responses={
        "Candidate Chunks:": claims_payload,
        "Claim: ": json.dumps({"supported": True, "reason": "grounded"}),
        "Passage A:": json.dumps({"is_contradiction": False, "reason": "no conflict"}),
    })

    client = CountingLLMClient(stub)
    controller = RuleBasedController()
    decomposer = Decomposer(llm_client=client)
    fusion = FusionPipeline(contradiction_screen=ContradictionScreen(llm_client=client))
    generator = AnswerGenerator(client)
    verifier = GroundingVerifier(client)

    orchestrator = StreamRAGOrchestrator(
        controller=controller,
        index=indexed_dev_corpus,
        fusion=fusion,
        generator=generator,
        verifier=verifier,
        decomposer=decomposer,
        llm_counter=client,
    )

    state = SessionState(session_id="test_single_intent")
    res = orchestrator.step(state, single_intent_prompt, timestamp_s=1.0)

    assert res.action == "RETRIEVE"
    assert res.turn_result is not None
    assert res.turn_result.path == "NEW_TOPIC"

    # Only 1 actual retrieval event for single intent (+ 1 controller decision)
    retrievals = [e for e in res.trace.retrieval_events if e.trigger != "provisional"]
    assert len(retrievals) == 1
    assert "DOC_06_§1" in retrievals[0].chunk_ids_returned


# ---------------------------------------------------------------------------
# C. REFINEMENT
# ---------------------------------------------------------------------------

def test_orchestrator_refinement(indexed_dev_corpus):
    turn1_payload = json.dumps({"claims": [
        {"claim": "Domestic travel requires standard receipts.", "chunk_id": "DOC_06_§1"},
    ]})
    turn2_payload = json.dumps({"claims": [
        {"claim": "International travel requires currency conversion forms.", "chunk_id": "DOC_07_§1"},
    ]})

    stub = StubLLM(responses={
        "Detail / Focus:": turn2_payload,
        "Candidate Chunks:": turn1_payload,
        "Claim: ": json.dumps({"supported": True, "reason": "grounded"}),
        "Passage A:": json.dumps({"is_contradiction": False, "reason": "distinct scope"}),
    })

    client = CountingLLMClient(stub)
    controller = RuleBasedController()
    fusion = FusionPipeline(contradiction_screen=ContradictionScreen(llm_client=client))
    generator = AnswerGenerator(client)
    verifier = GroundingVerifier(client)

    orchestrator = StreamRAGOrchestrator(
        controller=controller,
        index=indexed_dev_corpus,
        fusion=fusion,
        generator=generator,
        verifier=verifier,
        llm_counter=client,
    )

    state = SessionState(session_id="test_refine_flow")

    # Turn 1: NEW_TOPIC
    res1 = orchestrator.step(state, "I need to find the travel reimbursement rule for employees", timestamp_s=1.0)
    assert res1.action == "RETRIEVE"
    assert res1.turn_result.path == "NEW_TOPIC"
    assert state.ledger.entries["entry_1"].version == 1

    # Turn 2: REFINEMENT
    res2 = orchestrator.step(state, "I also want to know international booking made after travel rules", timestamp_s=2.0)
    assert res2.action == "RETRIEVE"
    assert res2.turn_result.path == "REFINEMENT"
    assert state.ledger.entries["entry_1"].version == 2
    assert res2.trace.version_transition is not None
    assert res2.trace.version_transition.old_version == 1
    assert res2.trace.version_transition.new_version == 2


# ---------------------------------------------------------------------------
# D. SUPPRESS Path
# ---------------------------------------------------------------------------

def test_orchestrator_suppress_path(indexed_dev_corpus):
    turn1_payload = json.dumps({"claims": [
        {"claim": "Domestic meals are capped at standard rates.", "chunk_id": "DOC_06_§1"},
    ]})

    stub = StubLLM(responses={
        "Candidate Chunks:": turn1_payload,
        "Claim: ": json.dumps({"supported": True, "reason": "grounded"}),
        "Passage A:": json.dumps({"is_contradiction": False, "reason": "no conflict"}),
    })

    client = CountingLLMClient(stub)
    controller = RuleBasedController()
    fusion = FusionPipeline(contradiction_screen=ContradictionScreen(llm_client=client))
    generator = AnswerGenerator(client)
    verifier = GroundingVerifier(client)

    orchestrator = StreamRAGOrchestrator(
        controller=controller,
        index=indexed_dev_corpus,
        fusion=fusion,
        generator=generator,
        verifier=verifier,
        llm_counter=client,
    )

    state = SessionState(session_id="test_suppress_flow")
    orchestrator.step(state, "I need to find the travel reimbursement rule for employees", timestamp_s=1.0)
    assert "entry_1" in state.ledger.entries
    initial_version = state.ledger.entries["entry_1"].version

    initial_llm_calls = client.calls

    # Turn 2: SUPPRESS cue
    res2 = orchestrator.step(state, "Please repeat that in bullet points", timestamp_s=2.0)

    assert res2.action == "SUPPRESS"
    assert res2.turn_result is not None
    assert res2.turn_result.path == "SUPPRESS"
    assert res2.turn_result.llm_calls == 0
    # Zero new LLM calls made during suppression turn
    assert client.calls == initial_llm_calls
    # Zero ledger mutation
    assert state.ledger.entries["entry_1"].version == initial_version
    # Telemetry path is SUPPRESS
    assert res2.trace.path == "SUPPRESS"
    assert any(e.stage == "render" for e in res2.trace.stage_events)


# ---------------------------------------------------------------------------
# E. WAIT Path
# ---------------------------------------------------------------------------

def test_orchestrator_wait_path():
    controller = RuleBasedController(wait_threshold=4)
    orchestrator = StreamRAGOrchestrator(controller=controller)

    state = SessionState(session_id="test_wait_flow")
    res = orchestrator.step(state, "I need to plan a customer workshop in", timestamp_s=0.5)

    assert res.action == "WAIT"
    assert res.turn_result is None
    assert state.ledger is None

    # Telemetry
    trace = res.trace
    assert trace is not None
    assert trace.path == "WAIT"
    assert trace.duration_s > 0
    assert len(trace.retrieval_events) == 1
    assert trace.retrieval_events[0].action == "WAIT"


# ---------------------------------------------------------------------------
# F. Exception Safety
# ---------------------------------------------------------------------------

def test_orchestrator_exception_safety(indexed_dev_corpus):
    log_file = Path("scratch") / "test_exception_trace.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if log_file.exists():
        log_file.unlink()

    failing_llm = RaisingLLM(RuntimeError("Simulated LLM outage"))
    client = CountingLLMClient(failing_llm)
    controller = RuleBasedController()
    fusion = FusionPipeline(contradiction_screen=ContradictionScreen(llm_client=client))
    generator = AnswerGenerator(client)
    verifier = GroundingVerifier(client)

    orchestrator = StreamRAGOrchestrator(
        controller=controller,
        index=indexed_dev_corpus,
        fusion=fusion,
        generator=generator,
        verifier=verifier,
        llm_counter=client,
        log_path=log_file,
    )

    state = SessionState(session_id="test_exc_safety")
    trace = TurnTrace(turn_number=1, session_id="test_exc_safety")

    with pytest.raises(RuntimeError, match="Simulated LLM outage"):
        orchestrator.step(state, "I need to plan a customer workshop in Pune for 30 people", timestamp_s=1.0, trace=trace)

    # Trace must still be finalized and persisted
    assert trace.wall_end > 0
    assert trace.duration_s >= 0
    assert trace.path in ("NEW_TOPIC", "ERROR")

    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["path"] in ("NEW_TOPIC", "ERROR")
    assert data["duration_s"] >= 0

    if log_file.exists():
        log_file.unlink()


# ---------------------------------------------------------------------------
# G. answer_suppress() direct unit verification
# ---------------------------------------------------------------------------

def test_answer_suppress_unit():
    state = SessionState(session_id="test_suppress_unit")
    from session.ledger import commit_entry, AnswerLedger, LedgerClaim
    state.ledger = AnswerLedger()
    commit_entry(
        ledger=state.ledger,
        entry_id="entry_1",
        topic="test topic",
        details=[],
        claims=[LedgerClaim("Fact 1", "DOC_01_§1", 1, "ACTIVE")],
        evidence={},
        version=1,
    )

    trace = TurnTrace(turn_number=2, session_id="test_suppress_unit")
    res = answer_suppress(state, instruction="reformat as bullet points", entry_id="entry_1", trace=trace)

    assert res.path == "SUPPRESS"
    assert res.llm_calls == 0
    assert "Fact 1" in res.text
    assert state.ledger.entries["entry_1"].version == 1
    assert trace.path == "SUPPRESS"
    assert trace.duration_s >= 0
    assert any(s.stage == "render" for s in trace.stage_events)
