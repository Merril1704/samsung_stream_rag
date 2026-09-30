from __future__ import annotations

import json
from pathlib import Path
import time
import pytest

from controller.llm_wrapper import CountingLLMClient
from controller.model_based import CONTROLLER_SYSTEM_PROMPT
from fusion.contradiction import ContradictionScreen, CONTRADICTION_SYSTEM_PROMPT
from fusion.fusion_pipeline import FusionPipeline
from grounding.decomposer import CLAIM_DECOMPOSER_SYSTEM_PROMPT
from grounding.entailment import ENTAILMENT_SYSTEM_PROMPT
from grounding.verifier import GroundingVerifier
from session.generator import AnswerGenerator, GENERATOR_SYSTEM_PROMPT
from session.pipeline import answer_new_topic, answer_refinement
from controller.types import SessionState
from telemetry import (
    TurnTrace,
    StageEvent,
    LLMCallEvent,
    RetrievalTriggerEvent,
    VersionTransitionEvent,
    estimate_tokens,
    stage_context,
    get_current_stage,
    make_llm_hook,
    write_event_log,
    format_summary,
)


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


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_stage_timing():
    trace = TurnTrace(turn_number=1, session_id="s1")
    trace.begin()
    with trace.start_stage("retrieval", query="test query"):
        time.sleep(0.01)

    assert len(trace.stage_events) == 1
    event = trace.stage_events[0]
    assert event.stage == "retrieval"
    assert event.duration_s > 0
    assert event.metadata == {"query": "test query"}


def test_wall_clock_vs_perf_counter_timing():
    trace = TurnTrace(turn_number=1, session_id="s1")
    trace.begin()
    with trace.start_stage("fusion"):
        time.sleep(0.005)
    trace.finalize(path="NEW_TOPIC")

    event = trace.stage_events[0]
    # Wall-clock timestamps must be valid epoch times (post-2023)
    assert event.wall_start > 1_700_000_000
    assert event.wall_end >= event.wall_start
    # Duration must be a small non-negative float measured via perf_counter, not an epoch timestamp
    assert 0 <= event.duration_s < 1000

    assert trace.wall_start > 1_700_000_000
    assert trace.wall_end >= trace.wall_start
    assert 0 <= trace.duration_s < 1000


def test_token_estimation():
    assert estimate_tokens(0) == 0
    assert estimate_tokens(-5) == 0
    assert estimate_tokens(7) == 2
    assert estimate_tokens(100) == 28
    assert estimate_tokens(3) == 1


def test_multiple_retrieval_events():
    trace = TurnTrace(turn_number=1, session_id="s1")
    trace.begin()

    # Event 1: Controller provisional wait
    e1 = trace.record_retrieval(
        action="WAIT",
        trigger="provisional",
        confidence=0.6,
        query="travel",
        reason="waiting for more tokens",
        transcript_timestamp_s=1.2,
    )

    # Event 2: Controller retrieve decision
    e2 = trace.record_retrieval(
        action="RETRIEVE",
        trigger="final",
        confidence=0.95,
        query="travel reimbursement policy",
        reason="intent stable",
        transcript_timestamp_s=3.5,
    )

    # Event 3: Actual retrieval execution
    e3 = trace.record_retrieval(
        action="RETRIEVE",
        query="travel reimbursement policy",
        chunk_ids_returned=["DOC_06_§1", "DOC_06_§2"],
        reason="execution",
    )

    assert len(trace.retrieval_events) == 3
    assert trace.retrieval_events[0].action == "WAIT"
    assert trace.retrieval_events[0].trigger == "provisional"
    assert trace.retrieval_events[1].action == "RETRIEVE"
    assert trace.retrieval_events[1].transcript_timestamp_s == 3.5
    assert trace.retrieval_events[2].chunk_ids_returned == ["DOC_06_§1", "DOC_06_§2"]


def test_record_version_transition():
    trace = TurnTrace(turn_number=2, session_id="s1")
    event = trace.record_version_transition(
        entry_id="entry_1",
        old_version=1,
        new_version=2,
        claims_added=["New claim added."],
        claims_retired=["Old claim retired."],
        cited_chunk_ids=["DOC_08_§2"],
    )

    assert trace.version_transition is not None
    assert trace.version_transition.entry_id == "entry_1"
    assert trace.version_transition.old_version == 1
    assert trace.version_transition.new_version == 2
    assert trace.version_transition.claims_added == ["New claim added."]
    assert trace.version_transition.claims_retired == ["Old claim retired."]
    assert trace.version_transition.cited_chunk_ids == ["DOC_08_§2"]


def test_finalize_computes_aggregates():
    trace = TurnTrace(turn_number=1, session_id="s1")
    trace.begin()

    trace.record_llm_call(
        stage="generator",
        log_entry={"system_chars": 350, "user_chars": 350, "out_chars": 175, "latency_s": 0.5},
    )
    trace.record_llm_call(
        stage="verifier",
        log_entry={"system_chars": 140, "user_chars": 210, "out_chars": 70, "latency_s": 0.25},
    )

    trace.finalize(path="NEW_TOPIC")

    assert trace.path == "NEW_TOPIC"
    assert trace.total_llm_calls == 2
    # 700 chars / 3.5 = 200; 350 chars / 3.5 = 100 -> total input tokens = 300
    assert trace.total_estimated_input_tokens == 300
    # 175 / 3.5 = 50; 70 / 3.5 = 20 -> total output tokens = 70
    assert trace.total_llm_latency_s == pytest.approx(0.75)
    assert trace.duration_s >= 0


def test_to_dict_json_serializable():
    trace = TurnTrace(turn_number=1, session_id="s1")
    trace.begin()
    with trace.start_stage("retrieval"):
        pass
    trace.record_retrieval(action="RETRIEVE", chunk_ids_returned=["DOC_01_§1"])
    trace.record_llm_call("generator", {"system_chars": 35, "user_chars": 35, "out_chars": 35, "latency_s": 0.1})
    trace.record_version_transition("entry_1", 0, 1, ["claim"], [], ["DOC_01_§1"])
    trace.finalize(path="NEW_TOPIC")

    d = trace.to_dict()
    serialized = json.dumps(d)
    deserialized = json.loads(serialized)

    assert deserialized["turn_number"] == 1
    assert deserialized["session_id"] == "s1"
    assert deserialized["path"] == "NEW_TOPIC"
    assert len(deserialized["stage_events"]) == 1
    assert len(deserialized["retrieval_events"]) == 1
    assert len(deserialized["llm_calls"]) == 1
    assert deserialized["version_transition"]["new_version"] == 1


def test_write_event_log_creates_valid_jsonl():
    trace1 = TurnTrace(turn_number=1, session_id="s1")
    trace1.begin()
    trace1.finalize(path="NEW_TOPIC")

    trace2 = TurnTrace(turn_number=2, session_id="s1")
    trace2.begin()
    trace2.finalize(path="REFINEMENT")

    log_file = Path("scratch") / "telemetry_test_output.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if log_file.exists():
        log_file.unlink()

    try:
        write_event_log([trace1, trace2], log_file)

        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        obj1 = json.loads(lines[0])
        obj2 = json.loads(lines[1])
        assert obj1["turn_number"] == 1
        assert obj2["turn_number"] == 2
    finally:
        if log_file.exists():
            log_file.unlink()


def test_format_summary_includes_key_fields():
    trace = TurnTrace(turn_number=1, session_id="s1")
    trace.begin()
    trace.record_llm_call("generator", {"system_chars": 70, "user_chars": 70, "out_chars": 35, "latency_s": 0.1})
    trace.finalize(path="NEW_TOPIC")

    summary = format_summary(trace)
    assert "Turn 1" in summary
    assert "[NEW_TOPIC]" in summary
    assert "duration=" in summary
    assert "llm_calls=1" in summary
    assert "tokens_in=" in summary
    assert "tokens_out=" in summary


# ---------------------------------------------------------------------------
# LLM Hook & Stage Attribution tests
# ---------------------------------------------------------------------------

def test_llm_callback_stage_attribution():
    recorded_calls = []

    def hook(stage: str, entry: dict):
        recorded_calls.append((stage, entry))

    stub = StubLLM(default="ok")
    client = CountingLLMClient(stub, on_call=hook)

    # 1. Scoped stage context explicitly set
    with stage_context("generator"):
        client.complete("custom prompt", "generate something")

    assert len(recorded_calls) == 1
    assert recorded_calls[-1][0] == "generator"

    with stage_context("verifier"):
        client.complete("custom prompt", "verify claim")

    assert len(recorded_calls) == 2
    assert recorded_calls[-1][0] == "verifier"

    with stage_context("contradiction"):
        client.complete("custom prompt", "screen pair")

    assert len(recorded_calls) == 3
    assert recorded_calls[-1][0] == "contradiction"

    # 2. Fallback prompt-based attribution when outside scoped stage
    assert get_current_stage() == "unknown"

    client.complete(CLAIM_DECOMPOSER_SYSTEM_PROMPT, "split claims")
    assert recorded_calls[-1][0] == "decomposer"

    client.complete(CONTRADICTION_SYSTEM_PROMPT, "check contradiction")
    assert recorded_calls[-1][0] == "contradiction"

    client.complete(ENTAILMENT_SYSTEM_PROMPT, "check entailment")
    assert recorded_calls[-1][0] == "verifier"

    client.complete(GENERATOR_SYSTEM_PROMPT, "generate claims")
    assert recorded_calls[-1][0] == "generator"

    client.complete(CONTROLLER_SYSTEM_PROMPT, "controller decision")
    assert recorded_calls[-1][0] == "controller"


# ---------------------------------------------------------------------------
# Integration tests with Pipeline
# ---------------------------------------------------------------------------

def test_pipeline_new_topic_with_trace(indexed_dev_corpus):
    turn1_payload = json.dumps({"claims": [
        {"claim": "Domestic meals are capped at standard rates.", "chunk_id": "DOC_06_§1"},
    ]})
    stub = StubLLM(responses={
        "Candidate Chunks:": turn1_payload,
        "Claim: ": json.dumps({"supported": True, "reason": "verified"}),
        "Passage A:": json.dumps({"is_contradiction": False, "reason": "distinct scope"}),
    })

    trace = TurnTrace(turn_number=1, session_id="test_sess")
    hook = make_llm_hook(trace)
    counting_client = CountingLLMClient(stub, on_call=hook)

    gen = AnswerGenerator(counting_client)
    verif = GroundingVerifier(counting_client)
    screen = ContradictionScreen(llm_client=counting_client)
    fusion = FusionPipeline(contradiction_screen=screen)
    state = SessionState(session_id="test_sess")

    result = answer_new_topic(
        state,
        "travel reimbursement policy",
        indexed_dev_corpus,
        fusion,
        gen,
        verif,
        llm_counter=counting_client,
        trace=trace,
    )

    assert result.trace is trace
    assert trace.path == "NEW_TOPIC"
    stage_names = [e.stage for e in trace.stage_events]
    assert "retrieval" in stage_names
    assert "fusion" in stage_names
    assert "generator" in stage_names
    assert "verifier" in stage_names
    assert "render" in stage_names

    # Check retrieval events
    assert len(trace.retrieval_events) >= 1
    assert trace.retrieval_events[0].action == "RETRIEVE"
    assert len(trace.retrieval_events[0].chunk_ids_returned) > 0

    # Check version transition
    assert trace.version_transition is not None
    assert trace.version_transition.old_version == 0
    assert trace.version_transition.new_version == 1
    assert "DOC_06_§1" in trace.version_transition.cited_chunk_ids

    # Check LLM call events
    assert trace.total_llm_calls > 0
    assert trace.total_estimated_input_tokens > 0
    assert all(c.stage in ("generator", "verifier", "contradiction") for c in trace.llm_calls)


def test_pipeline_refinement_with_trace(indexed_dev_corpus):
    turn1_payload = json.dumps({"claims": [
        {"claim": "Domestic meals are capped at standard rates.", "chunk_id": "DOC_06_§1"},
    ]})
    turn2_payload = json.dumps({"claims": [
        {"claim": "International travel requires original receipts.", "chunk_id": "DOC_07_§1"},
    ]})
    stub = StubLLM(responses={
        "Detail / Focus:": turn2_payload,
        "Candidate Chunks:": turn1_payload,
        "Claim: ": json.dumps({"supported": True, "reason": "verified"}),
        "Passage A:": json.dumps({"is_contradiction": False, "reason": "distinct scope"}),
    })

    state = SessionState(session_id="test_refine")
    counting_client = CountingLLMClient(stub)
    gen = AnswerGenerator(counting_client)
    verif = GroundingVerifier(counting_client)
    screen = ContradictionScreen(llm_client=counting_client)
    fusion = FusionPipeline(contradiction_screen=screen)

    # Turn 1
    answer_new_topic(state, "travel policy", indexed_dev_corpus, fusion, gen, verif)

    # Turn 2 with trace
    trace2 = TurnTrace(turn_number=2, session_id="test_refine")
    hook = make_llm_hook(trace2)
    counting_client.on_call = hook

    result2 = answer_refinement(
        state,
        "entry_1",
        "international rules",
        indexed_dev_corpus,
        fusion,
        gen,
        verif,
        llm_counter=counting_client,
        trace=trace2,
    )

    assert result2.trace is trace2
    assert trace2.path == "REFINEMENT"
    stage_names = [e.stage for e in trace2.stage_events]
    assert "retrieval" in stage_names
    assert "fusion" in stage_names
    assert "contradiction" in stage_names
    assert "generator" in stage_names
    assert "verifier" in stage_names
    assert "render" in stage_names

    assert trace2.version_transition is not None
    assert trace2.version_transition.old_version == 1
    assert trace2.version_transition.new_version == 2


def test_pipeline_without_trace_unchanged(indexed_dev_corpus):
    turn1_payload = json.dumps({"claims": [
        {"claim": "Domestic meals are capped at standard rates.", "chunk_id": "DOC_06_§1"},
    ]})
    stub = StubLLM(responses={
        "Candidate Chunks:": turn1_payload,
        "Claim: ": json.dumps({"supported": True, "reason": "verified"}),
        "Passage A:": json.dumps({"is_contradiction": False, "reason": "distinct scope"}),
    })

    gen = AnswerGenerator(stub)
    verif = GroundingVerifier(stub)
    screen = ContradictionScreen(llm_client=stub)
    fusion = FusionPipeline(contradiction_screen=screen)
    state = SessionState(session_id="test_notrace")

    result = answer_new_topic(
        state,
        "travel policy",
        indexed_dev_corpus,
        fusion,
        gen,
        verif,
        trace=None,
    )

    assert result.trace is None
    assert result.path == "NEW_TOPIC"
    assert result.text != ""
    assert state.ledger.entries["entry_1"].version == 1
