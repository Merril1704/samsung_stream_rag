from __future__ import annotations

import json
from pathlib import Path
import pytest

from controller.llm_wrapper import CountingLLMClient
from controller.types import SessionState
from fusion.contradiction import ContradictionScreen
from fusion.fusion_pipeline import FusionPipeline
from grounding.verifier import GroundingVerifier
from session.generator import AnswerGenerator
from session.pipeline import answer_new_topic, answer_refinement
from telemetry import TurnTrace, make_llm_hook, write_event_log


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


def test_telemetry_two_turn_e2e(indexed_dev_corpus):
    turn1_payload = json.dumps({"claims": [
        {"claim": "Standard domestic meal reimbursement is capped.", "chunk_id": "DOC_06_§1"},
        {"claim": "Itemized receipts are required for hotel expenses.", "chunk_id": "DOC_06_§2"},
    ]})
    turn2_payload = json.dumps({"claims": [
        {"claim": "International travel requires currency conversion slips.", "chunk_id": "DOC_07_§1"},
    ]})

    stub = StubLLM(responses={
        "Detail / Focus:": turn2_payload,
        "Candidate Chunks:": turn1_payload,
        "Claim: ": json.dumps({"supported": True, "reason": "entailed by text"}),
        "Passage A:": json.dumps({"is_contradiction": False, "reason": "different scope"}),
    })

    client = CountingLLMClient(stub)
    gen = AnswerGenerator(client)
    verif = GroundingVerifier(client)
    screen = ContradictionScreen(llm_client=client)
    fusion = FusionPipeline(contradiction_screen=screen)

    session_id = "sess_e2e_001"
    state = SessionState(session_id=session_id)

    # ---------------------------------------------------------
    # Turn 1: NEW_TOPIC
    # ---------------------------------------------------------
    trace1 = TurnTrace(turn_number=1, session_id=session_id)
    client.on_call = make_llm_hook(trace1)

    res1 = answer_new_topic(
        state=state,
        topic="travel expense reimbursement policy",
        index=indexed_dev_corpus,
        fusion=fusion,
        generator=gen,
        verifier=verif,
        llm_counter=client,
        trace=trace1,
    )

    assert res1.trace is trace1
    assert trace1.path == "NEW_TOPIC"

    # ---------------------------------------------------------
    # Turn 2: REFINEMENT
    # ---------------------------------------------------------
    trace2 = TurnTrace(turn_number=2, session_id=session_id)
    client.on_call = make_llm_hook(trace2)

    res2 = answer_refinement(
        state=state,
        entry_id="entry_1",
        detail="international travel currency conversion",
        index=indexed_dev_corpus,
        fusion=fusion,
        generator=gen,
        verifier=verif,
        llm_counter=client,
        trace=trace2,
    )

    assert res2.trace is trace2
    assert trace2.path == "REFINEMENT"

    # ---------------------------------------------------------
    # Persistence: JSONL
    # ---------------------------------------------------------
    log_file = Path("scratch") / "e2e_traces.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if log_file.exists():
        log_file.unlink()

    try:
        write_event_log([trace1, trace2], log_file)

        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2, f"Expected exactly two lines in JSONL, got {len(lines)}"

        parsed_turn1 = json.loads(lines[0])
        parsed_turn2 = json.loads(lines[1])

        # ---------------------------------------------------------
        # Assertions on Turn 1 & Turn 2 traces
        # ---------------------------------------------------------
        traces = [trace1, trace2]
        parsed_traces = [parsed_turn1, parsed_turn2]

        for idx, (tr, ptr) in enumerate(zip(traces, parsed_traces), start=1):
            assert tr.turn_number == idx
            assert tr.session_id == session_id
            # Duration must be positive
            assert tr.duration_s > 0, f"Turn {idx} duration should be > 0"
            # Wall timestamps must be epoch times post-2023 (> 1_700_000_000)
            assert tr.wall_start > 1_700_000_000, f"Turn {idx} wall_start not a valid wall-clock timestamp"
            assert tr.wall_end >= tr.wall_start, f"Turn {idx} wall_end before wall_start"
            # Duration is perf_counter delta, must NOT be an epoch timestamp
            assert tr.duration_s < 1000

            # Stage events validation
            stage_names = [s.stage for s in tr.stage_events]
            assert "retrieval" in stage_names
            assert "fusion" in stage_names
            assert "generator" in stage_names
            assert "verifier" in stage_names
            assert "render" in stage_names

            for stage_ev in tr.stage_events:
                assert stage_ev.wall_start > 1_700_000_000
                assert stage_ev.wall_end >= stage_ev.wall_start
                assert stage_ev.duration_s >= 0
                assert stage_ev.duration_s < 1000

            # LLM call events validation
            assert tr.total_llm_calls > 0
            assert tr.total_estimated_input_tokens > 0
            assert tr.total_estimated_output_tokens > 0
            for llm_ev in tr.llm_calls:
                assert llm_ev.wall_timestamp > 1_700_000_000
                assert llm_ev.latency_s >= 0
                assert llm_ev.latency_s < 1000
                assert llm_ev.estimated_input_tokens > 0
                assert llm_ev.estimated_output_tokens > 0
                # Ensure stage attribution is valid
                assert llm_ev.stage in ("generator", "verifier", "contradiction", "decomposer")

        # Turn 2 specific validation
        assert "contradiction" in [s.stage for s in trace2.stage_events]
        assert trace2.version_transition is not None
        assert trace2.version_transition.old_version == 1
        assert trace2.version_transition.new_version == 2
        assert "DOC_07_§1" in trace2.version_transition.cited_chunk_ids
    finally:
        if log_file.exists():
            log_file.unlink()
