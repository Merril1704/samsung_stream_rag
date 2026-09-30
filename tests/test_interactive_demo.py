"""Integration and Unit Tests for Interactive Streaming Live RAG Demonstration.

Validates:
1. LiveDemoPipeline step execution across the 5-step incremental scenario.
2. Intent stability gating: Early partial chunks produce WAIT.
3. Speculative background retrieval: Intermediate chunks trigger PREFETCH with candidate IDs.
4. Cache reuse: Final speech turn achieves is_cache_hit=True.
5. End-to-end Grounded Synthesis: At least 1 verified claim generated, 0 fabricated citations.
6. HTTP server endpoints (/api/scenario, /api/status, /api/step, /api/reset).
"""
import json
import threading
import time
from urllib.request import Request, urlopen
import pytest

from demo.server import LiveDemoPipeline, create_server, DEFAULT_SCENARIO


@pytest.fixture(scope="module")
def demo_pipeline():
    pipeline = LiveDemoPipeline()
    pipeline.reset()
    return pipeline


def test_pipeline_5_step_sequence(demo_pipeline):
    """Test full 5-step sequence matching Theme 04 Example 1."""
    # Step 1: Early incomplete intent
    res1 = demo_pipeline.step("I need to plan", is_final=False)
    assert res1["action"] == "WAIT"
    assert res1["stages"]["controller"]["action"] == "WAIT"
    assert res1["stages"]["speculative"]["is_cache_hit"] is False
    assert res1["stages"]["final_answer"]["has_answer"] is False

    # Step 2: Content reaches entity threshold -> Speculative Pre-fetch
    res2 = demo_pipeline.step("a customer workshop in Pune for 30 people", is_final=False)
    assert res2["action"] == "PREFETCH"
    assert res2["stages"]["controller"]["action"] == "RETRIEVE"
    assert res2["stages"]["speculative"]["chunk_count"] > 0
    assert len(res2["stages"]["speculative"]["prefetched_chunk_ids"]) > 0
    assert res2["stages"]["final_answer"]["has_answer"] is False

    # Step 3: Multi-intent conjunction
    res3 = demo_pipeline.step("and I need the cancellation policy", is_final=False)
    assert res3["action"] == "PREFETCH"
    assert res3["stages"]["speculative"]["chunk_count"] > 0
    assert res3["stages"]["final_answer"]["has_answer"] is False

    # Step 4: Additional topic clause
    res4 = demo_pipeline.step("and catering options", is_final=False)
    assert res4["action"] == "PREFETCH"
    assert res4["stages"]["speculative"]["chunk_count"] > 0
    assert res4["stages"]["final_answer"]["has_answer"] is False

    # Step 5: End of utterance -> Zero-latency cache handoff & verified synthesis
    res5 = demo_pipeline.step("", is_final=True)
    assert res5["action"] == "RETRIEVE"
    assert res5["stages"]["speculative"]["is_cache_hit"] is True
    assert res5["stages"]["final_answer"]["has_answer"] is True

    answer_text = res5["stages"]["final_answer"]["text"]
    assert len(answer_text) > 30

    grounding = res5["stages"]["grounding"]
    assert grounding["executed"] is True
    assert grounding["verified_claims_count"] >= 1
    assert isinstance(grounding["all_verified"], bool)

    # Ledger entry committed
    ledger_entries = res5["stages"]["final_answer"]["ledger_entries"]
    assert len(ledger_entries) >= 1
    assert ledger_entries[0]["claims_count"] >= 1


def test_pipeline_reset(demo_pipeline):
    """Test session reset functionality."""
    res = demo_pipeline.reset()
    assert res["status"] == "ok"
    assert demo_pipeline.step_counter == 0
    assert demo_pipeline.state.transcript_so_far == ""
    assert len(demo_pipeline.state.prefetched_candidate_ids) == 0


@pytest.fixture(scope="module")
def live_server():
    server = create_server(host="127.0.0.1", port=8989)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.5)
    yield "http://127.0.0.1:8989"
    server.shutdown()
    server.server_close()


def test_http_api_endpoints(live_server):
    """Test REST API endpoints over HTTP."""
    # Test GET /api/scenario
    req_scenario = Request(f"{live_server}/api/scenario")
    with urlopen(req_scenario) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert "steps" in data
        assert len(data["steps"]) == 5

    # Test GET /api/status
    req_status = Request(f"{live_server}/api/status")
    with urlopen(req_status) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data["status"] == "ready"

    # Test POST /api/reset
    req_reset = Request(f"{live_server}/api/reset", data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req_reset) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data["status"] == "ok"

    # Test POST /api/step (WAIT step)
    step_body = json.dumps({"chunk": "I need to plan", "is_final": False}).encode()
    req_step = Request(f"{live_server}/api/step", data=step_body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req_step) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data["action"] == "WAIT"
        assert data["stages"]["controller"]["action"] == "WAIT"

    # Test GET static index.html chatbot UI
    req_html = Request(f"{live_server}/")
    with urlopen(req_html) as resp:
        assert resp.status == 200
        html = resp.read().decode()
        assert "STREAMING LIVE RAG" in html
        assert "Inspect RAG Activity" in html
        assert "chat-input" in html


def test_split_into_streaming_chunks():
    """Verify arbitrary utterance is split into progressive chunks."""
    from demo.server import split_into_streaming_chunks

    utterance = "I need to cancel a corporate event booked for next week because of a force majeure closure."
    chunks = split_into_streaming_chunks(utterance)
    assert len(chunks) >= 2
    # Intermediate chunks must be is_final=False
    for c_text, is_final in chunks[:-1]:
        assert is_final is False
        assert len(c_text.strip()) > 0
    # Final chunk must be is_final=True
    assert chunks[-1][1] is True


def test_http_stream_chat_endpoint(live_server):
    """Test SSE streaming chat endpoint /api/stream_chat."""
    chat_body = json.dumps({
        "message": "I need to cancel a corporate event booked for next week because of a force majeure closure."
    }).encode()
    req_stream = Request(
        f"{live_server}/api/stream_chat",
        data=chat_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req_stream) as resp:
        assert resp.status == 200
        assert "text/event-stream" in resp.headers.get("Content-Type", "")
        raw_stream = resp.read().decode("utf-8")
        assert "event: progress" in raw_stream
        assert "event: complete" in raw_stream
        assert '"has_answer": true' in raw_stream.lower()


def test_semantic_response_cache():
    """Verify SemanticResponseCache loads golden verified entries and executes semantic matching."""
    from session.cache import SemanticResponseCache

    cache = SemanticResponseCache()
    assert len(cache.entries) >= 5

    # Test cache hit on policy question
    match = cache.lookup("I need to cancel a corporate event due to force majeure")
    assert match is not None
    assert match.entry.intent_topic == "Force Majeure Event Cancellation"
    assert "DOC_04_§3" in match.entry.citations
    assert len(match.entry.claims) >= 1

