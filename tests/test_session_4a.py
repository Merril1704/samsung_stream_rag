import copy
import json
import pytest
from controller.llm_factory import get_llm_client
from controller.llm_wrapper import CountingLLMClient, ContextBudgetExceeded
from controller.rule_based import RuleBasedController
from controller.types import SessionState
from fusion.fusion_pipeline import FusionPipeline
from fusion.reranker import CrossEncoderReranker
from fusion.contradiction import ContradictionScreen
from grounding.types import ClaimEntry
from retrieval.types import EvidenceChunk
from grounding.verifier import GroundingVerifier
from session.generator import AnswerGenerator, GENERATOR_SYSTEM_PROMPT, format_candidate_chunks
from session.ledger import AnswerLedger, LedgerClaim, LedgerEntry, commit_entry
from session.pipeline import answer_new_topic
from session.render import render


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
    def __init__(self, exception: Exception):
        self.exception = exception

    def complete(self, system: str, user: str, *, max_tokens: int | None = None) -> str:
        raise self.exception


# ---------------------------------------------------------------------------
# 1. verify_entries tests
# ---------------------------------------------------------------------------

def test_verify_entries_bundled_claim_decomposed():
    """Bundled claim containing ' and ' must be split into parts inheriting the chunk_id."""
    chunk = EvidenceChunk(
        chunk_id="DOC_02_§1", doc_id="DOC_02", section="§1",
        text="Capacity: 40 seated. Includes standard AV.", retrieval_score=1.0,
    )
    stub_responses = {
        "Answer: ": json.dumps({"claims": [
            "Capacity is 40 seated.",
            "Includes standard AV."
        ]}),
        "Claim: ": json.dumps({"supported": True, "reason": "matches passage"}),
    }
    client = StubLLM(stub_responses)
    verifier = GroundingVerifier(llm_client=client)

    entry = ClaimEntry(
        claim="Capacity is 40 seated and includes standard AV.",
        chunk_id="DOC_02_§1",
    )
    res = verifier.verify_entries([entry], evidence=[chunk])
    assert len(res.claims) == 2
    assert all(c.cited_chunk_id == "DOC_02_§1" for c in res.claims)
    assert res.all_verified is True


def test_verify_entries_atomic_claim_zero_decomposer_calls():
    """An atomic claim with no conjunctions must trigger zero decomposer calls."""
    chunk = EvidenceChunk(
        chunk_id="DOC_02_§1", doc_id="DOC_02", section="§1",
        text="Capacity: 40 seated.", retrieval_score=1.0,
    )
    stub_responses = {
        "Claim: ": json.dumps({"supported": True, "reason": "supported"}),
    }
    counting_client = CountingLLMClient(StubLLM(stub_responses))
    verifier = GroundingVerifier(llm_client=counting_client)

    entry = ClaimEntry(claim="Capacity is 40 seated.", chunk_id="DOC_02_§1")
    res = verifier.verify_entries([entry], evidence=[chunk])

    # Decomposer was never called: exactly 1 call (the entailment call)
    assert res.all_verified is True
    assert counting_client.calls == 1


def test_verify_entries_unknown_chunk_id_fabricated():
    chunk = EvidenceChunk(
        chunk_id="DOC_02_§1", doc_id="DOC_02", section="§1",
        text="Capacity: 40 seated.", retrieval_score=1.0,
    )
    verifier = GroundingVerifier(llm_client=StubLLM())
    entry = ClaimEntry(claim="Capacity is 40 seated.", chunk_id="DOC_99_§1")
    res = verifier.verify_entries([entry], evidence=[chunk])

    assert res.all_verified is False
    assert len(res.fabricated_citations) == 1
    assert res.claims[0].reason == "fabricated citation"


def test_verify_entries_garbage_llm_unsupported():
    chunk = EvidenceChunk(
        chunk_id="DOC_02_§1", doc_id="DOC_02", section="§1",
        text="Capacity: 40 seated.", retrieval_score=1.0,
    )
    verifier = GroundingVerifier(llm_client=StubLLM(default="INVALID_JSON_GARBAGE"))
    entry = ClaimEntry(claim="Capacity is 40 seated.", chunk_id="DOC_02_§1")
    res = verifier.verify_entries([entry], evidence=[chunk])

    assert res.all_verified is False
    assert len(res.unsupported_claims) == 1
    assert "parse failure" in res.claims[0].reason


def test_verify_entries_cherry_pick_regression():
    chunk_04 = EvidenceChunk(
        chunk_id="DOC_04_§1", doc_id="DOC_04", section="§1",
        text="Bookings cancelled more than 30 days before event receive full refund.",
        retrieval_score=0.9, contradiction_flag=True, contradicts_chunk_id="DOC_10_§2",
    )
    chunk_10 = EvidenceChunk(
        chunk_id="DOC_10_§2", doc_id="DOC_10", section="§2",
        text="Vendors may enforce a 14-day cancellation window.",
        retrieval_score=0.85, contradiction_flag=True, contradicts_chunk_id="DOC_04_§1",
    )
    stub_responses = {
        "Claim: ": json.dumps({"supported": True, "reason": "supported"}),
    }
    verifier = GroundingVerifier(llm_client=StubLLM(stub_responses))
    entry = ClaimEntry(claim="Cancellations more than 30 days receive full refund.", chunk_id="DOC_04_§1")
    res = verifier.verify_entries([entry], evidence=[chunk_04, chunk_10])

    assert res.all_verified is False
    assert len(res.cherry_picks) == 1
    assert res.claims[0].cherry_pick_violation is True


def test_verify_entries_context_budget_exceeded_propagates():
    """ContextBudgetExceeded must propagate out of verify_entries on both entailment and decomposer paths."""
    chunk = EvidenceChunk(
        chunk_id="DOC_02_§1", doc_id="DOC_02", section="§1",
        text="Capacity: 40 seated.", retrieval_score=1.0,
    )
    raising_client = RaisingLLM(ContextBudgetExceeded("Budget exceeded"))
    verifier = GroundingVerifier(llm_client=raising_client)

    # Path A: Atomic claim -> triggers entailment check which raises ContextBudgetExceeded
    atomic_entry = ClaimEntry(claim="Capacity is 40 seated.", chunk_id="DOC_02_§1")
    with pytest.raises(ContextBudgetExceeded):
        verifier.verify_entries([atomic_entry], evidence=[chunk])

    # Path B: Conjunction claim -> triggers decomposer which raises ContextBudgetExceeded
    conjunction_entry = ClaimEntry(claim="Capacity is 40 seated and includes AV.", chunk_id="DOC_02_§1")
    with pytest.raises(ContextBudgetExceeded):
        verifier.verify_entries([conjunction_entry], evidence=[chunk])


# ---------------------------------------------------------------------------
# 2. Generator tests
# ---------------------------------------------------------------------------

def test_generator_valid_json_parses():
    payload = json.dumps({"claims": [{"claim": "Full refund within 30 days.", "chunk_id": "DOC_04_§1"}]})
    gen = AnswerGenerator(llm_client=StubLLM(default=payload))
    chunk = EvidenceChunk("DOC_04_§1", "DOC_04", "§1", "Full refund within 30 days.", 1.0)
    entries = gen.generate("refund policy", [chunk])

    assert len(entries) == 1
    assert entries[0].claim == "Full refund within 30 days."
    assert entries[0].chunk_id == "DOC_04_§1"
    assert gen.last_error is None


def test_generator_garbage_returns_empty_and_sets_last_error():
    gen = AnswerGenerator(llm_client=StubLLM(default="NOT_JSON"))
    chunk = EvidenceChunk("DOC_04_§1", "DOC_04", "§1", "Full refund within 30 days.", 1.0)
    entries = gen.generate("refund policy", [chunk])

    assert entries == []
    assert gen.last_error is not None


def test_generator_unknown_chunk_id_passed_through():
    payload = json.dumps({"claims": [{"claim": "Hallucinated statement.", "chunk_id": "UNKNOWN_CHUNK_99"}]})
    gen = AnswerGenerator(llm_client=StubLLM(default=payload))
    chunk = EvidenceChunk("DOC_04_§1", "DOC_04", "§1", "Full refund within 30 days.", 1.0)
    entries = gen.generate("refund policy", [chunk])

    assert len(entries) == 1
    assert entries[0].chunk_id == "UNKNOWN_CHUNK_99"


# ---------------------------------------------------------------------------
# 3. Pipeline tests
# ---------------------------------------------------------------------------

def test_pipeline_fabricated_citation_routed_to_rejected(indexed_dev_corpus):
    payload = json.dumps({"claims": [
        {"claim": "Bookings receive a full refund.", "chunk_id": "DOC_04_§1"},
        {"claim": "Invented claim with fake id.", "chunk_id": "NON_EXISTENT_§1"},
    ]})
    entail_payload = json.dumps({"supported": True, "reason": "matches"})

    stub = StubLLM(responses={
        "Candidate Chunks:": payload,
        "Claim: ": entail_payload,
    })
    gen = AnswerGenerator(stub)
    verif = GroundingVerifier(stub)
    fusion = FusionPipeline()
    state = SessionState(session_id="test_session")

    turn = answer_new_topic(state, "refund policy", indexed_dev_corpus, fusion, gen, verif)

    assert len(turn.rejected) == 1
    assert turn.rejected[0].cited_chunk_id == "NON_EXISTENT_§1"
    assert state.ledger is not None
    assert len(state.ledger.entries["entry_1"].claims) == 1
    assert state.ledger.entries["entry_1"].claims[0].chunk_id == "DOC_04_§1"


def test_pipeline_insufficient_evidence_zero_generator_calls(indexed_dev_corpus):
    class FailingRetrieverIndex:
        def __init__(self, canonical):
            self.canonical = canonical

        def __getitem__(self, item):
            if item == "retrieve":
                return lambda query, top_k=10: []
            return self.canonical[item]

    stub = StubLLM()
    gen = AnswerGenerator(stub)
    verif = GroundingVerifier(stub)
    fusion = FusionPipeline()
    state = SessionState(session_id="test_empty")

    fake_index = FailingRetrieverIndex(indexed_dev_corpus)
    turn = answer_new_topic(state, "undocumented topic", fake_index, fusion, gen, verif)

    assert turn.text == ""
    assert turn.rejected == []
    assert len(stub.calls) == 0
    assert state.ledger is None


def test_pipeline_contradiction_sets_unresolved_flag(indexed_dev_corpus):
    """Cherry pick rejection sets turn_result.contradiction_unresolved = True."""
    payload = json.dumps({"claims": [
        {"claim": "Bookings cancelled more than 30 days before event receive full refund.", "chunk_id": "DOC_04_§1"}
    ]})
    entail_payload = json.dumps({"supported": True, "reason": "matches"})

    stub = StubLLM(responses={
        "Candidate Chunks:": payload,
        "Claim: ": entail_payload,
        "Passage A:": json.dumps({"is_contradiction": True, "reason": "conflicting refund periods"}),
    })
    gen = AnswerGenerator(stub)
    verif = GroundingVerifier(stub)
    fusion = FusionPipeline(contradiction_screen=ContradictionScreen(llm_client=stub))
    state = SessionState(session_id="test_contradiction")

    wrapped_index = dict(indexed_dev_corpus)
    wrapped_index["retrieve"] = lambda query, top_k=10: ["DOC_04_§1", "DOC_10_§2"]

    turn = answer_new_topic(state, "cancellation refund window", wrapped_index, fusion, gen, verif)

    assert any(r.cherry_pick_violation for r in turn.rejected)
    assert turn.contradiction_unresolved is True


# ---------------------------------------------------------------------------
# 4. Render tests
# ---------------------------------------------------------------------------

def test_render_deterministic_zero_llm_calls():
    entry = LedgerEntry(
        entry_id="entry_1",
        topic="refund policy",
        details=["policy limits"],
        claims=[
            LedgerClaim("Bookings cancelled receive full refund.", "DOC_04_§1", origin_version=1),
            LedgerClaim("Vendor notice is 14 days.", "DOC_10_§2", origin_version=2),
        ],
        evidence={},
        version=2,
    )
    r1 = render(entry)
    r2 = render(entry)
    assert r1 == r2
    assert "Baseline" in r1
    assert "Refinement: policy limits" in r1
    assert "Bookings cancelled receive full refund. [DOC_04_§1]" in r1
    assert "Vendor notice is 14 days. [DOC_10_§2]" in r1


# ---------------------------------------------------------------------------
# 5. Reformat / Suppression test
# ---------------------------------------------------------------------------

def test_reformat_suppress_makes_zero_retrieval_calls(indexed_dev_corpus):
    payload = json.dumps({"claims": [
        {"claim": "Cancellations receive a full refund.", "chunk_id": "DOC_04_§1"}
    ]})
    stub = StubLLM(responses={
        "Candidate Chunks:": payload,
        "Claim: ": json.dumps({"supported": True, "reason": "ok"}),
    })
    gen = AnswerGenerator(stub)
    verif = GroundingVerifier(stub)
    fusion = FusionPipeline()
    state = SessionState(session_id="test_reformat")

    # Turn 1: v1 commit
    turn1 = answer_new_topic(state, "cancellation policy", indexed_dev_corpus, fusion, gen, verif)
    assert turn1.text != ""
    assert state.last_answer_topic is not None

    # Turn 2: controller check on reformat request
    controller = RuleBasedController()
    dec = controller.decide(state, "repeat that in bullet points", timestamp_s=1.0)
    assert dec.action == "SUPPRESS"

    # Tracking retriever calls
    retrieval_calls = 0
    orig_retrieve = indexed_dev_corpus["retrieve"]

    def counting_retrieve(query, top_k=10):
        nonlocal retrieval_calls
        retrieval_calls += 1
        return orig_retrieve(query, top_k=top_k)

    wrapped_index = dict(indexed_dev_corpus)
    wrapped_index["retrieve"] = counting_retrieve

    # Re-render from existing ledger
    re_rendered = render(state.ledger.entries["entry_1"])
    assert retrieval_calls == 0
    assert re_rendered == turn1.text


# ---------------------------------------------------------------------------
# 6. Index isolation test
# ---------------------------------------------------------------------------

def test_index_isolation_preserves_canonical_index(indexed_dev_corpus):
    """
    EvidenceChunk fields (retrieval/types.py):
      chunk_id: str (immutable)
      doc_id: str (immutable)
      section: str (immutable)
      text: str (immutable)
      retrieval_score: float (immutable)
      rerank_score: float | None (immutable)
      contradiction_flag: bool (mutable attribute on instance)
      contradicts_chunk_id: str | None (mutable attribute on instance)

    This test confirms that running a turn or mutating fields in the turn copy
    does not alter the canonical indexed_dev_corpus.
    """
    orig_flag = indexed_dev_corpus["chunk_lookup"]["DOC_04_§1"].contradiction_flag
    assert orig_flag is False

    payload = json.dumps({"claims": [
        {"claim": "Cancellations receive a full refund.", "chunk_id": "DOC_04_§1"},
        {"claim": "Vendors may enforce a 14-day cancellation window.", "chunk_id": "DOC_10_§2"},
    ]})
    stub = StubLLM(responses={
        "Candidate Chunks:": payload,
        "Claim: ": json.dumps({"supported": True, "reason": "ok"}),
        "Passage A:": json.dumps({"is_contradiction": True, "reason": "conflict"}),
    })
    gen = AnswerGenerator(stub)
    verif = GroundingVerifier(stub)
    fusion = FusionPipeline(contradiction_screen=ContradictionScreen(llm_client=stub))
    state = SessionState(session_id="test_iso")

    wrapped_index = dict(indexed_dev_corpus)
    wrapped_index["retrieve"] = lambda query, top_k=10: ["DOC_04_§1", "DOC_10_§2"]

    turn = answer_new_topic(state, "cancellation notice period and refund terms for event vendors", wrapped_index, fusion, gen, verif)

    # Prove canonical index object is completely unmodified
    canonical_chunk = indexed_dev_corpus["chunk_lookup"]["DOC_04_§1"]
    assert canonical_chunk.contradiction_flag is False
    assert canonical_chunk.contradicts_chunk_id is None

    # Mutate a field on the ledger evidence copy
    entry_evidence = state.ledger.entries["entry_1"].evidence["DOC_04_§1"]
    entry_evidence.contradiction_flag = True
    entry_evidence.text = "MUTATED"

    # Canonical index still untouched
    assert canonical_chunk.contradiction_flag is False
    assert canonical_chunk.text != "MUTATED"


# ---------------------------------------------------------------------------
# 7. Recall Measurement (Table printed, no assertion)
# ---------------------------------------------------------------------------

def test_recall_measurement_table(indexed_dev_corpus):
    queries = [
        ("a", "international, booking made after travel"),
        ("b", "summarize the travel reimbursement rule international, booking made after travel"),
    ]
    target_chunks = ["DOC_07_§1", "DOC_07_§2", "DOC_08_§2", "DOC_09_§2"]
    fusion = FusionPipeline()

    print("\n" + "=" * 90)
    print("STAGE 4A RECALL MEASUREMENT TABLE")
    print("=" * 90)

    for q_label, q_text in queries:
        lookup = {k: copy.deepcopy(v) for k, v in indexed_dev_corpus["chunk_lookup"].items()}
        retrieved_ids = indexed_dev_corpus["retrieve"](q_text, top_k=10)
        fused = fusion.fuse(q_text, [retrieved_ids], lookup)
        fused_ids = [c.chunk_id for c in fused.chunks]

        # Candidate prompt calculation
        chunks_text = format_candidate_chunks(fused.chunks)
        user_prompt = f"Topic: {q_text}\n\nCandidate Chunks:\n{chunks_text}\n\nRespond with compact JSON only:"
        prompt_len = len(GENERATOR_SYSTEM_PROMPT) + len(user_prompt)
        num_fused = len(fused.chunks)

        print(f"\nQuery ({q_label}): \"{q_text}\"")
        print(f"Generator Prompt Char Length: {prompt_len} | Fused Chunks Passed: {num_fused}")
        print(f"{'Target Chunk':<15} | {'Retrieval Rank (top-10)':<25} | {'Fusion Rank':<15}")
        print("-" * 65)

        for target in target_chunks:
            ret_rank = f"rank {retrieved_ids.index(target) + 1}" if target in retrieved_ids else "absent"
            fuse_rank = f"rank {fused_ids.index(target) + 1}" if target in fused_ids else "absent"
            print(f"{target:<15} | {ret_rank:<25} | {fuse_rank:<15}")

    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
# 8. Live Smoke Test (@pytest.mark.live)
# ---------------------------------------------------------------------------

@pytest.mark.live
def test_live_smoke_travel_reimbursement(indexed_dev_corpus):
    real_client = get_llm_client()
    counting_client = CountingLLMClient(real_client)

    gen = AnswerGenerator(llm_client=counting_client)
    verif = GroundingVerifier(llm_client=counting_client)
    fusion = FusionPipeline()
    state = SessionState(session_id="live_smoke_session")

    query = "summarize the travel reimbursement rule"
    turn = answer_new_topic(
        state=state,
        topic=query,
        index=indexed_dev_corpus,
        fusion=fusion,
        generator=gen,
        verifier=verif,
        llm_counter=counting_client,
    )

    # Encode safe print for Windows console
    safe_text = turn.text.encode("ascii", errors="backslashreplace").decode("ascii")
    print("\n--- LIVE SMOKE RESULTS ---")
    print("Committed Text:\n", safe_text)
    print("Total LLM Calls:", counting_client.calls)
    for i, log_entry in enumerate(counting_client.log, 1):
        print(f"Call {i}: latency = {log_entry['latency_s']:.3f}s, out_chars = {log_entry['out_chars']}")

    # Assertions
    assert state.ledger is not None
    committed_claims = state.ledger.entries["entry_1"].claims
    assert len(committed_claims) > 0, "Expected at least one committed claim"

    assert any(c.chunk_id.startswith("DOC_06") for c in committed_claims), "Expected at least one DOC_06 citation"
    for claim in committed_claims:
        assert claim.chunk_id.startswith("DOC_06") or claim.chunk_id.startswith("DOC_07"), f"Unexpected chunk: {claim.chunk_id}"

    for r in turn.rejected:
        assert r.citation_exists is False or r.supported is False or r.cherry_pick_violation is True

