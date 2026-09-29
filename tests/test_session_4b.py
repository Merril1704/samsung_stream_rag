import json
import pytest
from controller.llm_factory import get_llm_client
from controller.llm_wrapper import CountingLLMClient
from controller.rule_based import RuleBasedController
from controller.types import SessionState
from fusion.fusion_pipeline import FusionPipeline
from fusion.contradiction import ContradictionScreen
from retrieval.types import EvidenceChunk
from grounding.verifier import GroundingVerifier
from session.generator import AnswerGenerator
from session.ledger import active_claims, LedgerClaim
from session.pipeline import answer_new_topic, answer_refinement
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


# ---------------------------------------------------------------------------
# 2a. Multi-turn Scenario B trace
# ---------------------------------------------------------------------------

def test_multiturn_scenario_b_trace(indexed_dev_corpus):
    turn1_payload = json.dumps({"claims": [
        {"claim": "Domestic meals are capped at standard rates.", "chunk_id": "DOC_06_§1"},
        {"claim": "Domestic expense reports require receipts over 500.", "chunk_id": "DOC_06_§2"},
    ]})
    turn2_payload = json.dumps({"claims": [
        {"claim": "International travel requires original receipts.", "chunk_id": "DOC_07_§1"},
        {"claim": "Post-travel bookings require exception approval.", "chunk_id": "DOC_08_§2"},
        {"claim": "Director sign-off required for workflow exceptions.", "chunk_id": "DOC_09_§2"},
    ]})
    entail_payload = json.dumps({"supported": True, "reason": "supported"})
    no_contra_payload = json.dumps({"is_contradiction": False, "reason": "different aspects"})

    stub = StubLLM(responses={
        "Detail / Focus: international": turn2_payload,
        "Candidate Chunks:": turn1_payload,
        "Claim: ": entail_payload,
        "Passage A:": no_contra_payload,
    })

    gen = AnswerGenerator(stub)
    verif = GroundingVerifier(stub)
    screen = ContradictionScreen(llm_client=stub)
    fusion = FusionPipeline(contradiction_screen=screen)
    state = SessionState(session_id="test_scenario_b")

    # Turn 1: Initial topic
    turn1 = answer_new_topic(state, "summarize the travel reimbursement rule", indexed_dev_corpus, fusion, gen, verif)
    assert turn1.path == "NEW_TOPIC"
    entry = state.ledger.entries["entry_1"]
    assert entry.version == 1
    t1_claims = active_claims(entry)
    assert len(t1_claims) == 2
    assert all(c.chunk_id.startswith("DOC_06") for c in t1_claims)
    assert all(c.origin_version == 1 and c.status == "ACTIVE" for c in t1_claims)

    # Turn 2: Refinement
    turn2 = answer_refinement(state, "entry_1", "international, booking made after travel", indexed_dev_corpus, fusion, gen, verif)
    assert turn2.path == "REFINEMENT"
    assert entry.version == 2
    assert len(entry.details) == 1
    assert entry.details[0] == "international, booking made after travel"

    t2_claims = active_claims(entry)
    active_chunk_ids = [c.chunk_id for c in t2_claims]
    assert len(active_chunk_ids) == len(set(active_chunk_ids)), "Duplicate chunk_id found in active claims"

    # All DOC_06 claims still active with origin_version=1
    for c in t2_claims:
        if c.chunk_id.startswith("DOC_06"):
            assert c.status == "ACTIVE"
            assert c.origin_version == 1
        else:
            assert c.status == "ACTIVE"
            assert c.origin_version == 2


# ---------------------------------------------------------------------------
# 2b. Dedup test
# ---------------------------------------------------------------------------

def test_refinement_dedup_skips_already_cited_chunk(indexed_dev_corpus):
    turn1_payload = json.dumps({"claims": [
        {"claim": "Domestic travel rules apply.", "chunk_id": "DOC_06_§1"},
    ]})
    # In turn 2, generator attempts to re-cite DOC_06_§1
    turn2_payload = json.dumps({"claims": [
        {"claim": "Domestic meals are capped at standard rates.", "chunk_id": "DOC_06_§1"},
    ]})
    stub = StubLLM(responses={
        "Detail / Focus: extra info": turn2_payload,
        "Candidate Chunks:": turn1_payload,
        "Claim: ": json.dumps({"supported": True, "reason": "ok"}),
        "Passage A:": json.dumps({"is_contradiction": False, "reason": "none"}),
    })

    gen = AnswerGenerator(stub)
    verif = GroundingVerifier(stub)
    screen = ContradictionScreen(llm_client=stub)
    fusion = FusionPipeline(contradiction_screen=screen)
    state = SessionState(session_id="test_dedup")

    turn1 = answer_new_topic(state, "travel policy", indexed_dev_corpus, fusion, gen, verif)
    assert state.ledger.entries["entry_1"].version == 1

    turn2 = answer_refinement(state, "entry_1", "extra info", indexed_dev_corpus, fusion, gen, verif)
    assert "Domestic meals are capped at standard rates." in turn2.skipped_duplicates
    assert len(active_claims(state.ledger.entries["entry_1"])) == 1
    # Version not bumped because it was the only candidate
    assert state.ledger.entries["entry_1"].version == 1


def test_refinement_multiple_claims_same_new_chunk_consolidated(indexed_dev_corpus):
    """
    When multiple distinct claims cite the same genuinely new chunk in refinement,
    they are consolidated into a single active LedgerClaim to satisfy the invariant
    that each chunk_id appears at most once in active_claims.
    """
    turn1_payload = json.dumps({"claims": [
        {"claim": "Domestic travel rules apply.", "chunk_id": "DOC_06_§1"},
    ]})
    # In turn 2, generator emits two distinct factual claims for DOC_08_§2
    turn2_payload = json.dumps({"claims": [
        {"claim": "Post-travel late booking requires written justification.", "chunk_id": "DOC_08_§2"},
        {"claim": "Dual approval is needed for processing post-travel late bookings.", "chunk_id": "DOC_08_§2"},
    ]})
    stub = StubLLM(responses={
        "Detail / Focus: booking made after travel": turn2_payload,
        "Candidate Chunks:": turn1_payload,
        "Claim: ": json.dumps({"supported": True, "reason": "ok"}),
        "Passage A:": json.dumps({"is_contradiction": False, "reason": "none"}),
    })

    gen = AnswerGenerator(stub)
    verif = GroundingVerifier(stub)
    screen = ContradictionScreen(llm_client=stub)
    fusion = FusionPipeline(contradiction_screen=screen)
    state = SessionState(session_id="test_consolidation")

    turn1 = answer_new_topic(state, "travel policy", indexed_dev_corpus, fusion, gen, verif)
    assert state.ledger.entries["entry_1"].version == 1

    turn2 = answer_refinement(state, "entry_1", "booking made after travel", indexed_dev_corpus, fusion, gen, verif)
    entry = state.ledger.entries["entry_1"]
    assert entry.version == 2

    t2_claims = active_claims(entry)
    active_chunk_ids = [c.chunk_id for c in t2_claims]
    assert len(active_chunk_ids) == len(set(active_chunk_ids)), f"Duplicate active chunk_ids found: {active_chunk_ids}"
    assert "DOC_08_§2" in active_chunk_ids

    # Both facts must be preserved in the consolidated claim without blindly deleting one
    doc08_claim = next(c for c in t2_claims if c.chunk_id == "DOC_08_§2")
    assert "written justification" in doc08_claim.claim
    assert "Dual approval" in doc08_claim.claim


# ---------------------------------------------------------------------------
# 2c. Supersession test (synthetic)
# ---------------------------------------------------------------------------

def test_refinement_supersession_retires_conflicted_claim():
    chunk_a = EvidenceChunk("CHUNK_A", "DOC_A", "§1", "The event notice is 30 days.", 1.0)
    chunk_b = EvidenceChunk("CHUNK_B", "DOC_B", "§1", "The event notice is 14 days.", 1.0)

    class CustomIndex:
        def __init__(self):
            self.chunk_lookup = {"CHUNK_A": chunk_a, "CHUNK_B": chunk_b}
        def __getitem__(self, item):
            if item == "chunk_lookup":
                return self.chunk_lookup
            if item == "retrieve":
                return lambda query, top_k=10: ["CHUNK_B"]
            raise KeyError(item)

    index = CustomIndex()
    stub = StubLLM(responses={
        "Detail / Focus: vendor notice": json.dumps({"claims": [
            {"claim": "Event notice is 14 days.", "chunk_id": "CHUNK_B"}
        ]}),
        "Claim: ": json.dumps({"supported": True, "reason": "supported"}),
        "Passage A:": json.dumps({"is_contradiction": True, "reason": "conflicting notice windows"}),
    })

    gen = AnswerGenerator(stub)
    verif = GroundingVerifier(stub)
    screen = ContradictionScreen(llm_client=stub)
    fusion = FusionPipeline(contradiction_screen=screen)

    state = SessionState(session_id="test_supersession")
    # Setup initial entry manually
    from session.ledger import AnswerLedger, LedgerEntry
    state.ledger = AnswerLedger()
    entry = LedgerEntry(
        entry_id="entry_1",
        topic="event notice policy",
        details=[],
        claims=[LedgerClaim("Event notice is 30 days.", "CHUNK_A", origin_version=1, status="ACTIVE")],
        evidence={"CHUNK_A": chunk_a},
        version=1,
    )
    state.ledger.entries["entry_1"] = entry

    turn2 = answer_refinement(state, "entry_1", "vendor notice", index, fusion, gen, verif)

    # CHUNK_A claim must be retired, CHUNK_B claim must be active
    assert entry.claims[0].status == "RETIRED"
    assert entry.claims[1].status == "ACTIVE"
    assert entry.claims[1].chunk_id == "CHUNK_B"
    assert entry.claims[1].origin_version == 2
    assert entry.version == 2

    # Verify history
    history_entry = state.ledger.history[-1]
    assert history_entry["action"] == "REFINEMENT"
    assert "Event notice is 30 days." in history_entry["retired"]
    assert "Event notice is 14 days." in history_entry["added"]


# ---------------------------------------------------------------------------
# 2d. Missing-screen guard
# ---------------------------------------------------------------------------

def test_answer_refinement_missing_screen_raises_value_error(indexed_dev_corpus):
    fusion = FusionPipeline(contradiction_screen=None)
    stub = StubLLM()
    gen = AnswerGenerator(stub)
    verif = GroundingVerifier(stub)
    state = SessionState(session_id="test_guard")

    # Construct initial ledger
    from session.ledger import AnswerLedger, LedgerEntry
    state.ledger = AnswerLedger()
    state.ledger.entries["entry_1"] = LedgerEntry("entry_1", "test", [], [], {}, version=1)

    with pytest.raises(ValueError, match="answer_refinement requires FusionPipeline with an active ContradictionScreen"):
        answer_refinement(state, "entry_1", "detail", indexed_dev_corpus, fusion, gen, verif)


# ---------------------------------------------------------------------------
# 2e. Empty-refinement test
# ---------------------------------------------------------------------------

def test_answer_refinement_empty_new_chunks_noop(indexed_dev_corpus):
    chunk = indexed_dev_corpus["chunk_lookup"]["DOC_04_§1"]
    class AlreadyCitedIndex:
        def __getitem__(self, item):
            if item == "chunk_lookup":
                return indexed_dev_corpus["chunk_lookup"]
            if item == "retrieve":
                # Returns only chunk already in entry.evidence
                return lambda query, top_k=10: ["DOC_04_§1"]
            return indexed_dev_corpus[item]

    stub = StubLLM()
    gen = AnswerGenerator(stub)
    verif = GroundingVerifier(stub)
    screen = ContradictionScreen(llm_client=stub)
    fusion = FusionPipeline(contradiction_screen=screen)
    state = SessionState(session_id="test_noop")

    from session.ledger import AnswerLedger, LedgerEntry
    state.ledger = AnswerLedger()
    entry = LedgerEntry(
        entry_id="entry_1",
        topic="cancellation",
        details=[],
        claims=[LedgerClaim("Full refund 30 days.", "DOC_04_§1", origin_version=1, status="ACTIVE")],
        evidence={"DOC_04_§1": chunk},
        version=1,
    )
    state.ledger.entries["entry_1"] = entry
    initial_render = render(entry)

    turn = answer_refinement(state, "entry_1", "more cancellation info", AlreadyCitedIndex(), fusion, gen, verif)
    assert turn.text == initial_render
    assert entry.version == 1
    assert len(state.ledger.history) == 0
    assert turn.llm_calls == 0


# ---------------------------------------------------------------------------
# 2f. Turn 3 Presentation suppression
# ---------------------------------------------------------------------------

def test_turn3_suppress_makes_zero_retrievals(indexed_dev_corpus):
    turn1_payload = json.dumps({"claims": [
        {"claim": "Domestic meals are capped at standard rates.", "chunk_id": "DOC_06_§1"},
    ]})
    turn2_payload = json.dumps({"claims": [
        {"claim": "International travel requires original receipts.", "chunk_id": "DOC_07_§1"},
    ]})
    stub = StubLLM(responses={
        "Detail / Focus: international": turn2_payload,
        "Candidate Chunks:": turn1_payload,
        "Claim: ": json.dumps({"supported": True, "reason": "ok"}),
        "Passage A:": json.dumps({"is_contradiction": False, "reason": "none"}),
    })

    gen = AnswerGenerator(stub)
    verif = GroundingVerifier(stub)
    screen = ContradictionScreen(llm_client=stub)
    fusion = FusionPipeline(contradiction_screen=screen)
    state = SessionState(session_id="test_turn3")

    turn1 = answer_new_topic(state, "travel policy", indexed_dev_corpus, fusion, gen, verif)
    turn2 = answer_refinement(state, "entry_1", "international", indexed_dev_corpus, fusion, gen, verif)
    assert turn2.text != ""

    controller = RuleBasedController()
    dec = controller.decide(state, "repeat that in bullet points", timestamp_s=2.0)
    assert dec.action == "SUPPRESS"

    # Wrap retrieve to ensure zero calls on re-render
    retrieval_calls = 0
    orig_retrieve = indexed_dev_corpus["retrieve"]
    def counting_retrieve(query, top_k=10):
        nonlocal retrieval_calls
        retrieval_calls += 1
        return orig_retrieve(query, top_k=top_k)

    wrapped_index = dict(indexed_dev_corpus)
    wrapped_index["retrieve"] = counting_retrieve

    turn3_text = render(state.ledger.entries["entry_1"])
    assert retrieval_calls == 0
    assert turn3_text == turn2.text


# ---------------------------------------------------------------------------
# 2g. Live Scenario B trace (@pytest.mark.live)
# ---------------------------------------------------------------------------

@pytest.mark.live
def test_live_scenario_b_trace(indexed_dev_corpus):
    real_client = get_llm_client()
    counting_client = CountingLLMClient(real_client)

    gen = AnswerGenerator(counting_client)
    verif = GroundingVerifier(counting_client)
    screen = ContradictionScreen(llm_client=counting_client)
    fusion = FusionPipeline(contradiction_screen=screen)
    state = SessionState(session_id="live_scenario_b")

    # Turn 1: New Topic
    t1_calls_start = counting_client.calls
    turn1 = answer_new_topic(
        state,
        "summarize the travel reimbursement rule",
        indexed_dev_corpus,
        fusion,
        gen,
        verif,
        llm_counter=counting_client,
    )
    t1_calls = counting_client.calls - t1_calls_start
    entry = state.ledger.entries["entry_1"]

    print("\n--- STAGE 4B LIVE TRACE ---")
    print(f"Turn 1 Calls: {t1_calls}")
    print("ACTIVE CLAIMS BEFORE TURN 2:")
    claims_before = active_claims(entry)
    for c in claims_before:
        safe_claim = c.claim.encode("ascii", errors="backslashreplace").decode("ascii")
        print(f"  [{c.chunk_id}] (v{c.origin_version}, {c.status}): {safe_claim}")

    # Turn 2: Refinement
    t2_calls_start = counting_client.calls
    turn2 = answer_refinement(
        state,
        "entry_1",
        "international, booking made after travel",
        indexed_dev_corpus,
        fusion,
        gen,
        verif,
        llm_counter=counting_client,
    )
    t2_calls = counting_client.calls - t2_calls_start

    print(f"\nTurn 2 Calls: {t2_calls}")
    print("ACTIVE CLAIMS AFTER TURN 2:")
    claims_after = active_claims(entry)
    for c in claims_after:
        safe_claim = c.claim.encode("ascii", errors="backslashreplace").decode("ascii")
        print(f"  [{c.chunk_id}] (v{c.origin_version}, {c.status}): {safe_claim}")

    # Invariant: No chunk_id appears in more than one ACTIVE claim within the entry
    active_chunk_ids = [c.chunk_id for c in claims_after]
    assert len(active_chunk_ids) == len(set(active_chunk_ids)), f"Duplicate active chunk_ids found: {active_chunk_ids}"

    # Turn 3: Controller Reformat Check
    controller = RuleBasedController()
    dec = controller.decide(state, "repeat that in bullet points", timestamp_s=3.0)
    assert dec.action == "SUPPRESS"

    retrieval_calls = 0
    orig_retrieve = indexed_dev_corpus["retrieve"]
    def counting_retrieve(query, top_k=10):
        nonlocal retrieval_calls
        retrieval_calls += 1
        return orig_retrieve(query, top_k=top_k)

    wrapped_index = dict(indexed_dev_corpus)
    wrapped_index["retrieve"] = counting_retrieve

    turn3_text = render(entry)
    assert retrieval_calls == 0

    safe_t3_text = turn3_text.encode("ascii", errors="backslashreplace").decode("ascii")
    print(f"\nTurn 3 Calls: 0 (retrieval: {retrieval_calls})")
    print("TURN 3 RENDER TEXT:\n", safe_t3_text)

    print("\nLEDGER HISTORY IN FULL:")
    for h in state.ledger.history:
        print(" ", h)
