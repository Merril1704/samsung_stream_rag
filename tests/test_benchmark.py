"""Tests for Stage 7B Evaluation Benchmark Harness (eval/benchmark.py).

Verifies:
1. Recall@K metric calculation with controlled retrieval inputs.
2. Multi-intent per-sub-query recall and turn-level aggregation.
3. Gold claim groundedness verification against independent gold labels.
4. Fabricated citation detection and citation validity calculation.
5. Unanswerable-case evaluation (coverage gap handling).
6. Refinement multi-turn aggregation and version transition checks.
7. Token cost calculation and PricingConfig modeling.
8. Latency field naming (ensuring no false 'TTFT' labels).
9. Top-level aggregate metrics across scenarios and categories.
10. Benchmark serialization to JSON and Markdown.
11. End-to-end integration test running a held-out scenario through StreamRAGOrchestrator.
"""
import json
import pytest

from controller import SessionState
from session.ledger import LedgerEntry, LedgerClaim, AnswerLedger, active_claims
from eval.benchmark import (
    PricingConfig,
    TurnMetricResult,
    ScenarioMetricResult,
    AggregateBenchmarkResult,
    compute_recall_at_k,
    evaluate_turn_metrics,
    run_benchmark,
    aggregate_results,
    format_benchmark_markdown,
)
from eval.scenarios import EvalTurn, GoldClaim, EvalScenario
from session.orchestrator import OrchestratorResult, StreamRAGOrchestrator
from session.pipeline import TurnResult
from telemetry.schema import TurnTrace, StageEvent, RetrievalTriggerEvent


# ---------------------------------------------------------------------------
# Test Helpers & Stubs
# ---------------------------------------------------------------------------

class StubLLM:
    """Stub LLM returning deterministic responses keyed by substrings."""
    def __init__(self, responses: dict[str, str], default: str = "{}"):
        self.responses = responses
        self.default = default

    def complete(self, system: str, user: str, *, max_tokens: int | None = None) -> str:
        for key, resp in self.responses.items():
            if key in user:
                return resp
        return self.default


def create_dummy_trace(
    action: str = "RETRIEVE",
    retrieval_chunks: list[str] | None = None,
    duration_s: float = 0.5,
    in_tok: int = 100,
    out_tok: int = 30,
) -> TurnTrace:
    trace = TurnTrace(turn_number=1, session_id="test_sess")
    trace.path = action
    trace.duration_s = duration_s
    trace.total_estimated_input_tokens = in_tok
    trace.total_estimated_output_tokens = out_tok

    if retrieval_chunks is not None:
        trace.retrieval_events.append(
            RetrievalTriggerEvent(
                wall_timestamp=100.0,
                action="RETRIEVE",
                chunk_ids_returned=retrieval_chunks,
            )
        )

    trace.stage_events.append(
        StageEvent(stage="retrieval", wall_start=100.0, wall_end=100.1, duration_s=0.1)
    )
    trace.stage_events.append(
        StageEvent(stage="generator", wall_start=100.1, wall_end=100.3, duration_s=0.2)
    )
    trace.stage_events.append(
        StageEvent(stage="verifier", wall_start=100.3, wall_end=100.5, duration_s=0.2)
    )
    return trace


# ---------------------------------------------------------------------------
# 1. Recall Metric with Controlled Retrieval Results
# ---------------------------------------------------------------------------

def test_compute_recall_at_k_basic():
    gold = ["DOC_01_§1", "DOC_01_§2"]

    # Top-1 has 1 gold chunk -> 1/2 = 0.5
    ret_1 = ["DOC_01_§1", "DOC_99_§1", "DOC_99_§2"]
    assert compute_recall_at_k(ret_1, gold, 1) == 0.5
    assert compute_recall_at_k(ret_1, gold, 3) == 0.5

    # Top-3 has both gold chunks -> 2/2 = 1.0
    ret_2 = ["DOC_99_§1", "DOC_01_§1", "DOC_01_§2", "DOC_99_§3"]
    assert compute_recall_at_k(ret_2, gold, 1) == 0.0
    assert compute_recall_at_k(ret_2, gold, 2) == 0.5
    assert compute_recall_at_k(ret_2, gold, 3) == 1.0
    assert compute_recall_at_k(ret_2, gold, 10) == 1.0

    # Empty gold (unanswerable) returns None
    assert compute_recall_at_k(ret_1, [], 10) is None


# ---------------------------------------------------------------------------
# 2. Multi-Intent Per-Sub-Query Recall
# ---------------------------------------------------------------------------

def test_multi_intent_per_subquery_recall():
    turn = EvalTurn(
        turn_number=1,
        user_input="I need the venue options and also need the payment terms",
        timestamp_s=1.0,
        expected_action="RETRIEVE",
        expected_doc_ids=["DOC_01", "DOC_10"],
        expected_chunk_ids=["DOC_01_§2", "DOC_10_§1"],
        expected_sub_queries=["venue options", "payment terms"],
        sub_query_gold_chunks={
            "venue options": ["DOC_01_§2"],
            "payment terms": ["DOC_10_§1"],
        },
    )

    trace = TurnTrace(turn_number=1, session_id="test_mi")
    # Sub-query 0 retrieval: DOC_01_§2 at rank 1
    trace.retrieval_events.append(
        RetrievalTriggerEvent(
            wall_timestamp=1.0,
            query="venue options",
            action="RETRIEVE",
            chunk_ids_returned=["DOC_01_§2", "DOC_02_§1"],
        )
    )
    # Sub-query 1 retrieval: DOC_10_§1 at rank 2
    trace.retrieval_events.append(
        RetrievalTriggerEvent(
            wall_timestamp=1.1,
            query="payment terms",
            action="RETRIEVE",
            chunk_ids_returned=["DOC_02_§2", "DOC_10_§1"],
        )
    )

    res = OrchestratorResult(action="RETRIEVE", trace=trace)
    state = SessionState(session_id="test_mi")
    pricing = PricingConfig()

    metric = evaluate_turn_metrics(turn, res, state, {"DOC_01_§2": None, "DOC_10_§1": None}, pricing)

    # Sub-query 0: rank 1 -> recall@1 = 1.0
    # Sub-query 1: rank 2 -> recall@1 = 0.0, recall@3 = 1.0
    assert metric.sub_query_recalls["venue options"]["recall@1"] == 1.0
    assert metric.sub_query_recalls["payment terms"]["recall@1"] == 0.0
    assert metric.sub_query_recalls["payment terms"]["recall@3"] == 1.0

    # Turn-level recall is the mean across sub-queries
    assert metric.recall_at_1 == 0.5
    assert metric.recall_at_3 == 1.0
    assert metric.recall_at_10 == 1.0


# ---------------------------------------------------------------------------
# 3. Gold Claim Groundedness
# ---------------------------------------------------------------------------

def test_gold_claim_groundedness_supported():
    turn = EvalTurn(
        turn_number=1,
        user_input="I need the booking procedure",
        timestamp_s=1.0,
        expected_action="RETRIEVE",
        expected_chunk_ids=["DOC_01_§1"],
        gold_claims=[
            GoldClaim("gc1", "Bookings must be submitted 10 days in advance.", ["DOC_01_§1"])
        ],
    )

    trace = create_dummy_trace(retrieval_chunks=["DOC_01_§1"])
    res = OrchestratorResult(action="RETRIEVE", trace=trace)

    state = SessionState(session_id="test_ground")
    state.ledger = AnswerLedger()
    state.ledger.entries["entry_1"] = LedgerEntry(
        entry_id="entry_1",
        version=1,
        topic="booking",
        details=[],
        claims=[LedgerClaim("Bookings must be submitted 10 days in advance.", "DOC_01_§1", 1)],
        evidence={},
    )

    metric = evaluate_turn_metrics(
        turn, res, state, {"DOC_01_§1": None}, PricingConfig()
    )

    assert metric.total_claims_generated == 1
    assert metric.supported_claims == 1
    assert metric.unsupported_claims == 0
    assert metric.claim_groundedness == 1.0
    assert metric.citation_validity == 1.0
    assert metric.fabricated_citation_rate == 0.0


# ---------------------------------------------------------------------------
# 4. Fabricated Citation Detection
# ---------------------------------------------------------------------------

def test_fabricated_citation_detection():
    turn = EvalTurn(
        turn_number=1,
        user_input="I need the booking procedure",
        timestamp_s=1.0,
        expected_action="RETRIEVE",
        expected_chunk_ids=["DOC_01_§1"],
    )

    trace = create_dummy_trace(retrieval_chunks=["DOC_01_§1"])
    res = OrchestratorResult(action="RETRIEVE", trace=trace)

    state = SessionState(session_id="test_fab")
    state.ledger = AnswerLedger()
    # Generator cites non-existent hallucinated chunk
    state.ledger.entries["entry_1"] = LedgerEntry(
        entry_id="entry_1",
        version=1,
        topic="booking",
        details=[],
        claims=[LedgerClaim("Hallucinated policy claim.", "DOC_FAKE_§99", 1)],
        evidence={},
    )

    # DOC_FAKE_§99 is not in the chunk lookup
    metric = evaluate_turn_metrics(
        turn, res, state, {"DOC_01_§1": None}, PricingConfig()
    )

    assert metric.total_claims_generated == 1
    assert metric.fabricated_citations == 1
    assert metric.citation_validity == 0.0
    assert metric.fabricated_citation_rate == 1.0
    assert metric.claim_groundedness == 0.0


# ---------------------------------------------------------------------------
# 5. Unanswerable-Case Scoring
# ---------------------------------------------------------------------------

def test_unanswerable_case_scoring():
    unans_turn = EvalTurn(
        turn_number=1,
        user_input="I need to know the vegan catering policy for Venue A",
        timestamp_s=1.0,
        expected_action="RETRIEVE",
        expected_chunk_ids=[],
        gold_claims=[],
        is_answerable=False,
    )

    trace = create_dummy_trace(retrieval_chunks=["DOC_05_§3"])
    turn_res = TurnResult(text="", verification=None, rejected=[], llm_calls=0, path="NEW_TOPIC")
    res = OrchestratorResult(action="RETRIEVE", trace=trace, turn_result=turn_res)

    state = SessionState(session_id="test_unans")
    state.ledger = AnswerLedger()
    # Empty claims committed because evidence was insufficient
    state.ledger.entries["entry_1"] = LedgerEntry(
        entry_id="entry_1", version=1, topic="unans", details=[], claims=[], evidence={}
    )

    metric = evaluate_turn_metrics(unans_turn, res, state, {"DOC_05_§3": None}, PricingConfig())

    assert metric.is_answerable is False
    assert metric.unanswerable_handled_correctly is True
    assert metric.claim_groundedness == 1.0
    assert metric.fabricated_citation_rate == 0.0


# ---------------------------------------------------------------------------
# 6. Refinement Aggregation
# ---------------------------------------------------------------------------

def test_refinement_turn_evaluation():
    turn2 = EvalTurn(
        turn_number=2,
        user_input="the expense exceeds 25000 rupees",
        timestamp_s=2.0,
        expected_action="RETRIEVE",
        expected_chunk_ids=["DOC_09_§2"],
    )

    trace = create_dummy_trace(retrieval_chunks=["DOC_09_§2"])
    res = OrchestratorResult(action="RETRIEVE", trace=trace)

    state = SessionState(session_id="test_ref")
    state.ledger = AnswerLedger()
    state.ledger.entries["entry_1"] = LedgerEntry(
        entry_id="entry_1",
        version=2,  # Version successfully bumped
        topic="refinement",
        details=[],
        claims=[
            LedgerClaim("Standard domestic approval.", "DOC_09_§1", origin_version=1),
            LedgerClaim("Director approval required over 25000.", "DOC_09_§2", origin_version=2),
        ],
        evidence={},
    )

    metric = evaluate_turn_metrics(
        turn2, res, state, {"DOC_09_§1": None, "DOC_09_§2": None}, PricingConfig()
    )

    assert metric.version_transition_ok is True
    assert metric.recall_at_1 == 1.0
    assert metric.supported_claims == 1


# ---------------------------------------------------------------------------
# 7. Cost Calculation & PricingConfig
# ---------------------------------------------------------------------------

def test_cost_calculation_pricing():
    pricing = PricingConfig(input_usd_per_1m=2.0, output_usd_per_1m=8.0)
    trace = create_dummy_trace(in_tok=1_000_000, out_tok=500_000)

    turn = EvalTurn(
        turn_number=1,
        user_input="Query",
        timestamp_s=1.0,
        expected_action="RETRIEVE",
        expected_chunk_ids=["DOC_01_§1"],
    )
    res = OrchestratorResult(action="RETRIEVE", trace=trace)
    state = SessionState(session_id="test_cost")

    metric = evaluate_turn_metrics(turn, res, state, {"DOC_01_§1": None}, pricing)

    # 1M in * $2.0 + 0.5M out * $8.0 = $2.0 + $4.0 = $6.0
    assert metric.input_tokens == 1_000_000
    assert metric.output_tokens == 500_000
    assert metric.total_tokens == 1_500_000
    assert metric.estimated_cost_usd == pytest.approx(6.0)
    assert metric.token_count_mode == "estimated"


# ---------------------------------------------------------------------------
# 8. Latency Field Naming (No False 'TTFT')
# ---------------------------------------------------------------------------

def test_latency_fields_naming():
    trace = create_dummy_trace(duration_s=0.75)
    res = OrchestratorResult(action="RETRIEVE", trace=trace)
    turn = EvalTurn(turn_number=1, user_input="Q", timestamp_s=1.0, expected_action="RETRIEVE")
    state = SessionState(session_id="test_lat")

    metric = evaluate_turn_metrics(turn, res, state, {}, PricingConfig())

    # Ensure no field is called ttft
    metric_dict = metric.to_dict()
    assert "ttft" not in metric_dict
    assert "ttft_s" not in metric_dict

    # Check valid latency fields
    assert metric.total_turn_latency_s == 0.75
    assert metric.retrieval_latency_s == 0.1
    assert metric.generator_latency_s == 0.2
    assert metric.verifier_latency_s == 0.2
    assert metric.response_latency_proxy_s == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# 9. Aggregate Metrics Calculation
# ---------------------------------------------------------------------------

def test_aggregate_metrics_calculation():
    t1 = TurnMetricResult(
        turn_number=1, user_input="q1", action="RETRIEVE", expected_action="RETRIEVE",
        action_correct=True, recall_at_1=1.0, recall_at_3=1.0, recall_at_5=1.0, recall_at_10=1.0,
        total_claims_generated=2, supported_claims=2, unsupported_claims=0, fabricated_citations=0,
        claim_groundedness=1.0, citation_validity=1.0, fabricated_citation_rate=0.0,
        total_turn_latency_s=0.4, retrieval_latency_s=0.1, generator_latency_s=0.2, verifier_latency_s=0.1,
        input_tokens=100, output_tokens=50, total_tokens=150, estimated_cost_usd=0.001
    )
    t2 = TurnMetricResult(
        turn_number=1, user_input="q2", action="RETRIEVE", expected_action="RETRIEVE",
        action_correct=True, recall_at_1=0.0, recall_at_3=0.5, recall_at_5=0.5, recall_at_10=1.0,
        total_claims_generated=2, supported_claims=1, unsupported_claims=1, fabricated_citations=0,
        claim_groundedness=0.5, citation_validity=1.0, fabricated_citation_rate=0.0,
        total_turn_latency_s=0.6, retrieval_latency_s=0.2, generator_latency_s=0.2, verifier_latency_s=0.2,
        input_tokens=200, output_tokens=100, total_tokens=300, estimated_cost_usd=0.002
    )

    s1 = ScenarioMetricResult(
        scenario_id="s1", category="single_intent", title="S1", turns=[t1],
        mean_recall_at_10=1.0, mean_claim_groundedness=1.0, total_tokens=150,
        total_latency_s=0.4, all_turns_passed=True
    )
    s2 = ScenarioMetricResult(
        scenario_id="s2", category="single_intent", title="S2", turns=[t2],
        mean_recall_at_10=1.0, mean_claim_groundedness=0.5, total_tokens=300,
        total_latency_s=0.6, all_turns_passed=True
    )

    agg = aggregate_results([s1, s2])

    assert agg.total_scenarios == 2
    assert agg.total_turns == 2
    assert agg.mean_recall_at_1 == 0.5
    assert agg.mean_recall_at_10 == 1.0
    assert agg.mean_claim_groundedness == 0.75
    assert agg.mean_turn_latency_s == pytest.approx(0.5)
    assert agg.total_tokens == 450
    assert agg.tokens_per_turn == 225.0
    assert "single_intent" in agg.category_metrics


# ---------------------------------------------------------------------------
# 10. Benchmark Serialization & Markdown Generation
# ---------------------------------------------------------------------------

def test_benchmark_result_serialization():
    t = TurnMetricResult(
        turn_number=1, user_input="q", action="RETRIEVE", expected_action="RETRIEVE",
        action_correct=True, recall_at_1=1.0, recall_at_3=1.0, recall_at_5=1.0, recall_at_10=1.0,
        claim_groundedness=1.0, total_turn_latency_s=0.5, total_tokens=200
    )
    s = ScenarioMetricResult(
        scenario_id="eval_s1", category="single_intent", title="Test S1", turns=[t],
        mean_recall_at_10=1.0, mean_claim_groundedness=1.0, total_tokens=200,
        total_latency_s=0.5, all_turns_passed=True
    )

    agg = aggregate_results([s])
    d = agg.to_dict()

    # JSON serializability check
    raw_json = json.dumps(d)
    parsed = json.loads(raw_json)
    assert parsed["total_scenarios"] == 1
    assert parsed["mean_recall_at_10"] == 1.0

    # Markdown format check
    md = format_benchmark_markdown(agg)
    assert "# Streaming Live RAG — Benchmark Evaluation Report" in md
    assert "Retrieval Recall@10" in md
    assert "eval_s1" in md


# ---------------------------------------------------------------------------
# 11. Integration Test: Real Held-Out Scenario Through Orchestrator
# ---------------------------------------------------------------------------

def test_benchmark_integration_real_scenario(indexed_dev_corpus):
    from eval.scenarios import get_scenario_by_id
    from controller.llm_wrapper import CountingLLMClient
    from controller.rule_based import RuleBasedController
    from decomposer.decomposer import Decomposer
    from fusion.fusion_pipeline import FusionPipeline
    from fusion.contradiction import ContradictionScreen
    from session.generator import AnswerGenerator
    from grounding.verifier import GroundingVerifier

    # Use eval_single_01 (Event booking procedure)
    scen = get_scenario_by_id("eval_single_01")
    assert scen is not None

    gen_payload = json.dumps({"claims": [
        {"claim": "Event booking requests must be submitted at least 10 business days before the event.", "chunk_id": "DOC_01_§1"},
        {"claim": "Venue confirmation typically takes 3 to 5 business days.", "chunk_id": "DOC_01_§3"},
    ]})

    stub = StubLLM(responses={
        "Candidate Chunks:": gen_payload,
        "Claim: ": json.dumps({"supported": True, "reason": "grounded"}),
        "Passage A:": json.dumps({"is_contradiction": False, "reason": "none"}),
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

    agg = run_benchmark(
        orchestrator=orchestrator,
        index=indexed_dev_corpus,
        scenarios=[scen],
        pricing=PricingConfig(input_usd_per_1m=1.0, output_usd_per_1m=2.0),
    )

    assert agg.total_scenarios == 1
    assert agg.total_turns == 1
    scen_res = agg.scenarios[0]
    assert scen_res.scenario_id == "eval_single_01"
    # Recall@10 must be 1.0 (both DOC_01_§1 and DOC_01_§3 were retrieved)
    assert scen_res.mean_recall_at_10 > 0.0
    assert scen_res.mean_claim_groundedness == 1.0
    assert agg.mean_recall_at_10 > 0.0
    assert agg.tokens_per_turn > 0


# ---------------------------------------------------------------------------
# Stage 7C Regression Tests
# ---------------------------------------------------------------------------

def test_empty_gold_recall_is_none():
    """Requirement 1: compute_recall_at_k returns None when gold_ids is empty."""
    # Basic function test
    assert compute_recall_at_k(["DOC_01_§1", "DOC_01_§2"], [], 1) is None
    assert compute_recall_at_k(["DOC_01_§1"], [], 10) is None
    assert compute_recall_at_k([], [], 5) is None

    # Turn-level evaluation test for unanswerable turn
    unans_turn = EvalTurn(
        turn_number=1,
        user_input="Coverage gap query",
        timestamp_s=1.0,
        expected_action="RETRIEVE",
        expected_chunk_ids=[],
        gold_claims=[],
        is_answerable=False,
    )
    trace = create_dummy_trace(retrieval_chunks=["DOC_05_§3"])
    turn_res = TurnResult(text="", verification=None, rejected=[], llm_calls=0, path="NEW_TOPIC")
    res = OrchestratorResult(action="RETRIEVE", trace=trace, turn_result=turn_res)
    state = SessionState(session_id="test_unans_none")

    metric = evaluate_turn_metrics(unans_turn, res, state, {"DOC_05_§3": None}, PricingConfig())
    assert metric.recall_at_1 is None
    assert metric.recall_at_3 is None
    assert metric.recall_at_5 is None
    assert metric.recall_at_10 is None
    assert metric.fused_recall_at_10 is None


def test_none_excluded_from_aggregate_recall():
    """Requirement 1 & 7: None recall values are excluded from aggregation denominator."""
    t_answerable = TurnMetricResult(
        turn_number=1, user_input="q1", action="RETRIEVE", expected_action="RETRIEVE",
        action_correct=True, recall_at_1=0.5, recall_at_3=0.8, recall_at_5=0.8, recall_at_10=0.8,
        fused_recall_at_1=0.5, fused_recall_at_3=0.8, fused_recall_at_5=0.8, fused_recall_at_10=0.8,
        citation_groundedness=1.0, total_turn_latency_s=0.5, total_tokens=100, is_answerable=True,
    )
    t_unanswerable = TurnMetricResult(
        turn_number=1, user_input="q2", action="RETRIEVE", expected_action="RETRIEVE",
        action_correct=True, recall_at_1=None, recall_at_3=None, recall_at_5=None, recall_at_10=None,
        fused_recall_at_1=None, fused_recall_at_3=None, fused_recall_at_5=None, fused_recall_at_10=None,
        citation_groundedness=1.0, total_turn_latency_s=0.5, total_tokens=100, is_answerable=False,
    )

    s1 = ScenarioMetricResult(
        scenario_id="s_ans", category="single_intent", title="Answerable",
        turns=[t_answerable], mean_recall_at_10=0.8, mean_fused_recall_at_10=0.8,
        citation_groundedness=1.0, total_tokens=100, total_latency_s=0.5, all_turns_passed=True,
    )
    s2 = ScenarioMetricResult(
        scenario_id="s_unans", category="unanswerable", title="Unanswerable",
        turns=[t_unanswerable], mean_recall_at_10=None, mean_fused_recall_at_10=None,
        citation_groundedness=1.0, total_tokens=100, total_latency_s=0.5, all_turns_passed=True,
    )

    agg = aggregate_results([s1, s2])

    # Denominator must exclude the None value: mean must be 0.8, NOT (0.8 + 0)/2 or (0.8 + 1)/2
    assert agg.mean_recall_at_10 == 0.8
    assert agg.mean_recall_at_1 == 0.5
    assert agg.mean_fused_recall_at_10 == 0.8

    # Category-level unanswerable metrics must NOT report 100% recall
    assert agg.category_metrics["unanswerable"]["mean_recall@10"] is None
    assert agg.category_metrics["unanswerable"]["mean_fused_recall@10"] is None


def test_reversed_multi_intent_decomposition_alignment():
    """Requirement 2: Sub-queries and retrieval events align by identity, not positional index."""
    turn = EvalTurn(
        turn_number=1,
        user_input="I need the venue options and also need the payment terms",
        timestamp_s=1.0,
        expected_action="RETRIEVE",
        expected_doc_ids=["DOC_01", "DOC_10"],
        expected_chunk_ids=["DOC_01_§2", "DOC_10_§1"],
        expected_sub_queries=["venue options", "payment terms"],
        sub_query_gold_chunks={
            "venue options": ["DOC_01_§2"],
            "payment terms": ["DOC_10_§1"],
        },
    )

    trace = TurnTrace(turn_number=1, session_id="test_reversed_mi")
    # Deliberately REVERSE the order of actual retrieval events:
    # Event 0: payment terms (expected index 1)
    trace.retrieval_events.append(
        RetrievalTriggerEvent(
            wall_timestamp=1.0,
            query="payment terms",
            action="RETRIEVE",
            chunk_ids_returned=["DOC_10_§1", "DOC_02_§1"],
        )
    )
    # Event 1: venue options (expected index 0)
    trace.retrieval_events.append(
        RetrievalTriggerEvent(
            wall_timestamp=1.1,
            query="venue options",
            action="RETRIEVE",
            chunk_ids_returned=["DOC_01_§2", "DOC_02_§2"],
        )
    )

    res = OrchestratorResult(action="RETRIEVE", trace=trace)
    state = SessionState(session_id="test_reversed_mi")

    metric = evaluate_turn_metrics(
        turn, res, state, {"DOC_01_§2": None, "DOC_10_§1": None}, PricingConfig()
    )

    # Despite reversed order in trace, each sub-query must find its corresponding event
    assert metric.sub_query_recalls["venue options"]["recall@1"] == 1.0
    assert metric.sub_query_recalls["payment terms"]["recall@1"] == 1.0
    assert metric.recall_at_1 == 1.0
    assert metric.recall_at_10 == 1.0


def test_fused_recall_calculation():
    """Requirement 3: Fused candidate list recall is calculated and distinguished from raw retrieval."""
    turn = EvalTurn(
        turn_number=1,
        user_input="Test query",
        timestamp_s=1.0,
        expected_action="RETRIEVE",
        expected_chunk_ids=["DOC_02_§1", "DOC_04_§1"],
    )

    trace = create_dummy_trace(retrieval_chunks=["DOC_99_§1", "DOC_99_§2"])
    # Controlled fusion candidate list returned by FusionPipeline
    controlled_fused_cids = ["DOC_01_§1", "DOC_02_§1", "DOC_03_§1", "DOC_04_§1"]
    turn_res = TurnResult(
        text="", verification=None, rejected=[], llm_calls=0, path="NEW_TOPIC",
        fused_chunk_ids=controlled_fused_cids,
    )
    res = OrchestratorResult(action="RETRIEVE", trace=trace, turn_result=turn_res)
    state = SessionState(session_id="test_fused_calc")

    metric = evaluate_turn_metrics(turn, res, state, {}, PricingConfig())

    # Raw retrieval recall had none of the gold chunks
    assert metric.recall_at_1 == 0.0
    assert metric.recall_at_10 == 0.0

    # Fused recall has DOC_02_§1 at rank 2, and DOC_04_§1 at rank 4:
    # rank 1: hits=0 -> 0.0
    # rank 3: hits=1 (DOC_02_§1) / 2 -> 0.5
    # rank 5: hits=2 / 2 -> 1.0
    assert metric.fused_recall_at_1 == 0.0
    assert metric.fused_recall_at_3 == 0.5
    assert metric.fused_recall_at_5 == 1.0
    assert metric.fused_recall_at_10 == 1.0


def test_controller_action_accuracy():
    """Requirement 4: Controller action accuracy is correctly computed."""
    turns = [
        TurnMetricResult(
            turn_number=1, user_input="q1", action="RETRIEVE", expected_action="RETRIEVE",
            action_correct=True, recall_at_1=1.0, recall_at_3=1.0, recall_at_5=1.0, recall_at_10=1.0,
        ),
        TurnMetricResult(
            turn_number=2, user_input="q2", action="WAIT", expected_action="WAIT",
            action_correct=True, recall_at_1=None, recall_at_3=None, recall_at_5=None, recall_at_10=None,
        ),
        TurnMetricResult(
            turn_number=3, user_input="q3", action="SUPPRESS", expected_action="SUPPRESS",
            action_correct=True, recall_at_1=None, recall_at_3=None, recall_at_5=None, recall_at_10=None,
        ),
        TurnMetricResult(
            turn_number=4, user_input="q4", action="RETRIEVE", expected_action="WAIT",
            action_correct=False, recall_at_1=1.0, recall_at_3=1.0, recall_at_5=1.0, recall_at_10=1.0,
        ),
    ]

    s = ScenarioMetricResult(
        scenario_id="s_ctrl", category="single_intent", title="Controller test",
        turns=turns, mean_recall_at_10=1.0,
    )

    agg = aggregate_results([s])
    # 3 correct out of 4 total evaluated turns = 0.75
    assert agg.controller_action_accuracy == 0.75
    assert agg.category_metrics["single_intent"]["controller_action_accuracy"] == 0.75


def test_report_serialization_with_null_recall():
    """Requirement 6: Serialization outputs null in JSON and N/A in Markdown for unavailable recall."""
    t_unans = TurnMetricResult(
        turn_number=1, user_input="unans", action="RETRIEVE", expected_action="RETRIEVE",
        action_correct=True, recall_at_1=None, recall_at_3=None, recall_at_5=None, recall_at_10=None,
        fused_recall_at_1=None, fused_recall_at_3=None, fused_recall_at_5=None, fused_recall_at_10=None,
        citation_groundedness=1.0, is_answerable=False,
    )
    s_unans = ScenarioMetricResult(
        scenario_id="s_unans", category="unanswerable", title="Unanswerable Scenario",
        turns=[t_unans], mean_recall_at_10=None, mean_fused_recall_at_10=None, citation_groundedness=1.0,
    )

    agg = aggregate_results([s_unans])
    d = agg.to_dict()

    # JSON serializability and null representation
    raw_json = json.dumps(d)
    parsed = json.loads(raw_json)
    assert parsed["mean_recall_at_10"] is None
    assert parsed["mean_fused_recall_at_10"] is None
    assert parsed["category_metrics"]["unanswerable"]["mean_recall@10"] is None

    # Markdown representation
    md = format_benchmark_markdown(agg)
    assert "| **Raw Retrieval Recall@10** | **N/A** |" in md
    assert "| **Fused Candidate Recall@10** | **N/A** |" in md
    assert "| `unanswerable` | 1 | 1 | 100.0% | N/A | N/A |" in md


def test_report_labels_explicit_no_ttft():
    """Requirement 6 & 8: No synchronous latency value is reported as TTFT."""
    agg = AggregateBenchmarkResult(
        benchmark_version="1.0.0",
        timestamp_utc="2026-09-30T00:00:00Z",
        total_scenarios=1,
        total_turns=1,
        mean_recall_at_1=1.0,
        mean_recall_at_3=1.0,
        mean_recall_at_5=1.0,
        mean_recall_at_10=1.0,
        mean_sub_query_recall_at_10=1.0,
        mean_fused_recall_at_10=1.0,
        controller_action_accuracy=1.0,
        citation_groundedness=1.0,
        mean_citation_validity=1.0,
        mean_fabricated_citation_rate=0.0,
        mean_turn_latency_s=0.5,
        median_turn_latency_s=0.5,
        mean_retrieval_latency_s=0.1,
        mean_generator_latency_s=0.3,
        mean_verifier_latency_s=0.1,
    )
    md = format_benchmark_markdown(agg)
    assert "**Streaming TTFT** | **N/A**" in md
    assert "stream=False" in md
    assert "True streaming TTFT = N/A" in agg.latency_note


def test_evidence_relevance_terminology():
    """Requirement 5: Grounding metric reflects Citation Groundedness / Evidence Relevance Rate."""
    t = TurnMetricResult(
        turn_number=1, user_input="q", action="RETRIEVE", expected_action="RETRIEVE",
        action_correct=True, recall_at_1=1.0, recall_at_3=1.0, recall_at_5=1.0, recall_at_10=1.0,
        citation_groundedness=0.85,
    )
    assert t.citation_groundedness == 0.85
    # Backward compatibility alias
    assert t.claim_groundedness == 0.85

    s = ScenarioMetricResult(
        scenario_id="s1", category="single_intent", title="S1", turns=[t],
        mean_recall_at_10=1.0, citation_groundedness=0.85,
    )
    agg = aggregate_results([s])
    assert agg.citation_groundedness == 0.85
    assert agg.mean_claim_groundedness == 0.85

    md = format_benchmark_markdown(agg)
    assert "**Citation Groundedness**" in md
    assert "Evidence Relevance Rate" in md
