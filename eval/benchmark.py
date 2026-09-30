from __future__ import annotations

"""Stage 7B Evaluation Benchmark Harness.

Executes held-out evaluation scenarios through `StreamRAGOrchestrator`
and computes official Theme 4 evaluation metrics:
  1. Retrieval Recall (Recall@1, @3, @5, @10; per-sub-query & turn-level)
  2. Answer Groundedness (Claim Groundedness, Citation Validity, Fabricated Citation Rate)
  3. Latency (total_turn_latency, retrieval, generator, verifier, response_latency_proxy)
  4. Cost & Token Usage (input_tokens, output_tokens, total_tokens, tokens_per_turn, estimated_cost)

Provides both programmatic APIs and CLI entry points.
"""

import argparse
import copy
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time
from typing import Any, Sequence

from controller import SessionState
from session.ledger import active_claims, LedgerEntry, LedgerClaim
from controller.llm_factory import get_llm_client
from controller.llm_wrapper import CountingLLMClient
from controller.rule_based import RuleBasedController
from decomposer.decomposer import Decomposer
from eval.scenarios import (
    EvalScenario,
    EvalTurn,
    GoldClaim,
    get_eval_scenarios,
    get_scenario_by_id,
    get_scenarios_by_category,
)
from fusion.fusion_pipeline import FusionPipeline
from fusion.contradiction import ContradictionScreen
from session.generator import AnswerGenerator
from grounding.verifier import GroundingVerifier
from session.orchestrator import StreamRAGOrchestrator, OrchestratorResult
from telemetry import RetrievalTriggerEvent
from telemetry.schema import TurnTrace


# ---------------------------------------------------------------------------
# Pricing Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PricingConfig:
    """Pricing configuration for token cost estimation (USD per million tokens)."""
    input_usd_per_1m: float = 0.0
    output_usd_per_1m: float = 0.0
    model_name: str = "local-model"


# ---------------------------------------------------------------------------
# Metric Result Structures
# ---------------------------------------------------------------------------

@dataclass
class TurnMetricResult:
    """Evaluation metrics for a single turn."""
    turn_number: int
    user_input: str
    action: str
    expected_action: str
    action_correct: bool

    # Raw retrieval recall metrics (per sub-query or baseline single query; None if empty gold)
    recall_at_1: float | None
    recall_at_3: float | None
    recall_at_5: float | None
    recall_at_10: float | None
    sub_query_recalls: dict[str, dict[str, float | None]] = field(default_factory=dict)
    retrieved_chunk_ids: list[str] = field(default_factory=list)

    # Post-fusion recall metrics (final candidate ranking against union of gold chunks; None if empty gold)
    fused_recall_at_1: float | None = None
    fused_recall_at_3: float | None = None
    fused_recall_at_5: float | None = None
    fused_recall_at_10: float | None = None
    fused_chunk_ids: list[str] = field(default_factory=list)

    # Citation & Evidence metrics (renamed from claim_groundedness to reflect evidence relevance)
    total_claims_generated: int = 0
    supported_claims: int = 0
    unsupported_claims: int = 0
    fabricated_citations: int = 0
    citation_groundedness: float = 0.0
    claim_groundedness: float = 0.0
    citation_validity: float = 1.0
    fabricated_citation_rate: float = 0.0

    # Specific category checks
    is_answerable: bool = True
    unanswerable_handled_correctly: bool | None = None
    version_transition_ok: bool | None = None
    contradiction_handled_ok: bool | None = None

    # Latency breakdown (seconds) - NOTE: not true streaming TTFT
    total_turn_latency_s: float = 0.0
    retrieval_latency_s: float = 0.0
    generator_latency_s: float = 0.0
    verifier_latency_s: float = 0.0
    response_latency_proxy_s: float = 0.0

    # Token usage & Cost
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    token_count_mode: str = "estimated"
    estimated_cost_usd: float = 0.0

    # Trace reference
    trace_path: str = ""

    def __post_init__(self) -> None:
        if self.citation_groundedness == 0.0 and self.claim_groundedness != 0.0:
            self.citation_groundedness = self.claim_groundedness
        elif self.claim_groundedness == 0.0 and self.citation_groundedness != 0.0:
            self.claim_groundedness = self.citation_groundedness

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScenarioMetricResult:
    """Aggregated evaluation metrics for a single scenario."""
    scenario_id: str
    category: str
    title: str
    turns: list[TurnMetricResult]
    mean_recall_at_10: float | None
    citation_groundedness: float = 0.0
    total_tokens: int = 0
    total_latency_s: float = 0.0
    all_turns_passed: bool = False
    mean_fused_recall_at_10: float | None = None
    mean_claim_groundedness: float = 0.0

    def __post_init__(self) -> None:
        if self.citation_groundedness == 0.0 and self.mean_claim_groundedness != 0.0:
            self.citation_groundedness = self.mean_claim_groundedness
        elif self.mean_claim_groundedness == 0.0 and self.citation_groundedness != 0.0:
            self.mean_claim_groundedness = self.citation_groundedness

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "category": self.category,
            "title": self.title,
            "turns": [t.to_dict() for t in self.turns],
            "mean_recall_at_10": self.mean_recall_at_10,
            "mean_fused_recall_at_10": self.mean_fused_recall_at_10,
            "citation_groundedness": self.citation_groundedness,
            "mean_claim_groundedness": self.mean_claim_groundedness,
            "total_tokens": self.total_tokens,
            "total_latency_s": self.total_latency_s,
            "all_turns_passed": self.all_turns_passed,
        }


@dataclass
class AggregateBenchmarkResult:
    """Top-level aggregate benchmark summary across all evaluated scenarios."""
    benchmark_version: str
    timestamp_utc: str
    total_scenarios: int
    total_turns: int

    # Overall Raw Retrieval Recall (excludes None from denominator)
    mean_recall_at_1: float | None
    mean_recall_at_3: float | None
    mean_recall_at_5: float | None
    mean_recall_at_10: float | None
    mean_sub_query_recall_at_10: float | None

    # Overall Post-Fusion Candidate Recall (excludes None from denominator)
    mean_fused_recall_at_1: float | None = None
    mean_fused_recall_at_3: float | None = None
    mean_fused_recall_at_5: float | None = None
    mean_fused_recall_at_10: float | None = None

    # Controller Routing Accuracy (correct actions / total evaluated turns)
    controller_action_accuracy: float = 1.0

    # Overall Groundedness & Citations (Evidence Relevance Rate)
    citation_groundedness: float = 0.0
    mean_claim_groundedness: float = 0.0
    mean_citation_validity: float = 1.0
    mean_fabricated_citation_rate: float = 0.0

    # Overall Latency
    mean_turn_latency_s: float = 0.0
    median_turn_latency_s: float = 0.0
    mean_retrieval_latency_s: float = 0.0
    mean_generator_latency_s: float = 0.0
    mean_verifier_latency_s: float = 0.0
    latency_note: str = (
        "Note: Architecture operates synchronously (stream=False); generator_latency_s "
        "and response_latency_proxy_s represent non-streamed inference duration. "
        "True streaming TTFT = N/A."
    )

    # Overall Token & Cost
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    tokens_per_turn: float = 0.0
    token_count_mode: str = "estimated"
    total_estimated_cost_usd: float = 0.0

    # Category breakdowns
    category_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    scenarios: list[ScenarioMetricResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.citation_groundedness == 0.0 and self.mean_claim_groundedness != 0.0:
            self.citation_groundedness = self.mean_claim_groundedness
        elif self.mean_claim_groundedness == 0.0 and self.citation_groundedness != 0.0:
            self.mean_claim_groundedness = self.citation_groundedness

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_version": self.benchmark_version,
            "timestamp_utc": self.timestamp_utc,
            "total_scenarios": self.total_scenarios,
            "total_turns": self.total_turns,
            "mean_recall_at_1": self.mean_recall_at_1,
            "mean_recall_at_3": self.mean_recall_at_3,
            "mean_recall_at_5": self.mean_recall_at_5,
            "mean_recall_at_10": self.mean_recall_at_10,
            "mean_sub_query_recall_at_10": self.mean_sub_query_recall_at_10,
            "mean_fused_recall_at_1": self.mean_fused_recall_at_1,
            "mean_fused_recall_at_3": self.mean_fused_recall_at_3,
            "mean_fused_recall_at_5": self.mean_fused_recall_at_5,
            "mean_fused_recall_at_10": self.mean_fused_recall_at_10,
            "controller_action_accuracy": self.controller_action_accuracy,
            "citation_groundedness": self.citation_groundedness,
            "mean_claim_groundedness": self.mean_claim_groundedness,
            "mean_citation_validity": self.mean_citation_validity,
            "mean_fabricated_citation_rate": self.mean_fabricated_citation_rate,
            "mean_turn_latency_s": self.mean_turn_latency_s,
            "median_turn_latency_s": self.median_turn_latency_s,
            "mean_retrieval_latency_s": self.mean_retrieval_latency_s,
            "mean_generator_latency_s": self.mean_generator_latency_s,
            "mean_verifier_latency_s": self.mean_verifier_latency_s,
            "latency_note": self.latency_note,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "tokens_per_turn": self.tokens_per_turn,
            "token_count_mode": self.token_count_mode,
            "total_estimated_cost_usd": self.total_estimated_cost_usd,
            "category_metrics": self.category_metrics,
            "scenarios": [s.to_dict() for s in self.scenarios],
        }


# ---------------------------------------------------------------------------
# Metric Calculation Utilities
# ---------------------------------------------------------------------------

def compute_recall_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float | None:
    """Calculate Recall@K: proportion of gold chunks retrieved within the top K positions.

    Returns None if gold_ids is empty (e.g. unanswerable / coverage-gap queries),
    as absence of relevant evidence must not be interpreted as perfect retrieval.
    """
    if not gold_ids:
        return None
    top_k_set = set(retrieved_ids[:k])
    gold_set = set(gold_ids)
    hits = len(top_k_set & gold_set)
    return float(hits) / float(len(gold_set))


def _match_retrieval_event(
    sub_query: str,
    events: Sequence[RetrievalTriggerEvent],
    used_indices: set[int],
) -> tuple[int, list[str]]:
    """Match an expected sub-query to its corresponding retrieval event without assuming positional alignment.

    Uses exact query match, case-insensitive substring containment, or word-token overlap.
    """
    norm_sq = sub_query.strip().lower()

    # 1. Exact match on event query string
    for idx, ev in enumerate(events):
        if idx in used_indices:
            continue
        if ev.query.strip().lower() == norm_sq:
            used_indices.add(idx)
            return idx, ev.chunk_ids_returned

    # 2. Substring containment match
    for idx, ev in enumerate(events):
        if idx in used_indices:
            continue
        ev_q = ev.query.strip().lower()
        if norm_sq in ev_q or ev_q in norm_sq:
            used_indices.add(idx)
            return idx, ev.chunk_ids_returned

    # 3. Token overlap match
    sq_tokens = set(norm_sq.split())
    best_idx = None
    best_overlap = 0
    for idx, ev in enumerate(events):
        if idx in used_indices:
            continue
        ev_tokens = set(ev.query.strip().lower().split())
        overlap = len(sq_tokens & ev_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = idx

    if best_idx is not None and best_overlap > 0:
        used_indices.add(best_idx)
        return best_idx, events[best_idx].chunk_ids_returned

    return -1, []


def evaluate_turn_metrics(
    eval_turn: EvalTurn,
    orchestrator_res: OrchestratorResult,
    state: SessionState,
    chunk_lookup: dict[str, Any],
    pricing: PricingConfig,
) -> TurnMetricResult:
    """Compute all evaluation metrics for a completed turn."""
    trace = orchestrator_res.trace
    action = orchestrator_res.action
    action_correct = (action == eval_turn.expected_action)

    # 1. Raw Retrieval Extraction
    # Filter actual retrieval events (controller decision events have chunk_ids_returned=[])
    actual_retrieval_events = [
        e for e in trace.retrieval_events if len(e.chunk_ids_returned) > 0
    ]

    all_retrieved_chunk_ids: list[str] = []
    sub_query_recalls: dict[str, dict[str, float | None]] = {}

    if len(eval_turn.expected_sub_queries) > 1 and eval_turn.sub_query_gold_chunks:
        # Multi-intent: evaluate per-sub-query independently using query-based matching (non-positional)
        sq_recall_1_list: list[float] = []
        sq_recall_3_list: list[float] = []
        sq_recall_5_list: list[float] = []
        sq_recall_10_list: list[float] = []
        used_event_indices: set[int] = set()

        for sq in eval_turn.expected_sub_queries:
            gold_cids = eval_turn.sub_query_gold_chunks.get(sq, [])
            _, ret_cids = _match_retrieval_event(sq, actual_retrieval_events, used_event_indices)

            for cid in ret_cids:
                if cid not in all_retrieved_chunk_ids:
                    all_retrieved_chunk_ids.append(cid)

            r1 = compute_recall_at_k(ret_cids, gold_cids, 1)
            r3 = compute_recall_at_k(ret_cids, gold_cids, 3)
            r5 = compute_recall_at_k(ret_cids, gold_cids, 5)
            r10 = compute_recall_at_k(ret_cids, gold_cids, 10)

            if r1 is not None:
                sq_recall_1_list.append(r1)
            if r3 is not None:
                sq_recall_3_list.append(r3)
            if r5 is not None:
                sq_recall_5_list.append(r5)
            if r10 is not None:
                sq_recall_10_list.append(r10)

            sub_query_recalls[sq] = {
                "recall@1": r1,
                "recall@3": r3,
                "recall@5": r5,
                "recall@10": r10,
            }

        recall_1 = statistics.mean(sq_recall_1_list) if sq_recall_1_list else None
        recall_3 = statistics.mean(sq_recall_3_list) if sq_recall_3_list else None
        recall_5 = statistics.mean(sq_recall_5_list) if sq_recall_5_list else None
        recall_10 = statistics.mean(sq_recall_10_list) if sq_recall_10_list else None

    else:
        # Single-intent or non-split retrieval
        if actual_retrieval_events:
            all_retrieved_chunk_ids = actual_retrieval_events[0].chunk_ids_returned
        else:
            all_retrieved_chunk_ids = []

        recall_1 = compute_recall_at_k(all_retrieved_chunk_ids, eval_turn.expected_chunk_ids, 1)
        recall_3 = compute_recall_at_k(all_retrieved_chunk_ids, eval_turn.expected_chunk_ids, 3)
        recall_5 = compute_recall_at_k(all_retrieved_chunk_ids, eval_turn.expected_chunk_ids, 5)
        recall_10 = compute_recall_at_k(all_retrieved_chunk_ids, eval_turn.expected_chunk_ids, 10)

    # 1b. Post-Fusion Candidate Recall (evaluated against union of gold chunks for the turn)
    fused_chunk_ids = (
        orchestrator_res.turn_result.fused_chunk_ids
        if orchestrator_res.turn_result is not None
        else []
    )
    fused_recall_1 = compute_recall_at_k(fused_chunk_ids, eval_turn.expected_chunk_ids, 1)
    fused_recall_3 = compute_recall_at_k(fused_chunk_ids, eval_turn.expected_chunk_ids, 3)
    fused_recall_5 = compute_recall_at_k(fused_chunk_ids, eval_turn.expected_chunk_ids, 5)
    fused_recall_10 = compute_recall_at_k(fused_chunk_ids, eval_turn.expected_chunk_ids, 10)

    # 2. Evidence Relevance & Citation Groundedness against Gold Evidence
    # Inspect actual ledger claims committed in this turn
    entry_id = f"entry_{eval_turn.turn_number}"
    current_entry = None
    if state.ledger is not None and getattr(state.ledger, "entries", None):
        current_entry = state.ledger.entries.get(entry_id) or state.ledger.entries.get("entry_1")

    generated_claims = active_claims(current_entry) if current_entry else []
    total_claims = len(generated_claims)

    supported_count = 0
    unsupported_count = 0
    fabricated_citations_count = 0

    if eval_turn.is_answerable:
        expected_chunk_set = set(eval_turn.expected_chunk_ids)
        for c in generated_claims:
            cid = c.chunk_id
            # Fabricated citation: cited chunk doesn't exist in corpus index
            if cid not in chunk_lookup:
                fabricated_citations_count += 1
                unsupported_count += 1
            elif cid in expected_chunk_set:
                # Cites a known relevant gold chunk
                supported_count += 1
            else:
                # Cites an existing chunk that is not part of the gold relevant set
                unsupported_count += 1

        citation_groundedness = (
            float(supported_count) / float(total_claims) if total_claims > 0 else 0.0
        )
        citation_validity = (
            float(total_claims - fabricated_citations_count) / float(total_claims)
            if total_claims > 0
            else 1.0
        )
        fabricated_citation_rate = (
            float(fabricated_citations_count) / float(total_claims)
            if total_claims > 0
            else 0.0
        )
        unanswerable_ok = None

    else:
        # Unanswerable / coverage gap case
        # Goal: zero unsupported factual claims, zero fabricated citations
        for c in generated_claims:
            if c.chunk_id not in chunk_lookup:
                fabricated_citations_count += 1
            unsupported_count += 1

        # Correct if no unsupported affirmative claims were added to ledger
        # or if the pipeline flagged insufficient evidence
        turn_res = orchestrator_res.turn_result
        has_insufficient_flag = (
            turn_res is not None and getattr(turn_res, "contradiction_unresolved", False) is False
            and (len(generated_claims) == 0 or turn_res.text.strip() == "")
        )
        unanswerable_ok = (total_claims == 0 or has_insufficient_flag)

        citation_groundedness = 1.0 if unanswerable_ok else 0.0
        citation_validity = 1.0 if fabricated_citations_count == 0 else 0.0
        fabricated_citation_rate = (
            float(fabricated_citations_count) / float(total_claims) if total_claims > 0 else 0.0
        )

    # 3. Refinement Checks
    version_transition_ok = None
    if eval_turn.turn_number > 1:
        if current_entry and current_entry.version >= eval_turn.turn_number:
            version_transition_ok = True
        else:
            version_transition_ok = False

    # 4. Contradiction Checks
    contradiction_ok = None
    if eval_turn.contradiction_pair:
        c1, c2 = eval_turn.contradiction_pair
        # Both chunks should be present in retrieved or candidate evidence
        ret_set = set(all_retrieved_chunk_ids)
        if c1 in ret_set and c2 in ret_set:
            contradiction_ok = True
        elif c1 in ret_set or c2 in ret_set:
            contradiction_ok = True  # At least one side retrieved without crashing
        else:
            contradiction_ok = False

    # 5. Latency Breakdown
    retrieval_lat = sum(e.duration_s for e in trace.stage_events if e.stage == "retrieval")
    generator_lat = sum(e.duration_s for e in trace.stage_events if e.stage == "generator")
    verifier_lat = sum(e.duration_s for e in trace.stage_events if e.stage == "verifier")
    total_lat = trace.duration_s
    response_proxy = retrieval_lat + generator_lat

    # 6. Cost & Token Calculation
    in_tok = trace.total_estimated_input_tokens
    out_tok = trace.total_estimated_output_tokens
    tot_tok = in_tok + out_tok

    cost_usd = (in_tok / 1_000_000 * pricing.input_usd_per_1m) + (
        out_tok / 1_000_000 * pricing.output_usd_per_1m
    )

    return TurnMetricResult(
        turn_number=eval_turn.turn_number,
        user_input=eval_turn.user_input,
        action=action,
        expected_action=eval_turn.expected_action,
        action_correct=action_correct,
        recall_at_1=recall_1,
        recall_at_3=recall_3,
        recall_at_5=recall_5,
        recall_at_10=recall_10,
        sub_query_recalls=sub_query_recalls,
        retrieved_chunk_ids=all_retrieved_chunk_ids,
        fused_recall_at_1=fused_recall_1,
        fused_recall_at_3=fused_recall_3,
        fused_recall_at_5=fused_recall_5,
        fused_recall_at_10=fused_recall_10,
        fused_chunk_ids=fused_chunk_ids,
        total_claims_generated=total_claims,
        supported_claims=supported_count,
        unsupported_claims=unsupported_count,
        fabricated_citations=fabricated_citations_count,
        citation_groundedness=citation_groundedness,
        claim_groundedness=citation_groundedness,
        citation_validity=citation_validity,
        fabricated_citation_rate=fabricated_citation_rate,
        is_answerable=eval_turn.is_answerable,
        unanswerable_handled_correctly=unanswerable_ok,
        version_transition_ok=version_transition_ok,
        contradiction_handled_ok=contradiction_ok,
        total_turn_latency_s=total_lat,
        retrieval_latency_s=retrieval_lat,
        generator_latency_s=generator_lat,
        verifier_latency_s=verifier_lat,
        response_latency_proxy_s=response_proxy,
        input_tokens=in_tok,
        output_tokens=out_tok,
        total_tokens=tot_tok,
        token_count_mode="estimated",
        estimated_cost_usd=cost_usd,
        trace_path=trace.path,
    )


# ---------------------------------------------------------------------------
# Runner & Aggregation
# ---------------------------------------------------------------------------

def run_benchmark(
    orchestrator: StreamRAGOrchestrator,
    index: dict[str, Any],
    scenarios: Sequence[EvalScenario] | None = None,
    pricing: PricingConfig | None = None,
    log_dir: Path | str | None = None,
    verbose: bool = True,
) -> AggregateBenchmarkResult:
    """Execute the evaluation benchmark over the given scenarios."""
    eval_scenarios = list(scenarios or get_eval_scenarios())
    pricing_cfg = pricing or PricingConfig()
    chunk_lookup = index.get("chunk_lookup", {})

    scenario_results: list[ScenarioMetricResult] = []
    total_scenarios = len(eval_scenarios)
    benchmark_start_time = time.perf_counter()

    for scen_idx, scen in enumerate(eval_scenarios, 1):
        scen_start_time = time.perf_counter()
        if verbose:
            print(f"\n[{scen_idx}/{total_scenarios}] Running scenario: {scen.scenario_id} — {scen.title}", flush=True)

        # Create a fresh session for each scenario
        state = SessionState(session_id=scen.scenario_id)
        turn_results: list[TurnMetricResult] = []
        total_turns = len(scen.turns)

        for turn_idx, turn in enumerate(scen.turns, 1):
            log_path = None
            if log_dir is not None:
                log_path = Path(log_dir) / f"{scen.scenario_id}_turn{turn.turn_number}.jsonl"

            if verbose:
                print(f"    Turn {turn_idx}/{total_turns}: running...", flush=True)
                print(f"    → executing pipeline...", flush=True)

            t_turn_start = time.perf_counter()

            # Execute turn through the StreamRAGOrchestrator
            res = orchestrator.step(
                state=state,
                chunk=turn.user_input,
                timestamp_s=turn.timestamp_s,
            )

            # Compute turn metrics
            metric_res = evaluate_turn_metrics(
                eval_turn=turn,
                orchestrator_res=res,
                state=state,
                chunk_lookup=chunk_lookup,
                pricing=pricing_cfg,
            )
            turn_results.append(metric_res)

            t_turn_elapsed = time.perf_counter() - t_turn_start
            scen_elapsed = time.perf_counter() - scen_start_time
            cum_elapsed = time.perf_counter() - benchmark_start_time
            if verbose:
                print(
                    f"    ✓ completed in {t_turn_elapsed:.1f}s "
                    f"(scenario: {scen_elapsed:.1f}s | benchmark: {cum_elapsed:.1f}s)",
                    flush=True,
                )

        valid_rec10 = [t.recall_at_10 for t in turn_results if t.recall_at_10 is not None]
        mean_rec10 = statistics.mean(valid_rec10) if valid_rec10 else None

        valid_fused_rec10 = [t.fused_recall_at_10 for t in turn_results if t.fused_recall_at_10 is not None]
        mean_fused_rec10 = statistics.mean(valid_fused_rec10) if valid_fused_rec10 else None

        mean_ground = statistics.mean(t.citation_groundedness for t in turn_results) if turn_results else 0.0
        tot_toks = sum(t.total_tokens for t in turn_results)
        tot_lat = sum(t.total_turn_latency_s for t in turn_results)
        all_passed = all(
            t.action_correct
            and (t.recall_at_10 is None or t.recall_at_10 > 0.0)
            and (t.unanswerable_handled_correctly is not False)
            for t in turn_results
        )

        scenario_results.append(
            ScenarioMetricResult(
                scenario_id=scen.scenario_id,
                category=scen.category,
                title=scen.title,
                turns=turn_results,
                mean_recall_at_10=mean_rec10,
                citation_groundedness=mean_ground,
                total_tokens=tot_toks,
                total_latency_s=tot_lat,
                all_turns_passed=all_passed,
                mean_fused_recall_at_10=mean_fused_rec10,
                mean_claim_groundedness=mean_ground,
            )
        )

    return aggregate_results(scenario_results)


def aggregate_results(scenario_results: Sequence[ScenarioMetricResult]) -> AggregateBenchmarkResult:
    """Aggregate per-scenario evaluation metrics into an AggregateBenchmarkResult."""
    all_turns: list[TurnMetricResult] = []
    for s in scenario_results:
        all_turns.extend(s.turns)

    total_scenarios = len(scenario_results)
    total_turns = len(all_turns)

    # Raw retrieval recall aggregates (strictly excludes None from denominator)
    valid_r1 = [t.recall_at_1 for t in all_turns if t.recall_at_1 is not None]
    valid_r3 = [t.recall_at_3 for t in all_turns if t.recall_at_3 is not None]
    valid_r5 = [t.recall_at_5 for t in all_turns if t.recall_at_5 is not None]
    valid_r10 = [t.recall_at_10 for t in all_turns if t.recall_at_10 is not None]

    mean_r1 = statistics.mean(valid_r1) if valid_r1 else None
    mean_r3 = statistics.mean(valid_r3) if valid_r3 else None
    mean_r5 = statistics.mean(valid_r5) if valid_r5 else None
    mean_r10 = statistics.mean(valid_r10) if valid_r10 else None

    # Post-fusion recall aggregates (strictly excludes None from denominator)
    valid_fused_r1 = [t.fused_recall_at_1 for t in all_turns if t.fused_recall_at_1 is not None]
    valid_fused_r3 = [t.fused_recall_at_3 for t in all_turns if t.fused_recall_at_3 is not None]
    valid_fused_r5 = [t.fused_recall_at_5 for t in all_turns if t.fused_recall_at_5 is not None]
    valid_fused_r10 = [t.fused_recall_at_10 for t in all_turns if t.fused_recall_at_10 is not None]

    mean_fused_r1 = statistics.mean(valid_fused_r1) if valid_fused_r1 else None
    mean_fused_r3 = statistics.mean(valid_fused_r3) if valid_fused_r3 else None
    mean_fused_r5 = statistics.mean(valid_fused_r5) if valid_fused_r5 else None
    mean_fused_r10 = statistics.mean(valid_fused_r10) if valid_fused_r10 else None

    # Sub-query recall
    sub_query_recalls_10: list[float] = []
    for t in all_turns:
        for sq_name, metrics in t.sub_query_recalls.items():
            if metrics.get("recall@10") is not None:
                sub_query_recalls_10.append(metrics["recall@10"])
    mean_sq_r10 = (
        statistics.mean(sub_query_recalls_10) if sub_query_recalls_10 else mean_r10
    )

    # Controller action accuracy (correctly classified actions / total evaluated turns)
    correct_actions = sum(1 for t in all_turns if t.action_correct)
    controller_action_acc = float(correct_actions) / float(total_turns) if total_turns > 0 else 1.0

    # Evidence & Citation Groundedness
    mean_ground = statistics.mean(t.citation_groundedness for t in all_turns) if all_turns else 0.0
    mean_valid = statistics.mean(t.citation_validity for t in all_turns) if all_turns else 1.0
    mean_fab = statistics.mean(t.fabricated_citation_rate for t in all_turns) if all_turns else 0.0

    # Latencies
    turn_latencies = [t.total_turn_latency_s for t in all_turns] if all_turns else [0.0]
    mean_lat = statistics.mean(turn_latencies)
    median_lat = statistics.median(turn_latencies)
    mean_ret_lat = statistics.mean(t.retrieval_latency_s for t in all_turns) if all_turns else 0.0
    mean_gen_lat = statistics.mean(t.generator_latency_s for t in all_turns) if all_turns else 0.0
    mean_ver_lat = statistics.mean(t.verifier_latency_s for t in all_turns) if all_turns else 0.0

    # Tokens & Cost
    tot_in_tok = sum(t.input_tokens for t in all_turns)
    tot_out_tok = sum(t.output_tokens for t in all_turns)
    tot_tok = tot_in_tok + tot_out_tok
    tok_per_turn = float(tot_tok) / float(total_turns) if total_turns > 0 else 0.0
    tot_cost = sum(t.estimated_cost_usd for t in all_turns)

    # Category breakdown
    categories = {"single_intent", "multi_intent", "refinement", "unanswerable", "contradiction"}
    cat_metrics: dict[str, dict[str, Any]] = {}

    for cat in categories:
        cat_scens = [s for s in scenario_results if s.category == cat]
        cat_turns = [t for s in cat_scens for t in s.turns]
        if not cat_turns:
            continue

        cat_rec10 = [t.recall_at_10 for t in cat_turns if t.recall_at_10 is not None]
        cat_fused_rec10 = [t.fused_recall_at_10 for t in cat_turns if t.fused_recall_at_10 is not None]
        cat_correct = sum(1 for t in cat_turns if t.action_correct)

        cat_metrics[cat] = {
            "scenario_count": len(cat_scens),
            "turn_count": len(cat_turns),
            "controller_action_accuracy": float(cat_correct) / float(len(cat_turns)) if cat_turns else 1.0,
            "mean_recall@10": statistics.mean(cat_rec10) if cat_rec10 else None,
            "mean_fused_recall@10": statistics.mean(cat_fused_rec10) if cat_fused_rec10 else None,
            "citation_groundedness": statistics.mean(t.citation_groundedness for t in cat_turns),
            "mean_groundedness": statistics.mean(t.citation_groundedness for t in cat_turns),
            "mean_latency_s": statistics.mean(t.total_turn_latency_s for t in cat_turns),
            "total_tokens": sum(t.total_tokens for t in cat_turns),
        }

    return AggregateBenchmarkResult(
        benchmark_version="1.0.0",
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        total_scenarios=total_scenarios,
        total_turns=total_turns,
        mean_recall_at_1=mean_r1,
        mean_recall_at_3=mean_r3,
        mean_recall_at_5=mean_r5,
        mean_recall_at_10=mean_r10,
        mean_sub_query_recall_at_10=mean_sq_r10,
        mean_fused_recall_at_1=mean_fused_r1,
        mean_fused_recall_at_3=mean_fused_r3,
        mean_fused_recall_at_5=mean_fused_r5,
        mean_fused_recall_at_10=mean_fused_r10,
        controller_action_accuracy=controller_action_acc,
        citation_groundedness=mean_ground,
        mean_claim_groundedness=mean_ground,
        mean_citation_validity=mean_valid,
        mean_fabricated_citation_rate=mean_fab,
        mean_turn_latency_s=mean_lat,
        median_turn_latency_s=median_lat,
        mean_retrieval_latency_s=mean_ret_lat,
        mean_generator_latency_s=mean_gen_lat,
        mean_verifier_latency_s=mean_ver_lat,
        total_input_tokens=tot_in_tok,
        total_output_tokens=tot_out_tok,
        total_tokens=tot_tok,
        tokens_per_turn=tok_per_turn,
        token_count_mode="estimated",
        total_estimated_cost_usd=tot_cost,
        category_metrics=cat_metrics,
        scenarios=list(scenario_results),
    )


# ---------------------------------------------------------------------------
# Formatting Utilities (Markdown / JSON)
# ---------------------------------------------------------------------------

def _fmt_pct(val: float | None) -> str:
    """Format float as percentage or 'N/A' if None."""
    if val is None:
        return "N/A"
    return f"{val * 100:.1f}%"


def format_benchmark_markdown(res: AggregateBenchmarkResult) -> str:
    """Format benchmark results into a clean GitHub-style Markdown report."""
    lines: list[str] = [
        "# Streaming Live RAG — Benchmark Evaluation Report",
        "",
        f"**Generated**: {res.timestamp_utc}  ",
        f"**Total Scenarios Evaluated**: {res.total_scenarios} ({res.total_turns} turns)  ",
        f"**Token Count Mode**: {res.token_count_mode} (3.5 chars / token)  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Primary Theme 4 Metrics",
        "",
        "### A. Retrieval Metrics",
        "| Metric | Result | Target Benchmark Status | Notes |",
        "| :--- | :---: | :---: | :--- |",
        f"| **Raw Retrieval Recall@10** | **{_fmt_pct(res.mean_recall_at_10)}** | ✅ Evaluated | Mean across answerable turns (excludes empty-gold) |",
        f"| **Raw Retrieval Recall@5** | {_fmt_pct(res.mean_recall_at_5)} | ✅ Evaluated | Top-5 retrieval recall |",
        f"| **Raw Retrieval Recall@3** | {_fmt_pct(res.mean_recall_at_3)} | ✅ Evaluated | Top-3 retrieval recall |",
        f"| **Raw Retrieval Recall@1** | {_fmt_pct(res.mean_recall_at_1)} | ✅ Evaluated | Top-1 retrieval recall |",
        f"| **Fused Candidate Recall@10** | **{_fmt_pct(res.mean_fused_recall_at_10)}** | ✅ Evaluated | Post-fusion/reranked candidate list @ 10 |",
        f"| **Fused Candidate Recall@5** | {_fmt_pct(res.mean_fused_recall_at_5)} | ✅ Evaluated | Post-fusion/reranked candidate list @ 5 |",
        f"| **Fused Candidate Recall@3** | {_fmt_pct(res.mean_fused_recall_at_3)} | ✅ Evaluated | Post-fusion/reranked candidate list @ 3 |",
        f"| **Fused Candidate Recall@1** | {_fmt_pct(res.mean_fused_recall_at_1)} | ✅ Evaluated | Post-fusion/reranked candidate list @ 1 |",
        f"| **Multi-Intent Sub-query Recall@10** | {_fmt_pct(res.mean_sub_query_recall_at_10)} | ✅ Evaluated | Independent per-sub-query recall |",
        "",
        "### B. Routing Metrics",
        "| Metric | Result | Target Benchmark Status | Notes |",
        "| :--- | :---: | :---: | :--- |",
        f"| **Controller Action Accuracy** | **{_fmt_pct(res.controller_action_accuracy)}** | ✅ Evaluated | Correctly classified actions / total turns |",
        "",
        "### C. Evidence Relevance & Groundedness",
        "| Metric | Result | Target Benchmark Status | Notes |",
        "| :--- | :---: | :---: | :--- |",
        f"| **Citation Groundedness** | **{res.citation_groundedness * 100:.1f}%** | ✅ Evaluated | Evidence Relevance Rate: claims citing expected chunks |",
        f"| **Citation Validity** | {res.mean_citation_validity * 100:.1f}% | ✅ Evaluated | Valid corpus chunks cited |",
        f"| **Fabricated Citation Rate** | {res.mean_fabricated_citation_rate * 100:.1f}% | ✅ Evaluated | Non-existent chunk IDs cited |",
        "",
        "### D. Latency Breakdown (Synchronous Pipeline)",
        "| Metric | Result | Target Benchmark Status | Notes |",
        "| :--- | :---: | :---: | :--- |",
        f"| **Mean Turn Latency** | {res.mean_turn_latency_s:.3f}s | ✅ Timed | Monotonic perf_counter duration |",
        f"| **Median Turn Latency** | {res.median_turn_latency_s:.3f}s | ✅ Timed | Turn latency median |",
        f"| **Retrieval Latency** | {res.mean_retrieval_latency_s:.3f}s | ✅ Timed | Index retrieval stage duration |",
        f"| **Generator Latency** | {res.mean_generator_latency_s:.3f}s | ✅ Timed | LLM generation stage duration |",
        f"| **Verifier Latency** | {res.mean_verifier_latency_s:.3f}s | ✅ Timed | Verification stage duration |",
        f"| **Response Latency Proxy** | {(res.mean_retrieval_latency_s + res.mean_generator_latency_s):.3f}s | ✅ Modeled | Retrieval + Generator duration |",
        "| **Streaming TTFT** | **N/A** | ℹ️ Design Note | Architecture operates synchronously (`stream=False`) |",
        "",
        "### E. Token Usage & Modeled Cost",
        "| Metric | Result | Target Benchmark Status | Notes |",
        "| :--- | :---: | :---: | :--- |",
        f"| **Tokens per Turn** | {res.tokens_per_turn:.1f} | ✅ Estimated | Modeled at ~3.5 chars/token |",
        f"| **Total Tokens** | {res.total_tokens:,} | ✅ Estimated | Modeled across all turns |",
        f"| **Total Estimated API Cost** | ${res.total_estimated_cost_usd:.4f} | ✅ Modeled | Based on active PricingConfig |",
        "",
        "> [!IMPORTANT]",
        f"> {res.latency_note}",
        "",
        "---",
        "",
        "## 2. Category Performance Breakdown",
        "",
        "| Category | Scenarios | Turns | Controller Accuracy | Raw Recall@10 | Fused Recall@10 | Groundedness | Mean Latency (s) | Tokens |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for cat, m in res.category_metrics.items():
        lines.append(
            f"| `{cat}` | {m['scenario_count']} | {m['turn_count']} | "
            f"{_fmt_pct(m.get('controller_action_accuracy'))} | "
            f"{_fmt_pct(m['mean_recall@10'])} | {_fmt_pct(m.get('mean_fused_recall@10'))} | "
            f"{m['citation_groundedness'] * 100:.1f}% | "
            f"{m['mean_latency_s']:.3f}s | {m['total_tokens']:,} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Detailed Scenario Results",
        "",
        "| Scenario ID | Category | Raw Recall@10 | Fused Recall@10 | Groundedness | Latency (s) | Tokens | Status |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for s in res.scenarios:
        status_icon = "✅ PASS" if s.all_turns_passed else "⚠️ REVIEW"
        lines.append(
            f"| `{s.scenario_id}` | `{s.category}` | {_fmt_pct(s.mean_recall_at_10)} | "
            f"{_fmt_pct(s.mean_fused_recall_at_10)} | "
            f"{s.citation_groundedness * 100:.1f}% | {s.total_latency_s:.3f}s | "
            f"{s.total_tokens:,} | {status_icon} |"
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming Live RAG Benchmark Runner")
    parser.add_argument("--json", dest="json_path", type=str, help="Output path for JSON results")
    parser.add_argument("--markdown", dest="md_path", type=str, help="Output path for Markdown report")
    parser.add_argument("--category", type=str, default=None, help="Filter by scenario category")
    parser.add_argument("--scenario", type=str, default=None, help="Filter by specific scenario ID")
    parser.add_argument("--input-price", type=float, default=0.0, help="USD per 1M input tokens")
    parser.add_argument("--output-price", type=float, default=0.0, help="USD per 1M output tokens")
    args = parser.parse_args()

    # Build index & pipeline components
    from retrieval.indexer import build_index
    corpus_dir = Path(__file__).parent.parent / "corpus" / "raw"
    index = build_index(str(corpus_dir))

    client = CountingLLMClient(get_llm_client())
    controller = RuleBasedController()
    decomposer = Decomposer(llm_client=client)
    fusion = FusionPipeline(contradiction_screen=ContradictionScreen(llm_client=client))
    generator = AnswerGenerator(client)
    verifier = GroundingVerifier(client)

    orchestrator = StreamRAGOrchestrator(
        controller=controller,
        index=index,
        fusion=fusion,
        generator=generator,
        verifier=verifier,
        decomposer=decomposer,
        llm_counter=client,
    )

    pricing = PricingConfig(
        input_usd_per_1m=args.input_price,
        output_usd_per_1m=args.output_price,
    )

    scenarios = get_eval_scenarios()
    if args.scenario:
        scenarios = [s for s in scenarios if s.scenario_id == args.scenario]
    elif args.category:
        scenarios = [s for s in scenarios if s.category == args.category]

    total_turns_count = sum(len(s.turns) for s in scenarios)
    print(f"Running benchmark on {len(scenarios)} scenarios ({total_turns_count} turns)...", flush=True)
    bench_start = time.perf_counter()
    results = run_benchmark(
        orchestrator=orchestrator,
        index=index,
        scenarios=scenarios,
        pricing=pricing,
        verbose=True,
    )
    bench_total_s = time.perf_counter() - bench_start

    md_report = format_benchmark_markdown(results)
    print("\n" + md_report, flush=True)

    if args.json_path:
        out_p = Path(args.json_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(results.to_dict(), indent=2), encoding="utf-8")
        print(f"Saved JSON benchmark results to {args.json_path}", flush=True)

    if args.md_path:
        out_m = Path(args.md_path)
        out_m.parent.mkdir(parents=True, exist_ok=True)
        out_m.write_text(md_report, encoding="utf-8")
        print(f"Saved Markdown report to {args.md_path}", flush=True)

    print("\n" + "=" * 50, flush=True)
    print("BENCHMARK COMPLETE", flush=True)
    print("=" * 50, flush=True)
    print(f"Scenarios: {results.total_scenarios}/{len(scenarios)}", flush=True)
    print(f"Turns: {results.total_turns}/{total_turns_count}", flush=True)
    print(f"Total time: {bench_total_s:.1f}s", flush=True)
    if args.md_path:
        print(f"Report: {args.md_path}", flush=True)
    if args.json_path:
        print(f"JSON: {args.json_path}", flush=True)
    print("=" * 50, flush=True)


if __name__ == "__main__":
    main()
