from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

from controller.types import Action, Controller, RetrievalDecision, SessionState
from session.pipeline import TurnResult, answer_new_topic, answer_refinement, answer_suppress
from telemetry.hooks import make_llm_hook
from telemetry.schema import TurnTrace
from telemetry.writer import write_event_log

if TYPE_CHECKING:
    from controller.llm_wrapper import CountingLLMClient
    from decomposer.decomposer import Decomposer
    from fusion.fusion_pipeline import FusionPipeline
    from grounding.verifier import GroundingVerifier
    from session.generator import AnswerGenerator


@dataclass
class OrchestratorResult:
    action: Action
    decision: RetrievalDecision | None = None
    turn_result: TurnResult | None = None
    trace: TurnTrace | None = None
    sub_queries: list[str] = field(default_factory=list)
    prefetched_chunk_ids: list[str] = field(default_factory=list)
    prefetched_latency_ms: float = 0.0
    is_cache_hit: bool = False

    @property
    def text(self) -> str:
        return self.turn_result.text if self.turn_result else ""


class StreamRAGOrchestrator:
    """Lightweight coordinator for streaming chunks, controller routing,
    pipeline execution, and trace persistence.
    """

    def __init__(
        self,
        controller: Controller,
        index: dict | None = None,
        fusion: FusionPipeline | None = None,
        generator: AnswerGenerator | None = None,
        verifier: GroundingVerifier | None = None,
        decomposer: Decomposer | None = None,
        llm_counter: CountingLLMClient | None = None,
        log_path: str | Path | None = None,
    ):
        self.controller = controller
        self.index = index
        self.fusion = fusion
        self.generator = generator
        self.verifier = verifier
        self.decomposer = decomposer
        self.llm_counter = llm_counter
        self.log_path = log_path
        self.turn_counter: int = 0

    def step(
        self,
        state: SessionState,
        chunk: str,
        timestamp_s: float,
        trace: TurnTrace | None = None,
        entry_id: str | None = None,
        is_refinement: bool | None = None,
        is_final: bool = True,
    ) -> OrchestratorResult:
        """Process a single incoming stream token chunk."""
        self.turn_counter += 1
        active_trace = trace
        if active_trace is None:
            active_trace = TurnTrace(turn_number=self.turn_counter, session_id=state.session_id)

        if self.llm_counter is not None:
            self.llm_counter.on_call = make_llm_hook(active_trace)

        if active_trace.wall_start == 0.0:
            active_trace.begin()

        turn_result: TurnResult | None = None
        decision: RetrievalDecision | None = None

        try:
            decision = self.controller.decide(state, chunk, timestamp_s)
            if is_final and decision.action == "WAIT" and state.transcript_so_far.strip():
                decision = RetrievalDecision("RETRIEVE", "end_of_utterance", 1.0, "Speaker finished utterance")

            state.decision_log.append({
                "timestamp_s": timestamp_s,
                "chunk": chunk,
                "action": decision.action,
                "trigger": decision.trigger,
                "confidence": decision.confidence,
                "reason": decision.reason,
            })

            # Record controller decision in trace
            active_trace.record_retrieval(
                action=decision.action,
                trigger=decision.trigger,
                confidence=decision.confidence,
                query=chunk,
                reason=decision.reason,
                transcript_timestamp_s=timestamp_s,
            )

            if decision.action == "WAIT":
                active_trace.finalize(path="WAIT")
                return OrchestratorResult(
                    action="WAIT",
                    decision=decision,
                    turn_result=None,
                    trace=active_trace,
                    sub_queries=[],
                    prefetched_chunk_ids=list(state.prefetched_candidate_ids) if state.prefetched_candidate_ids else [],
                    prefetched_latency_ms=0.0,
                    is_cache_hit=False,
                )

            elif decision.action == "SUPPRESS":
                target_entry = entry_id
                if not target_entry and state.ledger and state.ledger.entries:
                    target_entry = list(state.ledger.entries.keys())[-1]

                turn_result = answer_suppress(
                    state=state,
                    instruction=chunk,
                    entry_id=target_entry,
                    trace=active_trace,
                )
                return OrchestratorResult(
                    action="SUPPRESS",
                    decision=decision,
                    turn_result=turn_result,
                    trace=active_trace,
                )

            elif decision.action == "RETRIEVE":
                if self.index is None or self.fusion is None or self.generator is None or self.verifier is None:
                    # Controller-only mode (e.g. simulation or test harness without pipeline)
                    active_trace.finalize(path="RETRIEVE")
                    return OrchestratorResult(
                        action="RETRIEVE",
                        decision=decision,
                        turn_result=None,
                        trace=active_trace,
                    )

                # Speculative pre-fetch branch for intermediate streaming chunks
                if not is_final:
                    t_prefetch_start = time.perf_counter()
                    query = state.transcript_so_far or chunk
                    detected_subqueries: list[str] = []
                    candidate_ids: list[str] = []

                    if self.decomposer and getattr(decision, "possible_multi_intent", False):
                        decomp_res = self.decomposer.decompose(query)
                        if decomp_res.sub_queries:
                            detected_subqueries = list(decomp_res.sub_queries)
                            for sq in detected_subqueries:
                                sq_cids = self.index["retrieve"](sq, top_k=10)
                                for cid in sq_cids:
                                    if cid not in candidate_ids:
                                        candidate_ids.append(cid)

                    if not candidate_ids:
                        with active_trace.start_stage("retrieval"):
                            candidate_ids = self.index["retrieve"](query, top_k=10)

                    state.prefetched_candidate_ids = candidate_ids
                    state.prefetched_query = query
                    latency_ms = (time.perf_counter() - t_prefetch_start) * 1000
                    state.prefetched_latency_s = latency_ms / 1000

                    active_trace.record_retrieval(
                        action="PREFETCH",
                        trigger=decision.trigger,
                        confidence=decision.confidence,
                        query=query,
                        chunk_ids_returned=candidate_ids,
                        reason=f"speculative pre-fetch while user speaking ({decision.reason})",
                        transcript_timestamp_s=timestamp_s,
                    )
                    active_trace.finalize(path="PREFETCH")
                    return OrchestratorResult(
                        action="PREFETCH",
                        decision=decision,
                        turn_result=None,
                        trace=active_trace,
                        sub_queries=detected_subqueries,
                        prefetched_chunk_ids=candidate_ids,
                        prefetched_latency_ms=latency_ms,
                        is_cache_hit=False,
                    )

                # Final speech turn: reuse prefetched candidates if available and matching
                active_index = self.index
                topic = state.transcript_so_far or chunk
                is_cache_hit = False
                if state.prefetched_candidate_ids:
                    cached_cids = list(state.prefetched_candidate_ids)
                    cached_query = state.prefetched_query
                    is_cache_hit = True

                    def cached_retrieve(q: str, top_k: int = 10) -> list[str]:
                        if cached_query and (q == cached_query or q.startswith(cached_query) or cached_query in q):
                            return cached_cids[:top_k]
                        return self.index["retrieve"](q, top_k=top_k)

                    active_index = dict(self.index)
                    active_index["retrieve"] = cached_retrieve

                # Determine if NEW_TOPIC or REFINEMENT
                route_refinement = False
                target_entry = entry_id
                if is_refinement is True:
                    route_refinement = True
                elif is_refinement is False:
                    route_refinement = False
                elif target_entry and state.ledger and target_entry in state.ledger.entries:
                    route_refinement = True
                elif state.ledger and state.ledger.entries and state.last_answer_topic is not None:
                    route_refinement = True
                    target_entry = list(state.ledger.entries.keys())[-1]

                if route_refinement:
                    target_entry = target_entry or list(state.ledger.entries.keys())[-1]
                    turn_result = answer_refinement(
                        state=state,
                        entry_id=target_entry,
                        detail=chunk,
                        index=active_index,
                        fusion=self.fusion,
                        generator=self.generator,
                        verifier=self.verifier,
                        llm_counter=self.llm_counter,
                        trace=active_trace,
                    )
                else:
                    turn_result = answer_new_topic(
                        state=state,
                        topic=topic,
                        index=active_index,
                        fusion=self.fusion,
                        generator=self.generator,
                        verifier=self.verifier,
                        llm_counter=self.llm_counter,
                        trace=active_trace,
                        decomposer=self.decomposer,
                    )

                detected_subqueries = []
                for se in active_trace.stage_events:
                    if "sub_queries" in se.metadata:
                        detected_subqueries = list(se.metadata["sub_queries"])
                        break

                return OrchestratorResult(
                    action="RETRIEVE",
                    decision=decision,
                    turn_result=turn_result,
                    trace=active_trace,
                    sub_queries=detected_subqueries,
                    prefetched_chunk_ids=list(state.prefetched_candidate_ids) if state.prefetched_candidate_ids else [],
                    prefetched_latency_ms=state.prefetched_latency_s * 1000,
                    is_cache_hit=is_cache_hit,
                )

            else:
                active_trace.finalize(path="UNKNOWN")
                return OrchestratorResult(
                    action=decision.action,
                    decision=decision,
                    turn_result=None,
                    trace=active_trace,
                )

        finally:
            if active_trace.wall_end == 0.0:
                active_trace.finalize(path="ERROR")
            if self.log_path:
                write_event_log([active_trace], self.log_path)
