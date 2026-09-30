from __future__ import annotations
import copy
import dataclasses
from dataclasses import dataclass, field
from controller.types import SessionState
from grounding.types import ClaimVerdict, VerificationResult
from session.ledger import AnswerLedger, LedgerClaim, commit_entry, active_claims
from session.generator import AnswerGenerator
from session.render import render
from grounding.verifier import GroundingVerifier
from fusion.fusion_pipeline import FusionPipeline
from telemetry.schema import TurnTrace


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from decomposer.decomposer import Decomposer


@dataclass
class TurnResult:
    text: str
    verification: VerificationResult
    rejected: list[ClaimVerdict]
    llm_calls: int
    path: str = "NEW_TOPIC"
    contradiction_unresolved: bool = False
    skipped_duplicates: list[str] = field(default_factory=list)
    trace: TurnTrace | None = None
    fused_chunk_ids: list[str] = field(default_factory=list)


def _merge_claim_texts(texts: list[str]) -> str:
    merged: list[str] = []
    for t in texts:
        t = t.strip()
        if not t:
            continue
        if merged and not merged[-1].endswith((".", "!", "?")):
            merged[-1] += "."
        merged.append(t)
    return " ".join(merged)


def answer_new_topic(
    state: SessionState,
    topic: str,
    index: dict,
    fusion: FusionPipeline,
    generator: AnswerGenerator,
    verifier: GroundingVerifier,
    llm_counter=None,
    trace: TurnTrace | None = None,
    decomposer: Decomposer | None = None,
) -> TurnResult:
    # Handle optional argument flexibility
    if decomposer is None:
        if hasattr(llm_counter, "decompose"):
            decomposer = llm_counter
            llm_counter = None
        elif hasattr(trace, "decompose"):
            decomposer = trace
            trace = None

    if trace is not None and trace.wall_start == 0.0:
        trace.begin()

    try:
        # Per-turn defensive copy: index["chunk_lookup"] must never be mutated
        lookup = {k: copy.deepcopy(v) for k, v in index["chunk_lookup"].items()}

        # Multi-intent decomposition or single-query retrieval
        if decomposer is not None:
            if trace:
                with trace.start_stage("decomposition"):
                    decomp_res = decomposer.decompose(topic)
            else:
                decomp_res = decomposer.decompose(topic)

            if decomp_res.is_compound and len(decomp_res.sub_queries) > 1:
                ranked_lists: list[list[str]] = []
                if trace:
                    with trace.start_stage("retrieval"):
                        for sq in decomp_res.sub_queries:
                            cids = index["retrieve"](sq, top_k=10)
                            ranked_lists.append(cids)
                            trace.record_retrieval(
                                action="RETRIEVE",
                                trigger="multi_intent",
                                query=sq,
                                chunk_ids_returned=cids,
                                reason="sub_query_retrieval",
                            )
                    top_n = max(10, 5 * len(decomp_res.sub_queries))
                    with trace.start_stage("fusion", query=topic, sub_queries=decomp_res.sub_queries):
                        try:
                            fused = fusion.fuse(topic, ranked_lists, lookup, top_n=top_n)
                        except TypeError:
                            fused = fusion.fuse(topic, ranked_lists, lookup)
                else:
                    for sq in decomp_res.sub_queries:
                        cids = index["retrieve"](sq, top_k=10)
                        ranked_lists.append(cids)
                    top_n = max(10, 5 * len(decomp_res.sub_queries))
                    try:
                        fused = fusion.fuse(topic, ranked_lists, lookup, top_n=top_n)
                    except TypeError:
                        fused = fusion.fuse(topic, ranked_lists, lookup)
            else:
                single_query = decomp_res.sub_queries[0] if decomp_res.sub_queries else topic
                if trace:
                    with trace.start_stage("retrieval"):
                        candidate_ids = index["retrieve"](single_query, top_k=10)
                    trace.record_retrieval(
                        action="RETRIEVE",
                        query=single_query,
                        chunk_ids_returned=candidate_ids,
                        reason="new_topic_retrieval",
                    )
                    with trace.start_stage("fusion", query=topic, candidate_count=len(candidate_ids)):
                        fused = fusion.fuse(topic, [candidate_ids], lookup)
                else:
                    candidate_ids = index["retrieve"](single_query, top_k=10)
                    fused = fusion.fuse(topic, [candidate_ids], lookup)
        else:
            # Single-query baseline retrieval
            if trace:
                with trace.start_stage("retrieval"):
                    candidate_ids = index["retrieve"](topic, top_k=10)
                trace.record_retrieval(
                    action="RETRIEVE",
                    query=topic,
                    chunk_ids_returned=candidate_ids,
                    reason="new_topic_retrieval",
                )
                with trace.start_stage("fusion", query=topic, candidate_count=len(candidate_ids)):
                    fused = fusion.fuse(topic, [candidate_ids], lookup)
            else:
                candidate_ids = index["retrieve"](topic, top_k=10)
                fused = fusion.fuse(topic, [candidate_ids], lookup)

        initial_calls = llm_counter.calls if llm_counter is not None else 0

        if fused.insufficient_evidence:
            empty_verif = VerificationResult(
                answer_text="",
                claims=[],
                all_verified=True,
                fabricated_citations=[],
                unsupported_claims=[],
                cherry_picks=[],
                decomposition_degraded=False,
            )
            calls = (llm_counter.calls - initial_calls) if llm_counter is not None else 0
            if trace:
                trace.finalize(path="NEW_TOPIC")
            return TurnResult(
                text="",
                verification=empty_verif,
                rejected=[],
                llm_calls=calls,
                path="NEW_TOPIC",
                contradiction_unresolved=False,
                trace=trace,
                fused_chunk_ids=[c.chunk_id for c in fused.chunks],
            )

        # Generate claim entries
        if trace:
            with trace.start_stage("generator"):
                entries = generator.generate(topic, fused.chunks)
        else:
            entries = generator.generate(topic, fused.chunks)

        # Verify claim entries
        if trace:
            with trace.start_stage("verifier"):
                verification = verifier.verify_entries(entries, fused.chunks)
        else:
            verification = verifier.verify_entries(entries, fused.chunks)

        valid_verdicts: list[ClaimVerdict] = []
        rejected_verdicts: list[ClaimVerdict] = []
        contradiction_unresolved = False

        for verdict in verification.claims:
            if verdict.cherry_pick_violation:
                contradiction_unresolved = True

            if verdict.citation_exists and verdict.supported and not verdict.cherry_pick_violation:
                valid_verdicts.append(verdict)
            else:
                rejected_verdicts.append(verdict)

        claims_by_chunk: dict[str, list[str]] = {}
        for v in valid_verdicts:
            cid = v.cited_chunk_id or ""
            if v.claim_text not in claims_by_chunk.setdefault(cid, []):
                claims_by_chunk[cid].append(v.claim_text)

        passed_claims = [
            LedgerClaim(
                claim=_merge_claim_texts(texts),
                chunk_id=cid,
                origin_version=1,
                status="ACTIVE",
            )
            for cid, texts in claims_by_chunk.items()
        ]

        if state.ledger is None:
            state.ledger = AnswerLedger()

        rendered_text = ""
        if passed_claims:
            entry_id = f"entry_{len(state.ledger.entries) + 1}"
            fused_chunks_map = {c.chunk_id: c for c in fused.chunks}
            passed_chunk_ids = {c.chunk_id for c in passed_claims}
            cited_evidence = {cid: fused_chunks_map[cid] for cid in passed_chunk_ids if cid in fused_chunks_map}
            entry = commit_entry(
                ledger=state.ledger,
                entry_id=entry_id,
                topic=topic,
                details=[],
                claims=passed_claims,
                evidence=cited_evidence,
                version=1,
                turn=len(state.ledger.history) + 1,
                action="NEW_TOPIC",
            )
            if trace:
                with trace.start_stage("render"):
                    rendered_text = render(entry)
                trace.record_version_transition(
                    entry_id=entry_id,
                    old_version=0,
                    new_version=1,
                    claims_added=[c.claim for c in passed_claims],
                    claims_retired=[],
                    cited_chunk_ids=[c.chunk_id for c in passed_claims],
                )
            else:
                rendered_text = render(entry)

        # Set last_answer_topic so suppression path in controller can trigger on follow-ups
        state.last_answer_topic = topic + " " + " ".join(c.claim for c in passed_claims)

        total_calls = (llm_counter.calls - initial_calls) if llm_counter is not None else 0

        if trace:
            trace.finalize(path="NEW_TOPIC")

        return TurnResult(
            text=rendered_text,
            verification=verification,
            rejected=rejected_verdicts,
            llm_calls=total_calls,
            path="NEW_TOPIC",
            contradiction_unresolved=contradiction_unresolved,
            trace=trace,
            fused_chunk_ids=[c.chunk_id for c in fused.chunks],
        )
    finally:
        if trace is not None and trace.wall_end == 0.0:
            trace.finalize(path="NEW_TOPIC")


def answer_refinement(
    state: SessionState,
    entry_id: str,
    detail: str,
    index: dict,
    fusion: FusionPipeline,
    generator: AnswerGenerator,
    verifier: GroundingVerifier,
    llm_counter=None,
    trace: TurnTrace | None = None,
) -> TurnResult:
    if trace is not None and trace.wall_start == 0.0:
        trace.begin()

    try:
        # a. entry = state.ledger.entries[entry_id]; if missing, raise KeyError
        if state.ledger is None or entry_id not in state.ledger.entries:
            raise KeyError(f"Entry {entry_id!r} not found in ledger")
        entry = state.ledger.entries[entry_id]

        # e. Fusion MUST have been constructed with a real ContradictionScreen
        if fusion.contradiction_screen is None:
            raise ValueError("answer_refinement requires FusionPipeline with an active ContradictionScreen")

        initial_calls = llm_counter.calls if llm_counter is not None else 0

        # b. Per-turn deepcopy of index["chunk_lookup"]
        lookup = {k: copy.deepcopy(v) for k, v in index["chunk_lookup"].items()}

        # c. query = entry.topic + " " + detail
        query = entry.topic + " " + detail

        # d. candidate_ids = index["retrieve"](query, top_k=10)
        if trace:
            with trace.start_stage("retrieval"):
                candidate_ids = index["retrieve"](query, top_k=10)
            trace.record_retrieval(
                action="RETRIEVE",
                query=query,
                chunk_ids_returned=candidate_ids,
                reason="refinement_retrieval",
            )
            with trace.start_stage("fusion", query=query, candidate_count=len(candidate_ids)):
                fused = fusion.fuse(query, [candidate_ids], lookup)
        else:
            candidate_ids = index["retrieve"](query, top_k=10)
            fused = fusion.fuse(query, [candidate_ids], lookup)

        # f. new_chunks = [c for c in fused.chunks if c.chunk_id not in entry.evidence]
        new_chunks = [c for c in fused.chunks if c.chunk_id not in entry.evidence]

        # g. If new_chunks is empty: return TurnResult with version unchanged, no generation
        if not new_chunks:
            empty_verif = VerificationResult(
                answer_text="",
                claims=[],
                all_verified=True,
                fabricated_citations=[],
                unsupported_claims=[],
                cherry_picks=[],
                decomposition_degraded=False,
            )
            calls = (llm_counter.calls - initial_calls) if llm_counter is not None else 0
            if trace:
                with trace.start_stage("render"):
                    rendered = render(entry)
                trace.finalize(path="REFINEMENT")
            else:
                rendered = render(entry)
            return TurnResult(
                text=rendered,
                verification=empty_verif,
                rejected=[],
                llm_calls=calls,
                path="REFINEMENT",
                contradiction_unresolved=False,
                skipped_duplicates=[],
                trace=trace,
                fused_chunk_ids=[c.chunk_id for c in fused.chunks],
            )

        # h. Explicit contradiction check across retained and new chunks
        retained_chunks = list(entry.evidence.values())
        if trace:
            with trace.start_stage("contradiction"):
                fusion.contradiction_screen.screen(query, retained_chunks + new_chunks)
            with trace.start_stage("generator"):
                entries = generator.generate(entry.topic, new_chunks, detail=detail)
        else:
            fusion.contradiction_screen.screen(query, retained_chunks + new_chunks)
            entries = generator.generate(entry.topic, new_chunks, detail=detail)

        # j. verification = verifier.verify_entries
        all_evidence = new_chunks + list(entry.evidence.values())
        extra_cids = {c.chunk_id for c in active_claims(entry)}
        if trace:
            with trace.start_stage("verifier"):
                verification = verifier.verify_entries(entries, all_evidence, extra_cited_chunk_ids=extra_cids)
        else:
            verification = verifier.verify_entries(entries, all_evidence, extra_cited_chunk_ids=extra_cids)

        # k. DEDUP before touching the ledger
        active_chunk_ids = {c.chunk_id for c in active_claims(entry)}
        genuinely_new_verdicts: list[ClaimVerdict] = []
        rejected_verdicts: list[ClaimVerdict] = []
        skipped_duplicates: list[str] = []
        contradiction_unresolved = False

        for verdict in verification.claims:
            if verdict.cherry_pick_violation:
                contradiction_unresolved = True

            if verdict.citation_exists and verdict.supported and not verdict.cherry_pick_violation:
                if verdict.cited_chunk_id in active_chunk_ids:
                    skipped_duplicates.append(verdict.claim_text)
                else:
                    genuinely_new_verdicts.append(verdict)
            else:
                rejected_verdicts.append(verdict)

        # l. SUPERSESSION: for each retained active claim whose chunk has contradiction_flag=True
        # and whose paired contradicts_chunk_id belongs to a genuinely new claim from step k:
        genuinely_new_chunk_ids = {v.cited_chunk_id for v in genuinely_new_verdicts if v.cited_chunk_id}
        retired_claim_texts: list[str] = []

        for claim in active_claims(entry):
            chunk = entry.evidence.get(claim.chunk_id)
            if chunk and getattr(chunk, "contradiction_flag", False) and getattr(chunk, "contradicts_chunk_id", None):
                if chunk.contradicts_chunk_id in genuinely_new_chunk_ids:
                    claim.status = "RETIRED"
                    retired_claim_texts.append(claim.claim)

        old_version = entry.version
        # m. If any new claims survive k/l or any claims were retired:
        if genuinely_new_verdicts or retired_claim_texts:
            entry.version += 1

            claims_by_chunk: dict[str, list[str]] = {}
            for v in genuinely_new_verdicts:
                cid = v.cited_chunk_id or ""
                if v.claim_text not in claims_by_chunk.setdefault(cid, []):
                    claims_by_chunk[cid].append(v.claim_text)

            new_claims: list[LedgerClaim] = []
            for cid, texts in claims_by_chunk.items():
                new_claim = LedgerClaim(
                    claim=_merge_claim_texts(texts),
                    chunk_id=cid,
                    origin_version=entry.version,
                    status="ACTIVE",
                )
                entry.claims.append(new_claim)
                new_claims.append(new_claim)

            entry.details.append(detail)
            for c in new_chunks:
                entry.evidence[c.chunk_id] = dataclasses.replace(c)

            if trace:
                with trace.start_stage("render"):
                    entry_rendered = render(entry)
                trace.record_version_transition(
                    entry_id=entry.entry_id,
                    old_version=old_version,
                    new_version=entry.version,
                    claims_added=[c.claim for c in new_claims],
                    claims_retired=retired_claim_texts,
                    cited_chunk_ids=[c.chunk_id for c in active_claims(entry)],
                )
            else:
                entry_rendered = render(entry)

            state.ledger.history.append({
                "turn": len(state.ledger.history) + 1,
                "action": "REFINEMENT",
                "added": [c.claim for c in new_claims],
                "retired": retired_claim_texts,
            })
            state.last_answer_topic = entry.topic + " " + " ".join(c.claim for c in active_claims(entry))
        else:
            if trace:
                with trace.start_stage("render"):
                    entry_rendered = render(entry)
            else:
                entry_rendered = render(entry)

        total_calls = (llm_counter.calls - initial_calls) if llm_counter is not None else 0

        if trace:
            trace.finalize(path="REFINEMENT")

        return TurnResult(
            text=entry_rendered,
            verification=verification,
            rejected=rejected_verdicts,
            llm_calls=total_calls,
            path="REFINEMENT",
            contradiction_unresolved=contradiction_unresolved,
            skipped_duplicates=skipped_duplicates,
            trace=trace,
            fused_chunk_ids=[c.chunk_id for c in fused.chunks],
        )
    finally:
        if trace is not None and trace.wall_end == 0.0:
            trace.finalize(path="REFINEMENT")


def answer_suppress(
    state: SessionState,
    instruction: str = "",
    entry_id: str | None = None,
    trace: TurnTrace | None = None,
) -> TurnResult:
    """Renders the existing answer without new retrieval or generation for reformat requests."""
    if trace is not None and trace.wall_start == 0.0:
        trace.begin()

    try:
        if state.ledger is None or not state.ledger.entries:
            raise KeyError("No entries in ledger to reformat/suppress")
        target_entry_id = entry_id or list(state.ledger.entries.keys())[-1]
        if target_entry_id not in state.ledger.entries:
            raise KeyError(f"Entry {target_entry_id!r} not found in ledger")

        entry = state.ledger.entries[target_entry_id]

        if trace:
            with trace.start_stage("render", instruction=instruction):
                rendered_text = render(entry)
            trace.record_retrieval(
                action="SUPPRESS",
                trigger="suppression",
                query=instruction,
                reason="presentation-only cue, no retrieval required",
            )
        else:
            rendered_text = render(entry)

        empty_verif = VerificationResult(
            answer_text=rendered_text,
            claims=[],
            all_verified=True,
            fabricated_citations=[],
            unsupported_claims=[],
            cherry_picks=[],
            decomposition_degraded=False,
        )

        if trace:
            trace.finalize(path="SUPPRESS")

        return TurnResult(
            text=rendered_text,
            verification=empty_verif,
            rejected=[],
            llm_calls=0,
            path="SUPPRESS",
            contradiction_unresolved=False,
            skipped_duplicates=[],
            trace=trace,
        )
    finally:
        if trace is not None and trace.wall_end == 0.0:
            trace.finalize(path="SUPPRESS")

