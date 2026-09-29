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


@dataclass
class TurnResult:
    text: str
    verification: VerificationResult
    rejected: list[ClaimVerdict]
    llm_calls: int
    path: str = "NEW_TOPIC"
    contradiction_unresolved: bool = False
    skipped_duplicates: list[str] = field(default_factory=list)


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
) -> TurnResult:
    # Per-turn defensive copy: index["chunk_lookup"] must never be mutated
    lookup = {k: copy.deepcopy(v) for k, v in index["chunk_lookup"].items()}

    # Retrieve candidate chunk IDs
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
        return TurnResult(
            text="",
            verification=empty_verif,
            rejected=[],
            llm_calls=calls,
            path="NEW_TOPIC",
            contradiction_unresolved=False,
        )

    # Generate claim entries
    entries = generator.generate(topic, fused.chunks)

    # Verify claim entries
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
        rendered_text = render(entry)

    # Set last_answer_topic so suppression path in controller can trigger on follow-ups
    state.last_answer_topic = topic + " " + " ".join(c.claim for c in passed_claims)

    total_calls = (llm_counter.calls - initial_calls) if llm_counter is not None else 0

    return TurnResult(
        text=rendered_text,
        verification=verification,
        rejected=rejected_verdicts,
        llm_calls=total_calls,
        path="NEW_TOPIC",
        contradiction_unresolved=contradiction_unresolved,
    )


def answer_refinement(
    state: SessionState,
    entry_id: str,
    detail: str,
    index: dict,
    fusion: FusionPipeline,
    generator: AnswerGenerator,
    verifier: GroundingVerifier,
    llm_counter=None,
) -> TurnResult:
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
    candidate_ids = index["retrieve"](query, top_k=10)

    # e. fused = fusion.fuse(...)
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
        return TurnResult(
            text=render(entry),
            verification=empty_verif,
            rejected=[],
            llm_calls=calls,
            path="REFINEMENT",
            contradiction_unresolved=False,
            skipped_duplicates=[],
        )

    # h. Explicit contradiction check across retained and new chunks
    retained_chunks = list(entry.evidence.values())
    fusion.contradiction_screen.screen(query, retained_chunks + new_chunks)

    # i. entries = generator.generate(entry.topic, new_chunks, detail=detail)
    entries = generator.generate(entry.topic, new_chunks, detail=detail)

    # j. verification = verifier.verify_entries
    all_evidence = new_chunks + list(entry.evidence.values())
    extra_cids = {c.chunk_id for c in active_claims(entry)}
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

        entry_rendered = render(entry)
        state.ledger.history.append({
            "turn": len(state.ledger.history) + 1,
            "action": "REFINEMENT",
            "added": [c.claim for c in new_claims],
            "retired": retired_claim_texts,
        })
        state.last_answer_topic = entry.topic + " " + " ".join(c.claim for c in active_claims(entry))
    else:
        entry_rendered = render(entry)

    total_calls = (llm_counter.calls - initial_calls) if llm_counter is not None else 0

    return TurnResult(
        text=entry_rendered,
        verification=verification,
        rejected=rejected_verdicts,
        llm_calls=total_calls,
        path="REFINEMENT",
        contradiction_unresolved=contradiction_unresolved,
        skipped_duplicates=skipped_duplicates,
    )
