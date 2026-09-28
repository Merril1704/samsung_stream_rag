from __future__ import annotations
import copy
from dataclasses import dataclass, field
from controller.types import SessionState
from grounding.types import ClaimVerdict, VerificationResult
from session.ledger import AnswerLedger, LedgerClaim, commit_entry
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

    passed_claims: list[LedgerClaim] = []
    rejected_verdicts: list[ClaimVerdict] = []
    contradiction_unresolved = False

    for verdict in verification.claims:
        if verdict.cherry_pick_violation:
            contradiction_unresolved = True

        if verdict.citation_exists and verdict.supported and not verdict.cherry_pick_violation:
            passed_claims.append(
                LedgerClaim(
                    claim=verdict.claim_text,
                    chunk_id=verdict.cited_chunk_id or "",
                    origin_version=1,
                    status="ACTIVE",
                )
            )
        else:
            rejected_verdicts.append(verdict)

    if state.ledger is None:
        state.ledger = AnswerLedger()

    rendered_text = ""
    if passed_claims:
        entry_id = f"entry_{len(state.ledger.entries) + 1}"
        entry = commit_entry(
            ledger=state.ledger,
            entry_id=entry_id,
            topic=topic,
            details=[],
            claims=passed_claims,
            evidence={c.chunk_id: c for c in fused.chunks},
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
