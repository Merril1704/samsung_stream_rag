"""GroundingVerifier: orchestrates claim decomposition, citation attribution,
entailment checking, and contradiction/cherry-pick detection.
"""
from controller.llm_factory import get_llm_client
from retrieval.types import EvidenceChunk
from grounding.types import ClaimEntry, ClaimVerdict, VerificationResult
from grounding.decomposer import ClaimDecomposer
from grounding.entailment import EntailmentChecker

CONJUNCTION_PATTERNS = [
    " and ", " or ", ";", ", and", " as well as ",
    " but ", " while ", " whereas ",
]


def _lookup_cited_id(claim: str, cited_chunks_by_claim: dict[str, str]) -> str | None:
    if not cited_chunks_by_claim:
        return None
    if claim in cited_chunks_by_claim:
        return cited_chunks_by_claim[claim]
    c_clean = claim.strip()
    if c_clean in cited_chunks_by_claim:
        return cited_chunks_by_claim[c_clean]
    c_norm = c_clean.rstrip(".?!;:").strip()
    for k, v in cited_chunks_by_claim.items():
        if k.strip().rstrip(".?!;:").strip() == c_norm:
            return v
    return None


def _is_cherry_pick(chunk: EvidenceChunk, all_cited_chunk_ids: set[str]) -> bool:
    if getattr(chunk, "contradiction_flag", False) and getattr(chunk, "contradicts_chunk_id", None):
        paired_id = chunk.contradicts_chunk_id
        if paired_id not in all_cited_chunk_ids:
            return True
    return False


class GroundingVerifier:
    def __init__(self, llm_client=None):
        client = llm_client or get_llm_client()
        self.decomposer = ClaimDecomposer(client)
        self.entailment = EntailmentChecker(client)

    def verify(
        self,
        answer_text: str,
        cited_chunks_by_claim: dict[str, str],
        evidence: list[EvidenceChunk],
    ) -> VerificationResult:
        claims = self.decomposer.decompose(answer_text)
        if not claims:
            return VerificationResult(
                answer_text=answer_text,
                claims=[],
                all_verified=True,
                fabricated_citations=[],
                unsupported_claims=[],
                cherry_picks=[],
                decomposition_degraded=self.decomposer.last_fallback,
            )

        evidence_map = {c.chunk_id: c for c in evidence}

        # Pre-resolve cited IDs for all claims to check whether paired contradiction chunks are cited
        claim_cited_map = {claim: _lookup_cited_id(claim, cited_chunks_by_claim) for claim in claims}
        all_cited_chunk_ids = {cid for cid in claim_cited_map.values() if cid is not None}

        verdicts: list[ClaimVerdict] = []
        for claim in claims:
            cited_id = claim_cited_map[claim]
            citation_exists = cited_id is not None and cited_id in evidence_map

            if not citation_exists:
                verdicts.append(
                    ClaimVerdict(
                        claim_text=claim,
                        cited_chunk_id=cited_id,
                        citation_exists=False,
                        supported=False,
                        reason="fabricated citation",
                        cherry_pick_violation=False,
                    )
                )
            else:
                chunk = evidence_map[cited_id]
                supported, reason = self.entailment.check(claim, chunk.text)
                cherry_pick_violation = _is_cherry_pick(chunk, all_cited_chunk_ids)

                verdicts.append(
                    ClaimVerdict(
                        claim_text=claim,
                        cited_chunk_id=cited_id,
                        citation_exists=True,
                        supported=supported,
                        reason=reason,
                        cherry_pick_violation=cherry_pick_violation,
                    )
                )

        all_verified = all(
            v.citation_exists and v.supported and not v.cherry_pick_violation
            for v in verdicts
        )
        fabricated_citations = [v.claim_text for v in verdicts if not v.citation_exists]
        unsupported_claims = [v.claim_text for v in verdicts if not v.supported]
        cherry_picks = [v.claim_text for v in verdicts if v.cherry_pick_violation]

        return VerificationResult(
            answer_text=answer_text,
            claims=verdicts,
            all_verified=all_verified,
            fabricated_citations=fabricated_citations,
            unsupported_claims=unsupported_claims,
            cherry_picks=cherry_picks,
            decomposition_degraded=self.decomposer.last_fallback,
        )

    def verify_entries(
        self,
        entries: list[ClaimEntry],
        evidence: list[EvidenceChunk],
    ) -> VerificationResult:
        """
        Verifies a list of ClaimEntry objects against evidence.
        - Decomposes entries whose claims contain conjunctions; atomic claims skip decomposer.
        - Checks citation existence and entailment per atomic claim.
        - Detects cherry-pick violations if a contradiction chunk is cited without its pair.
        """
        if not entries:
            return VerificationResult(
                answer_text="",
                claims=[],
                all_verified=True,
                fabricated_citations=[],
                unsupported_claims=[],
                cherry_picks=[],
                decomposition_degraded=False,
            )

        evidence_map = {c.chunk_id: c for c in evidence}
        atomic_items: list[tuple[str, str]] = []
        any_degraded = False

        for entry in entries:
            claim_lower = entry.claim.lower()
            if any(p in claim_lower for p in CONJUNCTION_PATTERNS):
                sub_claims = self.decomposer.decompose(entry.claim)
                if self.decomposer.last_fallback:
                    any_degraded = True
                for sc in sub_claims:
                    atomic_items.append((sc, entry.chunk_id))
            else:
                atomic_items.append((entry.claim, entry.chunk_id))

        all_cited_chunk_ids = {cid for _, cid in atomic_items if cid is not None}

        verdicts: list[ClaimVerdict] = []
        for claim_text, cited_id in atomic_items:
            citation_exists = cited_id is not None and cited_id in evidence_map

            if not citation_exists:
                verdicts.append(
                    ClaimVerdict(
                        claim_text=claim_text,
                        cited_chunk_id=cited_id,
                        citation_exists=False,
                        supported=False,
                        reason="fabricated citation",
                        cherry_pick_violation=False,
                    )
                )
            else:
                chunk = evidence_map[cited_id]
                supported, reason = self.entailment.check(claim_text, chunk.text)
                cherry_pick_violation = _is_cherry_pick(chunk, all_cited_chunk_ids)

                verdicts.append(
                    ClaimVerdict(
                        claim_text=claim_text,
                        cited_chunk_id=cited_id,
                        citation_exists=True,
                        supported=supported,
                        reason=reason,
                        cherry_pick_violation=cherry_pick_violation,
                    )
                )

        all_verified = all(
            v.citation_exists and v.supported and not v.cherry_pick_violation
            for v in verdicts
        )
        fabricated_citations = [v.claim_text for v in verdicts if not v.citation_exists]
        unsupported_claims = [v.claim_text for v in verdicts if not v.supported]
        cherry_picks = [v.claim_text for v in verdicts if v.cherry_pick_violation]
        answer_text = " ".join(e.claim for e in entries)

        return VerificationResult(
            answer_text=answer_text,
            claims=verdicts,
            all_verified=all_verified,
            fabricated_citations=fabricated_citations,
            unsupported_claims=unsupported_claims,
            cherry_picks=cherry_picks,
            decomposition_degraded=any_degraded,
        )

