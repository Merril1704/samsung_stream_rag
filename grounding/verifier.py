"""GroundingVerifier: orchestrates claim decomposition, citation attribution,
entailment checking, and contradiction/cherry-pick detection.
"""
from controller.llm_factory import get_llm_client
from retrieval.types import EvidenceChunk
from grounding.types import ClaimVerdict, VerificationResult
from grounding.decomposer import ClaimDecomposer
from grounding.entailment import EntailmentChecker


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
        """
        cited_chunks_by_claim: maps each atomic claim (after decomposition)
        to the chunk_id the answer-generation step claims supports it.
        (How the synthesis step produces this mapping is out of scope for
        this stage — assume it's provided as input for now; NOTE this
        as an open interface question for whenever the synthesis/answer-
        generation component is built.)

        Steps:
        1. claims = self.decomposer.decompose(answer_text)
        2. For each claim:
           a. cited_id = cited_chunks_by_claim.get(claim)
           b. citation_exists = cited_id is not None and cited_id in {c.chunk_id for c in evidence}
           c. if not citation_exists -> ClaimVerdict(supported=False, reason="fabricated citation")
           d. else: find the EvidenceChunk for cited_id, run self.entailment.check(claim, chunk.text)
           e. if that chunk has contradiction_flag=True, check whether ANY claim in claims
              corresponds to the paired chunk (chunk.contradicts_chunk_id) being cited too.
              If not, set cherry_pick_violation=True on this claim's verdict.
        3. Aggregate into VerificationResult: all_verified is True only if
           every claim has citation_exists=True, supported=True, and
           cherry_pick_violation=False.
        """
        claims = self.decomposer.decompose(answer_text)
        if not claims:
            return VerificationResult(
                answer_text=answer_text,
                claims=[],
                all_verified=True,
                fabricated_citations=[],
                unsupported_claims=[],
                cherry_picks=[],
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

                cherry_pick_violation = False
                if getattr(chunk, "contradiction_flag", False) and getattr(chunk, "contradicts_chunk_id", None):
                    paired_id = chunk.contradicts_chunk_id
                    if paired_id not in all_cited_chunk_ids:
                        cherry_pick_violation = True

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
        )
