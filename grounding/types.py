from dataclasses import dataclass


@dataclass
class ClaimVerdict:
    claim_text: str
    cited_chunk_id: str | None       # what the answer claimed to cite, may be None if no citation given
    citation_exists: bool             # is cited_chunk_id actually in the evidence set?
    supported: bool                   # did entailment check pass?
    reason: str
    cherry_pick_violation: bool = False   # True if cited chunk has contradiction_flag and the paired claim wasn't surfaced


@dataclass
class VerificationResult:
    answer_text: str
    claims: list[ClaimVerdict]
    all_verified: bool                 # True iff every claim passed citation-exists + supported + no cherry-pick violation
    fabricated_citations: list[str]      # claim_texts with citation_exists=False
    unsupported_claims: list[str]        # claim_texts with supported=False
    cherry_picks: list[str]             # claim_texts with cherry_pick_violation=True
