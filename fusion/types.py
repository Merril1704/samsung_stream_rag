from dataclasses import dataclass, field
from retrieval.types import EvidenceChunk


@dataclass
class FusedResult:
    sub_query: str
    chunks: list[EvidenceChunk]         # final ranked order after fusion + rerank
    has_contradiction: bool = False
    contradiction_pairs: list[tuple[str, str]] = field(default_factory=list)  # (chunk_id, chunk_id)
    insufficient_evidence: bool = False   # True if fused result is empty/very low confidence


__all__ = ["EvidenceChunk", "FusedResult"]
