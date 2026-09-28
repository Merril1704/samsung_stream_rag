"""Fusion pipeline orchestrator.

Orchestrates:
  1. Reciprocal Rank Fusion over candidate chunk_id lists
  2. Resolving chunk_ids to EvidenceChunk objects
  3. Insufficient evidence detection
  4. Cross-encoder reranking
  5. Contradiction screening
"""
from .types import EvidenceChunk, FusedResult
from .rrf import reciprocal_rank_fusion
from .reranker import CrossEncoderReranker
from .contradiction import ContradictionScreen


class FusionPipeline:
    def __init__(self, reranker: CrossEncoderReranker | None = None, contradiction_screen: ContradictionScreen | None = None):
        self.reranker = reranker or CrossEncoderReranker()
        self.contradiction_screen = contradiction_screen  # may be None -> skip contradiction screening

    def fuse(
        self,
        sub_query: str,
        ranked_lists: list[list[str]],
        chunk_lookup: dict[str, EvidenceChunk],
        top_n: int = 10,
    ) -> FusedResult:
        """
        1. Run reciprocal_rank_fusion(ranked_lists) to get fused chunk_id order.
        2. Resolve chunk_ids to EvidenceChunk objects via chunk_lookup.
        3. If the fused list is empty -> return FusedResult(sub_query, [], insufficient_evidence=True).
        4. Run self.reranker.rerank(sub_query, chunks, top_n=top_n).
        5. If self.contradiction_screen is not None, run it on the reranked top chunks;
           populate has_contradiction / contradiction_pairs and per-chunk flags accordingly.
        6. Return the populated FusedResult.
        """
        fused_scores = reciprocal_rank_fusion(ranked_lists)
        if not fused_scores:
            return FusedResult(sub_query=sub_query, chunks=[], insufficient_evidence=True)

        chunks: list[EvidenceChunk] = []
        fused_rank_map: dict[str, int] = {}
        for rank, (chunk_id, score) in enumerate(fused_scores):
            if chunk_id in chunk_lookup:
                # Make a shallow copy to prevent state contamination across queries
                c = chunk_lookup[chunk_id]
                chunk_copy = EvidenceChunk(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    section=c.section,
                    text=c.text,
                    retrieval_score=c.retrieval_score,
                    rerank_score=c.rerank_score,
                    contradiction_flag=c.contradiction_flag,
                    contradicts_chunk_id=c.contradicts_chunk_id,
                )
                chunks.append(chunk_copy)
                fused_rank_map[chunk_id] = rank

        if not chunks:
            return FusedResult(sub_query=sub_query, chunks=[], insufficient_evidence=True)

        reranked_chunks = self.reranker.rerank(
            query=sub_query,
            chunks=chunks,
            top_n=top_n,
            fused_rank_map=fused_rank_map,
        )

        has_contradiction = False
        contradiction_pairs: list[tuple[str, str]] = []

        if self.contradiction_screen is not None:
            contradiction_pairs = self.contradiction_screen.screen(sub_query, reranked_chunks)
            has_contradiction = len(contradiction_pairs) > 0

        # Check for coverage gap / insufficient evidence:
        # e.g., if top-1 chunk explicitly states information is not available / undocumented
        # or if the chunk text itself indicates an unanswerable coverage gap
        insufficient = False
        if reranked_chunks:
            top_text = reranked_chunks[0].text.lower()
            if (
                "is not available in this corpus version" in top_text
                or "coverage gap" in top_text
                or "undocumented" in top_text
            ):
                insufficient = True

        return FusedResult(
            sub_query=sub_query,
            chunks=reranked_chunks,
            has_contradiction=has_contradiction,
            contradiction_pairs=contradiction_pairs,
            insufficient_evidence=insufficient,
        )
