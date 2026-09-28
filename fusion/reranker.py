"""Cross-Encoder Reranker wrapping FastEmbed TextCrossEncoder.

Uses Xenova/ms-marco-MiniLM-L-6-v2 to compute query-passage relevance scores.
Scores are assigned to chunk.rerank_score and used solely for sorting; they
are never combined, normalized, or rescaled with original retrieval_scores.
"""
from typing import Sequence
from .types import EvidenceChunk


class CrossEncoderReranker:
    def __init__(self, model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"):
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            self.model = TextCrossEncoder(model_name=model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load cross-encoder model '{model_name}': {e}") from e

    def rerank(
        self,
        query: str,
        chunks: list[EvidenceChunk],
        top_n: int = 10,
        fused_rank_map: dict[str, int] | None = None,
    ) -> list[EvidenceChunk]:
        """
        Scores each chunk.text against query using the cross-encoder.
        Sets chunk.rerank_score for every input chunk.
        Returns chunks sorted descending by rerank_score, truncated to top_n.
        If two chunks tie on rerank_score, breaks tie using the original RRF
        fused_score/rank order (if provided) or original input position.
        Does not normalize, rescale, or combine rerank_score with retrieval_score.
        """
        if not chunks:
            return []

        documents = [c.text for c in chunks]
        scores = list(self.model.rerank(query, documents))

        # Build fallback rank lookup for tie-breaking
        if fused_rank_map is None:
            fused_rank_map = {c.chunk_id: i for i, c in enumerate(chunks)}

        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = float(score)

        # Sort descending by rerank_score. Secondary key: lower fused_rank (earlier in RRF)
        sorted_chunks = sorted(
            chunks,
            key=lambda c: (
                c.rerank_score if c.rerank_score is not None else float("-inf"),
                -fused_rank_map.get(c.chunk_id, 999999),
            ),
            reverse=True,
        )

        return sorted_chunks[:top_n]
