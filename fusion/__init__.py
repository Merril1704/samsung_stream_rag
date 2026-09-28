"""Stage 3 fusion package.

Exports:
  EvidenceChunk           – chunk data structure
  FusedResult             – output of fusion pipeline
  reciprocal_rank_fusion  – pure RRF function
  CrossEncoderReranker    – local cross-encoder reranker
  ContradictionScreen     – contradiction detector (rule + LLM)
  FusionPipeline          – end-to-end fusion orchestrator
"""
from .types import EvidenceChunk, FusedResult
from .rrf import reciprocal_rank_fusion
from .reranker import CrossEncoderReranker
from .contradiction import ContradictionScreen
from .fusion_pipeline import FusionPipeline

__all__ = [
    "EvidenceChunk",
    "FusedResult",
    "reciprocal_rank_fusion",
    "CrossEncoderReranker",
    "ContradictionScreen",
    "FusionPipeline",
]
