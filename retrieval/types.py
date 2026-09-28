from dataclasses import dataclass


@dataclass
class EvidenceChunk:
    chunk_id: str
    doc_id: str          # e.g. "DOC_04" — comes from corpus metadata, never hardcoded in production code
    section: str         # e.g. "§1"
    text: str
    retrieval_score: float    # original score from whichever retriever(s) found it
    rerank_score: float | None = None
    contradiction_flag: bool = False
    contradicts_chunk_id: str | None = None
