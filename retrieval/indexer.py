"""Corpus indexing and vector store building for the streaming RAG pipeline."""

import glob
import os
import re
from typing import Callable, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding
from retrieval.types import EvidenceChunk


def parse_corpus_chunks(corpus_dir: str) -> list[dict]:
    """
    Parses sections from corpus/*.md into structured chunks:
    (chunk_id, doc_id, section, text)
    """
    chunks = []
    md_files = glob.glob(os.path.join(corpus_dir, "*.md"))
    for filepath in md_files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        doc_id_match = re.search(r"doc_id:\s*([A-Za-z0-9_]+)", content)
        doc_id = doc_id_match.group(1) if doc_id_match else os.path.basename(filepath)

        title_match = re.search(r"title:\s*([^\n]+)", content)
        title = title_match.group(1).strip() if title_match else ""

        # Split by section headers e.g. ## §1
        sections = re.split(r"(##\s*§\d+[^#\n]*)", content)
        if len(sections) <= 1:
            chunk_id = f"{doc_id}_full"
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "section": "§0",
                "text": content.strip()
            })
            continue

        for i in range(1, len(sections), 2):
            header = sections[i].strip()
            body = sections[i + 1].strip() if i + 1 < len(sections) else ""
            section_match = re.search(r"§\d+", header)
            sec_num = section_match.group(0) if section_match else "§1"
            chunk_id = f"{doc_id}_{sec_num}"
            section_content = f"{header}\n{body}".strip()
            full_text = f"{title}\n{section_content}".strip() if title else section_content
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "section": sec_num,
                "text": full_text
            })

    return chunks


def build_index(corpus_dir: str, embed_model_name: str = "BAAI/bge-small-en-v1.5") -> dict[str, Any]:
    """
    Builds an in-memory vector index over markdown files in corpus_dir using FastEmbed and QdrantClient.
    Returns:
      - client: QdrantClient
      - embed_model: TextEmbedding
      - collection_name: str
      - chunks: list[dict]
      - chunk_lookup: dict[str, EvidenceChunk]
      - retrieve: callable(query: str, top_k: int) -> list[str]
    """
    raw_chunks = parse_corpus_chunks(corpus_dir)

    embed_model = TextEmbedding(model_name=embed_model_name)
    texts = [c["text"] for c in raw_chunks]
    embeddings = list(embed_model.embed(texts))
    dim = len(embeddings[0])

    client = QdrantClient(":memory:")
    collection_name = "dev_corpus"
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    chunk_lookup: dict[str, EvidenceChunk] = {}
    points = []
    for idx, (c, emb) in enumerate(zip(raw_chunks, embeddings)):
        points.append(
            PointStruct(
                id=idx,
                vector=emb.tolist(),
                payload={
                    "chunk_id": c["chunk_id"],
                    "doc_id": c["doc_id"],
                    "section": c["section"],
                    "text": c["text"],
                },
            )
        )
        chunk_lookup[c["chunk_id"]] = EvidenceChunk(
            chunk_id=c["chunk_id"],
            doc_id=c["doc_id"],
            section=c["section"],
            text=c["text"],
            retrieval_score=1.0,
        )

    client.upsert(collection_name=collection_name, points=points)

    def retrieve(query: str, top_k: int = 10) -> list[str]:
        q_emb = list(embed_model.embed([query]))[0].tolist()
        res = client.query_points(
            collection_name=collection_name,
            query=q_emb,
            limit=top_k,
        )
        return [point.payload["chunk_id"] for point in res.points]

    return {
        "client": client,
        "embed_model": embed_model,
        "collection_name": collection_name,
        "chunks": raw_chunks,
        "chunk_lookup": chunk_lookup,
        "retrieve": retrieve,
    }
