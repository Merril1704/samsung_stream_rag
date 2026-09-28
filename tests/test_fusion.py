"""Acceptance tests for Stage 3 — Fusion and Contradiction Screening.

Automated pytest tests validating RRF, cross-encoder reranking, contradiction
screening, and coverage gap detection on the dev corpus.
"""
import os
import pytest
from fusion.rrf import reciprocal_rank_fusion
from fusion.types import EvidenceChunk, FusedResult
from fusion.reranker import CrossEncoderReranker
from fusion.contradiction import ContradictionScreen
from fusion.fusion_pipeline import FusionPipeline
from controller.llm_factory import get_llm_client


def test_rrf_hand_computed_example():
    """Two ranked lists with known overlap; verify RRF output matches hand-computed scores."""
    lists = [["a", "b", "c"], ["b", "c", "a"]]
    result = reciprocal_rank_fusion(lists, k=60)
    scores = dict(result)
    # a: rank0 in list1 (1/60) + rank2 in list2 (1/62)
    # b: rank1 in list1 (1/61) + rank0 in list2 (1/60)
    # c: rank2 in list1 (1/62) + rank1 in list2 (1/61)
    assert abs(scores["a"] - (1 / 60 + 1 / 62)) < 1e-9
    assert abs(scores["b"] - (1 / 61 + 1 / 60)) < 1e-9
    assert abs(scores["c"] - (1 / 62 + 1 / 61)) < 1e-9


def test_rrf_favors_consistently_ranked_items():
    """An item appearing near the top of both lists should outrank one appearing top of only one list."""
    lists = [["x", "y"], ["x", "z"]]
    result = reciprocal_rank_fusion(lists, k=60)
    ranked_ids = [chunk_id for chunk_id, _ in result]
    assert ranked_ids[0] == "x"


def test_reranker_corrects_distractor_above_true_answer(indexed_dev_corpus):
    """
    DOC_03 is the corpus's deliberate distractor (Bangalore venues).
    Construct a case where initial lexical/dense retrieval places DOC_03 above
    the correct Pune venue (DOC_02), and confirm cross-encoder reranking corrects the order.
    """
    chunk_lookup = indexed_dev_corpus["chunk_lookup"]
    query = "Customer workshop venue in Pune for 30 people"

    # Pre-rerank ranked list where distractor DOC_03_§2 is placed ahead of DOC_02_§2
    distractor_id = "DOC_03_§2"
    correct_id = "DOC_02_§2"

    candidate_chunks = [
        chunk_lookup[distractor_id],
        chunk_lookup[correct_id],
    ]

    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(query, candidate_chunks, top_n=2)

    # Cross-encoder should correctly rank DOC_02 (Pune venue) above DOC_03 (Bangalore venue)
    assert reranked[0].chunk_id == correct_id
    assert reranked[1].chunk_id == distractor_id
    assert reranked[0].rerank_score > reranked[1].rerank_score


def test_contradiction_detected_doc04_doc10():
    """
    A query touching the refund/cancellation topic must surface BOTH
    DOC_04 §1 and DOC_10 §2 in the fused result, with contradiction_flag=True
    on both and contradicts_chunk_id pointing at each other.
    """
    chunk_04 = EvidenceChunk(
        chunk_id="DOC_04_§1",
        doc_id="DOC_04",
        section="§1",
        text="Bookings cancelled more than 30 days before the event date receive a full refund of any deposit paid. Cancellations made 15-30 days before the event receive a 50% refund. Cancellations made fewer than 15 days before the event are non-refundable.",
        retrieval_score=0.9,
    )
    chunk_10 = EvidenceChunk(
        chunk_id="DOC_10_§2",
        doc_id="DOC_10",
        section="§2",
        text="Event vendors contracted directly (catering, AV, staging) may enforce a 14-day cancellation notice period for full refund eligibility, which may differ from the internal Event Cancellation & Refund Policy's 30-day window; employees should confirm vendor-specific terms at time of contracting.",
        retrieval_score=0.85,
    )

    lookup = {chunk_04.chunk_id: chunk_04, chunk_10.chunk_id: chunk_10}
    screen = ContradictionScreen(llm_client=get_llm_client())
    pipeline = FusionPipeline(contradiction_screen=screen)

    res = pipeline.fuse(
        sub_query="cancellation notice period and refund terms for event vendors",
        ranked_lists=[["DOC_04_§1", "DOC_10_§2"], ["DOC_10_§2", "DOC_04_§1"]],
        chunk_lookup=lookup,
    )

    assert res.has_contradiction is True
    assert len(res.contradiction_pairs) == 1
    pair = res.contradiction_pairs[0]
    assert ("DOC_04_§1" in pair) and ("DOC_10_§2" in pair)

    c04 = next(c for c in res.chunks if c.chunk_id == "DOC_04_§1")
    c10 = next(c for c in res.chunks if c.chunk_id == "DOC_10_§2")
    assert c04.contradiction_flag is True
    assert c04.contradicts_chunk_id == "DOC_10_§2"
    assert c10.contradiction_flag is True
    assert c10.contradicts_chunk_id == "DOC_04_§1"


def test_contradiction_screen_no_false_positive_on_related_nonconflicting_chunks():
    """
    Two chunks discussing the same general topic (e.g. venue capacity)
    without actually disagreeing must NOT be flagged as a contradiction.
    """
    chunk_a = EvidenceChunk(
        chunk_id="DOC_02_§1",
        doc_id="DOC_02",
        section="§1",
        text="Venue A — Riverside Business Center. Capacity: 40 seated / 60 standing. Includes projector, whiteboard, and standard AV.",
        retrieval_score=0.9,
    )
    chunk_b = EvidenceChunk(
        chunk_id="DOC_02_§2",
        doc_id="DOC_02",
        section="§2",
        text="Venue B — Grand Hall Conference Suite. Capacity: 30 seated theatre-style. Offers breakout rooms for up to 3 concurrent sessions.",
        retrieval_score=0.88,
    )

    screen = ContradictionScreen(llm_client=get_llm_client())
    # Both chunks have doc_id DOC_02; even if different doc_ids, different subjects should not contradict
    pairs = screen.screen("venue capacity for meetings", [chunk_a, chunk_b])
    assert len(pairs) == 0
    assert chunk_a.contradiction_flag is False
    assert chunk_b.contradiction_flag is False


def test_coverage_gap_returns_insufficient_evidence_not_fabrication():
    """
    A query targeting DOC_05 §3's deliberate coverage gap must result in
    FusedResult.insufficient_evidence=True, never a confident assertion.
    """
    chunk_05_3 = EvidenceChunk(
        chunk_id="DOC_05_§3",
        doc_id="DOC_05",
        section="§3",
        text="Venue B and Venue C accept external catering vendors without restriction beyond the standard notice periods in §2. Documentation of Venue A's dietary-accommodation and external-vendor policy is not available in this corpus version; confirm directly with venue management before booking.",
        retrieval_score=0.95,
    )

    lookup = {chunk_05_3.chunk_id: chunk_05_3}
    pipeline = FusionPipeline()

    res = pipeline.fuse(
        sub_query="What is Venue A's dietary accommodation and external catering policy?",
        ranked_lists=[["DOC_05_§3"]],
        chunk_lookup=lookup,
    )

    assert res.insufficient_evidence is True


def test_contradiction_pairs_never_silently_drop_either_side():
    """
    When a contradiction is confirmed, both chunks must remain present in
    FusedResult.chunks — neither is removed or suppressed.
    """
    chunk_04 = EvidenceChunk(
        chunk_id="DOC_04_§1",
        doc_id="DOC_04",
        section="§1",
        text="Bookings cancelled more than 30 days before the event date receive a full refund of any deposit paid.",
        retrieval_score=0.9,
    )
    chunk_10 = EvidenceChunk(
        chunk_id="DOC_10_§2",
        doc_id="DOC_10",
        section="§2",
        text="Event vendors contracted directly may enforce a 14-day cancellation notice period for full refund eligibility, differing from the 30-day window.",
        retrieval_score=0.85,
    )

    lookup = {chunk_04.chunk_id: chunk_04, chunk_10.chunk_id: chunk_10}
    screen = ContradictionScreen(llm_client=get_llm_client())
    pipeline = FusionPipeline(contradiction_screen=screen)

    res = pipeline.fuse(
        sub_query="cancellation notice period",
        ranked_lists=[["DOC_04_§1", "DOC_10_§2"]],
        chunk_lookup=lookup,
    )

    # Both chunks must remain in res.chunks
    chunk_ids = [c.chunk_id for c in res.chunks]
    assert "DOC_04_§1" in chunk_ids
    assert "DOC_10_§2" in chunk_ids
    assert len(res.chunks) == 2
