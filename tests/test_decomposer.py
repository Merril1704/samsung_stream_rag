"""Acceptance tests for Stage 2 — Multi-Intent Decomposer.

All test utterances are deliberately generic/out-of-domain; no corpus-specific
entity names, vendor names, or vocabulary from corpus/raw/ appear here.
"""
import pytest
from decomposer.decomposer import Decomposer
from controller.llm_factory import get_llm_client


@pytest.fixture
def decomposer():
    return Decomposer(llm_client=get_llm_client())


def test_single_intent_not_split(decomposer):
    """Pure single-intent query: no connector, must not be split."""
    r = decomposer.decompose("what is the return window for this item")
    assert r.is_compound is False, f"Expected is_compound=False, got: {r}"
    assert r.sub_queries == ["what is the return window for this item"], (
        f"sub_queries should equal [utterance], got: {r.sub_queries}"
    )


def test_clean_two_intent_split(decomposer):
    """Clean two-part compound: both parts are standalone and content-rich."""
    r = decomposer.decompose("book a room and tell me the parking options")
    assert r.is_compound is True, f"Expected is_compound=True, got: {r}"
    assert len(r.sub_queries) == 2, f"Expected 2 sub_queries, got {len(r.sub_queries)}: {r.sub_queries}"


def test_triple_intent_split(decomposer):
    """Three independently-answerable requests joined by multiple connectors."""
    r = decomposer.decompose("book a room and also send me the menu and the parking info")
    assert r.is_compound is True, f"Expected is_compound=True, got: {r}"
    assert len(r.sub_queries) == 3, f"Expected 3 sub_queries, got {len(r.sub_queries)}: {r.sub_queries}"


def test_same_topic_compound_carried_forward_from_stage1_gap(decomposer):
    """
    This is the exact phrasing that Stage 1's model-based controller
    failed to flag as multi-intent (possible_multi_intent came back False).
    The decomposer MUST catch this independently of that flag.
    """
    r = decomposer.decompose("cancel my booking and tell me the refund policy")
    assert r.is_compound is True, f"Expected is_compound=True, got: {r}"
    assert len(r.sub_queries) == 2, (
        f"Expected 2 sub_queries, got {len(r.sub_queries)}: {r.sub_queries}"
    )


def test_over_fragmentation_guardrail_prepositional_fragment(decomposer):
    """
    Canonical over-fragmentation failure: 'what is attention in transformers'
    must NOT be split into 'what is attention' + 'in transformers'.
    The second part is a prepositional fragment, not a standalone request.
    """
    r = decomposer.decompose("what is attention in transformers")
    assert r.is_compound is False, f"Expected is_compound=False, got: {r}"


def test_entity_protection_capitalized_pair_not_split(decomposer):
    """
    Generic multi-word proper noun spanning the connector must not be
    split apart.  No real corpus entity names are used here.
    Accept either: no split at all, or a split that keeps
    'Silver Lake Conference Center' intact as one unit.
    """
    r = decomposer.decompose("tell me about Java and Silver Lake Conference Center availability")
    for part in r.sub_queries:
        assert not ("Silver Lake" in part and "Conference Center" not in part), (
            f"Entity 'Silver Lake Conference Center' was split across parts: {r.sub_queries}"
        )


def test_reformat_only_request_not_treated_as_compound(decomposer):
    """
    A reformatting request with 'in' must not be split into a compound.
    'can you say that in bullet points' → single intent.
    """
    r = decomposer.decompose("can you say that in bullet points")
    assert r.is_compound is False, f"Expected is_compound=False, got: {r}"


def test_rejected_splits_logged_on_failed_fragmentation(decomposer):
    """
    When a split is rejected (e.g. failing standalone check), the failed
    candidate(s) must be logged in rejected_splits for audit purposes.
    "explain the policy and details" splits on " and ", and "details" (1 content word)
    fails the standalone check.
    """
    r = decomposer.decompose("explain the policy and details")
    assert r.is_compound is False, f"Expected is_compound=False, got: {r}"
    assert len(r.rejected_splits) >= 1, (
        f"Expected at least one entry in rejected_splits, got: {r.rejected_splits!r}"
    )


def test_preposition_fragment_check_independent_of_word_count(decomposer):
    """
    Proves preposition-fragment check works on its own merits, independent
    of word-count: 'explain gradient descent and in neural network training'.
    The second part ('in neural network training') has 3 content words
    (neural, network, training) which would otherwise pass word-count,
    but it must fail strictly because it is a prepositional fragment starting with 'in'.
    """
    r = decomposer.decompose("explain gradient descent and in neural network training")
    assert r.is_compound is False, f"Expected is_compound=False, got: {r}"
    assert any("prepositional fragment" in s.lower() for s in r.rejected_splits), (
        f"Expected rejected_splits to mention prepositional fragment, got: {r.rejected_splits!r}"
    )
