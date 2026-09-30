"""Quality assurance tests for the held-out evaluation dataset (eval/scenarios.py).

Validates:
1. Scenario uniqueness and identifier integrity.
2. Every expected chunk ID exists in the actual indexed corpus.
3. Every expected document ID exists in the corpus.
4. Every gold claim has valid, existing supporting chunk IDs.
5. Multi-intent cases have consistent sub-query and gold-chunk alignments.
6. Unanswerable cases specify zero expected chunk IDs and zero gold claims.
7. No evaluation prompt leaks or duplicates known regression test prompts.
"""
import pytest
from eval.scenarios import (
    get_eval_scenarios,
    get_scenario_by_id,
    get_scenarios_by_category,
    all_gold_chunk_ids,
    all_gold_doc_ids,
)


KNOWN_REGRESSION_PROMPTS = {
    # Scenario A and variants
    "i need to plan a customer workshop in pune for 30 people, and i need the cancellation policy and the catering options.",
    "i need to plan a customer workshop in pune for 30 people",
    "i need to plan a customer workshop in",
    "pune for 30 people, and i need",
    "the cancellation policy and the catering options.",
    # Scenario B and variants
    "summarize the travel reimbursement rule",
    "international, booking made after travel",
    "summarize the travel reimbursement rule international, booking made after travel",
    "i need to find the travel reimbursement rule for employees",
    # Suppression cues
    "please repeat your last answer in two bullet points.",
    "please repeat that in bullet points",
    "repeat that in bullet points",
    "make it shorter",
    # Stage 2 Decomposer test utterances
    "what is the return window for this item",
    "book a room and tell me the parking options",
    "book a room and also send me the menu and the parking info",
    "cancel my booking and tell me the refund policy",
    "what is attention in transformers",
    "visit new york and los angeles",
    "summarize this document and make it concise",
    "in neural network training and also optimize hyper parameters",
    "for corporate events and at the main hall",
}


def test_scenario_counts_and_ids():
    scenarios = get_eval_scenarios()
    assert 10 <= len(scenarios) <= 15, f"Expected 10-15 scenarios, got {len(scenarios)}"

    ids = [s.scenario_id for s in scenarios]
    assert len(ids) == len(set(ids)), f"Duplicate scenario IDs found: {ids}"
    assert all(s.scenario_id.startswith("eval_") for s in scenarios)


def test_all_categories_represented():
    categories = {s.category for s in get_eval_scenarios()}
    expected = {"single_intent", "multi_intent", "refinement", "unanswerable", "contradiction"}
    assert expected.issubset(categories), f"Missing categories: {expected - categories}"


def test_every_expected_chunk_exists_in_corpus(indexed_dev_corpus):
    chunk_lookup = indexed_dev_corpus["chunk_lookup"]
    gold_chunks = all_gold_chunk_ids()

    assert len(gold_chunks) > 0, "Evaluation dataset must reference at least one chunk"
    for cid in gold_chunks:
        assert cid in chunk_lookup, f"Gold chunk ID '{cid}' does not exist in indexed_dev_corpus!"


def test_every_expected_doc_id_exists(indexed_dev_corpus):
    chunk_lookup = indexed_dev_corpus["chunk_lookup"]
    existing_docs = {chunk.doc_id for chunk in chunk_lookup.values()}
    gold_docs = all_gold_doc_ids()

    for doc_id in gold_docs:
        assert doc_id in existing_docs, f"Gold doc ID '{doc_id}' does not exist in corpus!"


def test_gold_claims_integrity(indexed_dev_corpus):
    chunk_lookup = indexed_dev_corpus["chunk_lookup"]
    scenarios = get_eval_scenarios()

    total_claims = 0
    for s in scenarios:
        for turn in s.turns:
            for gc in turn.gold_claims:
                total_claims += 1
                assert gc.claim_id, f"Empty claim ID in scenario {s.scenario_id}"
                assert gc.text.strip(), f"Empty claim text for {gc.claim_id}"
                assert len(gc.supporting_chunk_ids) > 0, f"Claim {gc.claim_id} has no supporting chunk IDs"
                for cid in gc.supporting_chunk_ids:
                    assert cid in chunk_lookup, f"Claim {gc.claim_id} cites non-existent chunk '{cid}'"
                    # Supporting chunk must also be among expected chunk IDs
                    assert cid in turn.expected_chunk_ids, (
                        f"Claim {gc.claim_id} cites chunk {cid} not listed in turn.expected_chunk_ids"
                    )

    assert total_claims >= 15, f"Expected at least 15 atomic gold claims, got {total_claims}"


def test_multi_intent_alignment():
    multi_cases = get_scenarios_by_category("multi_intent")
    assert len(multi_cases) >= 2

    for s in multi_cases:
        for turn in s.turns:
            assert len(turn.expected_sub_queries) >= 2, (
                f"Multi-intent scenario {s.scenario_id} must have >= 2 expected sub-queries"
            )
            assert len(turn.sub_query_gold_chunks) == len(turn.expected_sub_queries), (
                f"Scenario {s.scenario_id} sub_query_gold_chunks count does not match expected_sub_queries"
            )
            # All sub-query chunks must be part of expected_chunk_ids
            union_chunks = set()
            for sq, cids in turn.sub_query_gold_chunks.items():
                assert sq in turn.expected_sub_queries
                assert len(cids) > 0, f"Sub-query '{sq}' has empty gold chunk list"
                union_chunks.update(cids)

            assert set(turn.expected_chunk_ids) == union_chunks, (
                f"Scenario {s.scenario_id} expected_chunk_ids does not equal union of sub-query chunks"
            )


def test_unanswerable_cases_have_zero_supporting_evidence():
    unans_cases = get_scenarios_by_category("unanswerable")
    assert len(unans_cases) >= 2

    for s in unans_cases:
        for turn in s.turns:
            assert turn.is_answerable is False
            assert len(turn.expected_chunk_ids) == 0, (
                f"Unanswerable turn in {s.scenario_id} must not list expected_chunk_ids"
            )
            assert len(turn.gold_claims) == 0, (
                f"Unanswerable turn in {s.scenario_id} must not specify gold_claims"
            )


def test_contradiction_cases_have_valid_pairs(indexed_dev_corpus):
    chunk_lookup = indexed_dev_corpus["chunk_lookup"]
    contra_cases = get_scenarios_by_category("contradiction")
    assert len(contra_cases) >= 1

    for s in contra_cases:
        for turn in s.turns:
            assert turn.contradiction_pair is not None, f"Contradiction scenario {s.scenario_id} missing pair"
            c1, c2 = turn.contradiction_pair
            assert c1 in chunk_lookup, f"Chunk {c1} does not exist"
            assert c2 in chunk_lookup, f"Chunk {c2} does not exist"
            assert c1 in turn.expected_chunk_ids
            assert c2 in turn.expected_chunk_ids


def test_no_leakage_with_regression_prompts():
    scenarios = get_eval_scenarios()
    normalized_reg = {p.strip().lower() for p in KNOWN_REGRESSION_PROMPTS}

    for s in scenarios:
        for turn in s.turns:
            prompt_norm = turn.user_input.strip().lower()
            assert prompt_norm not in normalized_reg, (
                f"LEAKAGE DETECTED: Scenario {s.scenario_id} uses prompt '{turn.user_input}' "
                f"which is identical to a regression test prompt!"
            )


def test_scenario_accessor_helpers():
    s = get_scenario_by_id("eval_single_01")
    assert s is not None
    assert s.scenario_id == "eval_single_01"

    none_s = get_scenario_by_id("non_existent_id")
    assert none_s is None

    singles = get_scenarios_by_category("single_intent")
    assert len(singles) == 6
