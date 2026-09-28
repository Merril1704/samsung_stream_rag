import os
from pathlib import Path
import pytest
from grounding.verifier import GroundingVerifier
from retrieval.types import EvidenceChunk
from controller.llm_factory import get_llm_client

@pytest.fixture
def verifier():
    return GroundingVerifier(llm_client=get_llm_client())


def test_correctly_grounded_answer_passes(verifier):
    chunk = EvidenceChunk(
        chunk_id="DOC_04_§1", doc_id="DOC_04", section="§1",
        text="Bookings cancelled more than 30 days before the event date receive a full refund of any deposit paid.",
        retrieval_score=0.9,
    )
    result = verifier.verify(
        answer_text="Cancellations made more than 30 days before the event receive a full refund.",
        cited_chunks_by_claim={"Cancellations made more than 30 days before the event receive a full refund.": "DOC_04_§1"},
        evidence=[chunk],
    )
    assert result.all_verified is True


def test_fabricated_citation_caught(verifier):
    chunk = EvidenceChunk(
        chunk_id="DOC_04_§1", doc_id="DOC_04", section="§1",
        text="Bookings cancelled more than 30 days before the event date receive a full refund of any deposit paid.",
        retrieval_score=0.9,
    )
    result = verifier.verify(
        answer_text="Cancellations made more than 30 days before the event receive a full refund.",
        cited_chunks_by_claim={"Cancellations made more than 30 days before the event receive a full refund.": "DOC_99_§1"},  # never actually retrieved
        evidence=[chunk],
    )
    assert result.all_verified is False
    assert len(result.fabricated_citations) == 1


def test_unsupported_claim_caught(verifier):
    chunk = EvidenceChunk(
        chunk_id="DOC_02_§1", doc_id="DOC_02", section="§1",
        text="Venue A — Riverside Business Center. On-site catering limited to the venue's preferred vendor list.",
        retrieval_score=0.9,
    )
    result = verifier.verify(
        answer_text="Venue A allows external catering with no restrictions.",
        cited_chunks_by_claim={"Venue A allows external catering with no restrictions.": "DOC_02_§1"},
        evidence=[chunk],
    )
    assert result.all_verified is False
    assert len(result.unsupported_claims) == 1


def test_compound_claim_gets_decomposed_before_checking(verifier):
    """
    This is the direct regression test for the Case 6 failure mode found
    during isolated Step 3 validation: a compound claim bundling a true
    fact and a false/exaggerated fact must be split by the decomposer,
    and the false part must be caught individually.
    """
    chunk = EvidenceChunk(
        chunk_id="DOC_02_§1", doc_id="DOC_02", section="§1",
        text="Capacity: 40 seated / 60 standing. Includes projector, whiteboard, and standard AV. Booking lead time: 5 business days.",
        retrieval_score=0.9,
    )
    answer = "The venue is available for booking with 5 business days notice and includes full AV support."
    claims = verifier.decomposer.decompose(answer)
    assert len(claims) >= 2, f"Expected compound claim to be split, got: {claims}"

    cited = {c: "DOC_02_§1" for c in claims}
    result = verifier.verify(answer_text=answer, cited_chunks_by_claim=cited, evidence=[chunk])
    assert result.all_verified is False
    assert len(result.unsupported_claims) >= 1


def test_cherry_pick_violation_on_contradiction(verifier):
    """
    DOC_04/DOC_10 contradiction from Stage 3. If the answer cites only
    DOC_04's 30-day claim and doesn't also surface DOC_10's conflicting
    14-day claim, that's a cherry-pick violation.
    """
    chunk_04 = EvidenceChunk(
        chunk_id="DOC_04_§1", doc_id="DOC_04", section="§1",
        text="Bookings cancelled more than 30 days before the event date receive a full refund of any deposit paid.",
        retrieval_score=0.9, contradiction_flag=True, contradicts_chunk_id="DOC_10_§2",
    )
    chunk_10 = EvidenceChunk(
        chunk_id="DOC_10_§2", doc_id="DOC_10", section="§2",
        text="Event vendors contracted directly may enforce a 14-day cancellation notice period for full refund eligibility, which may differ from the internal Event Cancellation & Refund Policy's 30-day window.",
        retrieval_score=0.85, contradiction_flag=True, contradicts_chunk_id="DOC_04_§1",
    )
    answer = "Cancellations made more than 30 days before the event receive a full refund."
    result = verifier.verify(
        answer_text=answer,
        cited_chunks_by_claim={answer: "DOC_04_§1"},
        evidence=[chunk_04, chunk_10],   # both were retrieved, but only DOC_04 was cited
    )
    assert result.all_verified is False
    assert len(result.cherry_picks) == 1


def test_no_cherry_pick_when_both_sides_surfaced(verifier):
    """Same setup, but the answer correctly surfaces both conflicting claims -- must pass."""
    chunk_04 = EvidenceChunk(
        chunk_id="DOC_04_§1", doc_id="DOC_04", section="§1",
        text="Bookings cancelled more than 30 days before the event date receive a full refund of any deposit paid.",
        retrieval_score=0.9, contradiction_flag=True, contradicts_chunk_id="DOC_10_§2",
    )
    chunk_10 = EvidenceChunk(
        chunk_id="DOC_10_§2", doc_id="DOC_10", section="§2",
        text="Event vendors contracted directly may enforce a 14-day cancellation notice period for full refund eligibility, which may differ from the internal Event Cancellation & Refund Policy's 30-day window.",
        retrieval_score=0.85, contradiction_flag=True, contradicts_chunk_id="DOC_04_§1",
    )
    claim_a = "Internal policy grants a full refund for cancellations more than 30 days before the event."
    claim_b = "Event vendors may separately enforce a 14-day cancellation notice period for full refund eligibility."
    result = verifier.verify(
        answer_text=f"{claim_a} {claim_b}",
        cited_chunks_by_claim={claim_a: "DOC_04_§1", claim_b: "DOC_10_§2"},
        evidence=[chunk_04, chunk_10],
    )
    assert result.all_verified is True


def test_fabrication_never_defaults_to_supported_on_parse_failure():
    """
    Sanity check on the fail-safe: if entailment check parsing fails for
    any reason, it must never silently default to supported=True.
    """
    class BrokenLLMClient:
        def complete(self, system_prompt: str, user_prompt: str) -> str:
            return "NON_JSON_CORRUPTED_RESPONSE_12345"

    verifier = GroundingVerifier(llm_client=BrokenLLMClient())
    chunk = EvidenceChunk(
        chunk_id="DOC_04_§1", doc_id="DOC_04", section="§1",
        text="Bookings cancelled more than 30 days before the event date receive a full refund of any deposit paid.",
        retrieval_score=0.9,
    )
    answer = "Cancellations made more than 30 days before the event receive a full refund."
    result = verifier.verify(
        answer_text=answer,
        cited_chunks_by_claim={answer: "DOC_04_§1"},
        evidence=[chunk],
    )
    assert result.all_verified is False
    assert len(result.unsupported_claims) >= 1
    assert result.claims[0].supported is False
    assert "parse failure" in result.claims[0].reason.lower()


def test_client_exception_never_defaults_to_supported():
    """
    Sanity check on error handling: if complete() raises an exception,
    it must never silently default to supported=True.
    """
    class TimeoutLLMClient:
        def complete(self, system_prompt: str, user_prompt: str, *, max_tokens: int | None = None) -> str:
            raise RuntimeError("timeout")

    verifier = GroundingVerifier(llm_client=TimeoutLLMClient())
    chunk = EvidenceChunk(
        chunk_id="DOC_04_§1", doc_id="DOC_04", section="§1",
        text="Bookings cancelled more than 30 days before the event date receive a full refund of any deposit paid.",
        retrieval_score=0.9,
    )
    answer = "Cancellations made more than 30 days before the event receive a full refund."
    result = verifier.verify(
        answer_text=answer,
        cited_chunks_by_claim={answer: "DOC_04_§1"},
        evidence=[chunk],
    )
    assert result.all_verified is False
    assert len(result.unsupported_claims) >= 1
    assert result.claims[0].supported is False


def test_wrong_schema_json_never_defaults_to_supported():
    """
    Sanity check on schema validation: if complete() returns JSON missing
    the 'supported' field, it must never silently default to supported=True.
    """
    class WrongSchemaLLMClient:
        def complete(self, system_prompt: str, user_prompt: str, *, max_tokens: int | None = None) -> str:
            return '{"foo": 1}'

    verifier = GroundingVerifier(llm_client=WrongSchemaLLMClient())
    chunk = EvidenceChunk(
        chunk_id="DOC_04_§1", doc_id="DOC_04", section="§1",
        text="Bookings cancelled more than 30 days before the event date receive a full refund of any deposit paid.",
        retrieval_score=0.9,
    )
    answer = "Cancellations made more than 30 days before the event receive a full refund."
    result = verifier.verify(
        answer_text=answer,
        cited_chunks_by_claim={answer: "DOC_04_§1"},
        evidence=[chunk],
    )
    assert result.all_verified is False
    assert len(result.unsupported_claims) >= 1
    assert result.claims[0].supported is False

