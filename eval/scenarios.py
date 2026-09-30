"""Held-out Evaluation Dataset for Stage 7 Benchmarking.

Derived strictly from the 10 corpus documents in `corpus/raw/`.
Separated from the regression test fixtures (Scenario A / Scenario B)
to prevent in-sample evaluation leakage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class GoldClaim:
    """An atomic, independently checkable factual statement directly verifiable
    against the corpus.
    """
    claim_id: str
    text: str
    supporting_chunk_ids: list[str]


@dataclass(frozen=True)
class EvalTurn:
    """A single turn within an evaluation scenario."""
    turn_number: int
    user_input: str
    timestamp_s: float
    expected_action: Literal["RETRIEVE", "WAIT", "SUPPRESS"]
    expected_doc_ids: list[str] = field(default_factory=list)
    expected_chunk_ids: list[str] = field(default_factory=list)
    expected_sub_queries: list[str] = field(default_factory=list)
    sub_query_gold_chunks: dict[str, list[str]] = field(default_factory=dict)
    gold_claims: list[GoldClaim] = field(default_factory=list)
    is_answerable: bool = True
    contradiction_pair: tuple[str, str] | None = None
    notes: str = ""


@dataclass(frozen=True)
class EvalScenario:
    """A complete evaluation scenario (single-turn or multi-turn)."""
    scenario_id: str
    category: Literal["single_intent", "multi_intent", "refinement", "unanswerable", "contradiction"]
    title: str
    turns: list[EvalTurn]
    provenance: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# Held-Out Evaluation Scenarios (13 Cases)
# ---------------------------------------------------------------------------

HELD_OUT_SCENARIOS: list[EvalScenario] = [
    # ── Category 1: Single-Intent Retrieval (6 cases) ─────────────────────────
    EvalScenario(
        scenario_id="eval_single_01",
        category="single_intent",
        title="Event Booking Submission Window & Information Requirements",
        provenance="Derived from DOC_01 §1 and §3 (booking procedures and confirmation lead times).",
        notes="Tests basic corporate event booking rules; DOC_01 was previously unexercised by test queries.",
        turns=[
            EvalTurn(
                turn_number=1,
                user_input="I need to submit an event booking request for our team offsite",
                timestamp_s=1.0,
                expected_action="RETRIEVE",
                expected_doc_ids=["DOC_01"],
                expected_chunk_ids=["DOC_01_§1", "DOC_01_§3"],
                expected_sub_queries=["I need to submit an event booking request for our team offsite"],
                sub_query_gold_chunks={
                    "I need to submit an event booking request for our team offsite": ["DOC_01_§1", "DOC_01_§3"]
                },
                gold_claims=[
                    GoldClaim(
                        claim_id="gc_s01_1",
                        text="Event booking requests must be submitted through the Events Portal at least 10 business days before the event date.",
                        supporting_chunk_ids=["DOC_01_§1"],
                    ),
                    GoldClaim(
                        claim_id="gc_s01_2",
                        text="Requests must include expected attendee count, preferred city, and event duration.",
                        supporting_chunk_ids=["DOC_01_§1"],
                    ),
                    GoldClaim(
                        claim_id="gc_s01_3",
                        text="Venue confirmation typically takes 3 to 5 business days after submission.",
                        supporting_chunk_ids=["DOC_01_§3"],
                    ),
                ],
                is_answerable=True,
                notes="Requires retrieval of booking timeline from §1 and lead time from §3.",
            )
        ],
    ),
    EvalScenario(
        scenario_id="eval_single_02",
        category="single_intent",
        title="Bangalore Small Workshop Venue Query",
        provenance="Derived from DOC_03 §2 (MG Road conference rooms in Bangalore).",
        notes="Tests city-scoped venue retrieval for Bangalore; DOC_03 was only used as a distractor previously.",
        turns=[
            EvalTurn(
                turn_number=1,
                user_input="I need a conference room in Bangalore for a small workshop of 20 people",
                timestamp_s=1.0,
                expected_action="RETRIEVE",
                expected_doc_ids=["DOC_03"],
                expected_chunk_ids=["DOC_03_§2"],
                expected_sub_queries=["I need a conference room in Bangalore for a small workshop of 20 people"],
                sub_query_gold_chunks={
                    "I need a conference room in Bangalore for a small workshop of 20 people": ["DOC_03_§2"]
                },
                gold_claims=[
                    GoldClaim(
                        claim_id="gc_s02_1",
                        text="Venue E at MG Road has a seated capacity of 25 suitable for small team workshops.",
                        supporting_chunk_ids=["DOC_03_§2"],
                    ),
                    GoldClaim(
                        claim_id="gc_s02_2",
                        text="Venue E has no on-site catering services and external vendors must be arranged independently.",
                        supporting_chunk_ids=["DOC_03_§2"],
                    ),
                ],
                is_answerable=True,
                notes="Precision test: must retrieve Venue E (Bangalore) and not Pune venues from DOC_02.",
            )
        ],
    ),
    EvalScenario(
        scenario_id="eval_single_03",
        category="single_intent",
        title="Emergency Business Travel Notice Exemption",
        provenance="Derived from DOC_08 §1 and §3 (standard booking window and emergency provisions).",
        notes="Tests emergency travel exception rules; tests previously only exercised §2 late bookings.",
        turns=[
            EvalTurn(
                turn_number=1,
                user_input="I need to arrange emergency business travel departing in less than 24 hours",
                timestamp_s=1.0,
                expected_action="RETRIEVE",
                expected_doc_ids=["DOC_08"],
                expected_chunk_ids=["DOC_08_§1", "DOC_08_§2", "DOC_08_§3"],
                expected_sub_queries=["I need to arrange emergency business travel departing in less than 24 hours"],
                sub_query_gold_chunks={
                    "I need to arrange emergency business travel departing in less than 24 hours": [
                        "DOC_08_§1",
                        "DOC_08_§2",
                        "DOC_08_§3",
                    ]
                },
                gold_claims=[
                    GoldClaim(
                        claim_id="gc_s03_1",
                        text="Standard travel bookings must be made at least 7 days in advance through the approved travel portal.",
                        supporting_chunk_ids=["DOC_08_§1"],
                    ),
                    GoldClaim(
                        claim_id="gc_s03_2",
                        text="Emergency travel arranged with less than 24 hours notice is exempt from the standard 7-day booking window.",
                        supporting_chunk_ids=["DOC_08_§3"],
                    ),
                    GoldClaim(
                        claim_id="gc_s03_3",
                        text="Emergency travel requires post-trip written justification and approval from manager and Travel Exceptions desk.",
                        supporting_chunk_ids=["DOC_08_§2", "DOC_08_§3"],
                    ),
                ],
                is_answerable=True,
                notes="Connects §3 exemption back to §2 justification requirement.",
            )
        ],
    ),
    EvalScenario(
        scenario_id="eval_single_04",
        category="single_intent",
        title="Force Majeure Event Cancellation Policy",
        provenance="Derived from DOC_04 §3 (Force majeure cancellations).",
        notes="Tests cancellation refund rules specifically under documented force majeure events.",
        turns=[
            EvalTurn(
                turn_number=1,
                user_input="I need to know the refund policy for event cancellations caused by natural disasters",
                timestamp_s=1.0,
                expected_action="RETRIEVE",
                expected_doc_ids=["DOC_04"],
                expected_chunk_ids=["DOC_04_§3"],
                expected_sub_queries=["I need to know the refund policy for event cancellations caused by natural disasters"],
                sub_query_gold_chunks={
                    "I need to know the refund policy for event cancellations caused by natural disasters": ["DOC_04_§3"]
                },
                gold_claims=[
                    GoldClaim(
                        claim_id="gc_s04_1",
                        text="Cancellations resulting from documented force majeure events like natural disasters or government closures are exempt from standard cancellation windows.",
                        supporting_chunk_ids=["DOC_04_§3"],
                    ),
                    GoldClaim(
                        claim_id="gc_s04_2",
                        text="Force majeure cancellations are eligible for a full refund regardless of the notice period.",
                        supporting_chunk_ids=["DOC_04_§3"],
                    ),
                ],
                is_answerable=True,
                notes="Checks §3 specific clause rather than the general 30-day window in §1.",
            )
        ],
    ),
    EvalScenario(
        scenario_id="eval_single_05",
        category="single_intent",
        title="International Expense Currency Conversion Date Rule",
        provenance="Derived from DOC_07 §3 (Currency conversion rules).",
        notes="Tests currency conversion date rules for international travel; tests previously only verified receipts/approvals.",
        turns=[
            EvalTurn(
                turn_number=1,
                user_input="I need to find the currency conversion exchange rate rules for international expense reports",
                timestamp_s=1.0,
                expected_action="RETRIEVE",
                expected_doc_ids=["DOC_07"],
                expected_chunk_ids=["DOC_07_§3"],
                expected_sub_queries=["I need to find the currency conversion exchange rate rules for international expense reports"],
                sub_query_gold_chunks={
                    "I need to find the currency conversion exchange rate rules for international expense reports": ["DOC_07_§3"]
                },
                gold_claims=[
                    GoldClaim(
                        claim_id="gc_s05_1",
                        text="Currency conversion uses the exchange rate published by Finance on the date of the transaction.",
                        supporting_chunk_ids=["DOC_07_§3"],
                    ),
                    GoldClaim(
                        claim_id="gc_s05_2",
                        text="The conversion exchange rate is not based on the date of reimbursement submission.",
                        supporting_chunk_ids=["DOC_07_§3"],
                    ),
                ],
                is_answerable=True,
                notes="Tests precise knowledge of transaction date vs submission date.",
            )
        ],
    ),
    EvalScenario(
        scenario_id="eval_single_06",
        category="single_intent",
        title="Domestic Reimbursement Payout Timeline",
        provenance="Derived from DOC_06 §3 (Reimbursement disbursement timeline).",
        notes="Tests payment disbursement timeline for domestic expense claims.",
        turns=[
            EvalTurn(
                turn_number=1,
                user_input="I need to check when approved domestic travel expense reimbursements get paid out",
                timestamp_s=1.0,
                expected_action="RETRIEVE",
                expected_doc_ids=["DOC_06"],
                expected_chunk_ids=["DOC_06_§3"],
                expected_sub_queries=["I need to check when approved domestic travel expense reimbursements get paid out"],
                sub_query_gold_chunks={
                    "I need to check when approved domestic travel expense reimbursements get paid out": ["DOC_06_§3"]
                },
                gold_claims=[
                    GoldClaim(
                        claim_id="gc_s06_1",
                        text="Approved domestic travel reimbursements are disbursed within 7 business days of final approval.",
                        supporting_chunk_ids=["DOC_06_§3"],
                    )
                ],
                is_answerable=True,
                notes="Targets §3 disbursement timeframe.",
            )
        ],
    ),

    # ── Category 2: Multi-Intent Retrieval (2 cases) ──────────────────────────
    EvalScenario(
        scenario_id="eval_multi_01",
        category="multi_intent",
        title="Event Attendee Approval Tiers + Procurement Vendor Payment Terms",
        provenance="Derived from DOC_01 §2 (Capacity & Approval Tiers) and DOC_10 §1 (Vendor Payment Terms).",
        notes="Compound request combining event management approval thresholds with procurement invoice terms.",
        turns=[
            EvalTurn(
                turn_number=1,
                user_input="I need the approval tiers for large company events and also need the standard vendor payment terms",
                timestamp_s=1.0,
                expected_action="RETRIEVE",
                expected_doc_ids=["DOC_01", "DOC_10"],
                expected_chunk_ids=["DOC_01_§2", "DOC_10_§1"],
                expected_sub_queries=[
                    "I need the approval tiers for large company events",
                    "the standard vendor payment terms",
                ],
                sub_query_gold_chunks={
                    "I need the approval tiers for large company events": ["DOC_01_§2"],
                    "the standard vendor payment terms": ["DOC_10_§1"],
                },
                gold_claims=[
                    GoldClaim(
                        claim_id="gc_m01_1",
                        text="Events up to 50 attendees require only direct manager approval.",
                        supporting_chunk_ids=["DOC_01_§2"],
                    ),
                    GoldClaim(
                        claim_id="gc_m01_2",
                        text="Events exceeding 50 attendees require Facilities Coordinator sign-off, while events over 150 require Regional Operations approval.",
                        supporting_chunk_ids=["DOC_01_§2"],
                    ),
                    GoldClaim(
                        claim_id="gc_m01_3",
                        text="Standard vendor payment terms are net-30 from invoice date unless otherwise negotiated.",
                        supporting_chunk_ids=["DOC_10_§1"],
                    ),
                ],
                is_answerable=True,
                notes="Evaluates decomposition into event approval (DOC_01) and payment terms (DOC_10).",
            )
        ],
    ),
    EvalScenario(
        scenario_id="eval_multi_02",
        category="multi_intent",
        title="Bangalore Venue Booking + Standard Catering Menu Packages",
        provenance="Derived from DOC_03 §1 (Whitefield venue) and DOC_05 §1 (Catering packages).",
        notes="Compound request across Bangalore venue directory and standard catering offerings.",
        turns=[
            EvalTurn(
                turn_number=1,
                user_input="I need to find a venue in Bangalore for 50 people and also need the standard catering packages",
                timestamp_s=1.0,
                expected_action="RETRIEVE",
                expected_doc_ids=["DOC_03", "DOC_05"],
                expected_chunk_ids=["DOC_03_§1", "DOC_05_§1"],
                expected_sub_queries=[
                    "I need to find a venue in Bangalore for 50 people",
                    "the standard catering packages",
                ],
                sub_query_gold_chunks={
                    "I need to find a venue in Bangalore for 50 people": ["DOC_03_§1"],
                    "the standard catering packages": ["DOC_05_§1"],
                },
                gold_claims=[
                    GoldClaim(
                        claim_id="gc_m02_1",
                        text="Venue D at Whitefield Business Hub in Bangalore has a seated capacity of 50 with standard AV included.",
                        supporting_chunk_ids=["DOC_03_§1"],
                    ),
                    GoldClaim(
                        claim_id="gc_m02_2",
                        text="Three standard catering packages are available: Package 1 breakfast/coffee, Package 2 working lunch, and Package 3 full-day catering.",
                        supporting_chunk_ids=["DOC_05_§1"],
                    ),
                ],
                is_answerable=True,
                notes="Distinct from Scenario A which targeted Pune (DOC_02) and cancellation (DOC_04).",
            )
        ],
    ),

    # ── Category 3: Multi-Turn Refinement (2 cases) ───────────────────────────
    EvalScenario(
        scenario_id="eval_refine_01",
        category="refinement",
        title="Domestic Expense Approval Chain Elevated by Amount Threshold",
        provenance="Derived from DOC_09 §1 (Standard Approval Chain) and DOC_09 §2 (Elevated Approval Thresholds).",
        notes="Evaluates evolutionary refinement when cost exceeds the ₹25,000 elevation threshold.",
        turns=[
            EvalTurn(
                turn_number=1,
                user_input="I need the approval requirements for standard domestic travel expenses",
                timestamp_s=1.0,
                expected_action="RETRIEVE",
                expected_doc_ids=["DOC_09"],
                expected_chunk_ids=["DOC_09_§1"],
                expected_sub_queries=["I need the approval requirements for standard domestic travel expenses"],
                sub_query_gold_chunks={
                    "I need the approval requirements for standard domestic travel expenses": ["DOC_09_§1"]
                },
                gold_claims=[
                    GoldClaim(
                        claim_id="gc_r01_1",
                        text="Expense reports under ₹25,000 require only direct manager approval.",
                        supporting_chunk_ids=["DOC_09_§1"],
                    )
                ],
                is_answerable=True,
                notes="Initial turn: standard manager sign-off.",
            ),
            EvalTurn(
                turn_number=2,
                user_input="the total expense claim exceeds 25000 rupees",
                timestamp_s=2.0,
                expected_action="RETRIEVE",
                expected_doc_ids=["DOC_09"],
                expected_chunk_ids=["DOC_09_§2"],
                expected_sub_queries=["the total expense claim exceeds 25000 rupees"],
                sub_query_gold_chunks={
                    "the total expense claim exceeds 25000 rupees": ["DOC_09_§2"]
                },
                gold_claims=[
                    GoldClaim(
                        claim_id="gc_r01_2",
                        text="Expense reports exceeding ₹25,000 require Senior Director approval in addition to manager sign-off.",
                        supporting_chunk_ids=["DOC_09_§2"],
                    )
                ],
                is_answerable=True,
                notes="Refinement turn: adds elevated approval requirement without session restart.",
            ),
        ],
    ),
    EvalScenario(
        scenario_id="eval_refine_02",
        category="refinement",
        title="Standard Cancellation Window Refined by Force Majeure Exception",
        provenance="Derived from DOC_04 §1 (Standard Windows) and DOC_04 §3 (Force Majeure Exceptions).",
        notes="Evaluates ledger versioning when standard cancellation terms are supplemented by a natural disaster.",
        turns=[
            EvalTurn(
                turn_number=1,
                user_input="I need the standard cancellation windows for booked corporate events",
                timestamp_s=1.0,
                expected_action="RETRIEVE",
                expected_doc_ids=["DOC_04"],
                expected_chunk_ids=["DOC_04_§1"],
                expected_sub_queries=["I need the standard cancellation windows for booked corporate events"],
                sub_query_gold_chunks={
                    "I need the standard cancellation windows for booked corporate events": ["DOC_04_§1"]
                },
                gold_claims=[
                    GoldClaim(
                        claim_id="gc_r02_1",
                        text="Bookings cancelled more than 30 days before event receive a full refund, 15 to 30 days receive 50%, and under 15 days are non-refundable.",
                        supporting_chunk_ids=["DOC_04_§1"],
                    )
                ],
                is_answerable=True,
                notes="Initial turn: standard refund tiers.",
            ),
            EvalTurn(
                turn_number=2,
                user_input="the cancellation was due to a government mandated closure",
                timestamp_s=2.0,
                expected_action="RETRIEVE",
                expected_doc_ids=["DOC_04"],
                expected_chunk_ids=["DOC_04_§3"],
                expected_sub_queries=["the cancellation was due to a government mandated closure"],
                sub_query_gold_chunks={
                    "the cancellation was due to a government mandated closure": ["DOC_04_§3"]
                },
                gold_claims=[
                    GoldClaim(
                        claim_id="gc_r02_2",
                        text="Cancellations resulting from documented force majeure events like government-mandated closures are exempt from standard windows and eligible for full refund.",
                        supporting_chunk_ids=["DOC_04_§3"],
                    )
                ],
                is_answerable=True,
                notes="Refinement turn: adds force majeure exemption while keeping standard policy context.",
            ),
        ],
    ),

    # ── Category 4: Unanswerable / Coverage Gap (2 cases) ─────────────────────
    EvalScenario(
        scenario_id="eval_unans_01",
        category="unanswerable",
        title="Deliberate Coverage Gap: Venue A Dietary Accommodation Policy",
        provenance="Explicitly declared as undocumented in DOC_05 §3.",
        notes="System must recognize that Venue A dietary/vendor policy is unavailable and refrain from hallucinating.",
        turns=[
            EvalTurn(
                turn_number=1,
                user_input="I need to verify the vegan dietary accommodation policy specifically for Venue A in Pune",
                timestamp_s=1.0,
                expected_action="RETRIEVE",
                expected_doc_ids=[],
                expected_chunk_ids=[],
                expected_sub_queries=["I need to verify the vegan dietary accommodation policy specifically for Venue A in Pune"],
                sub_query_gold_chunks={
                    "I need to verify the vegan dietary accommodation policy specifically for Venue A in Pune": []
                },
                gold_claims=[],
                is_answerable=False,
                notes="DOC_05 §3 states Venue A dietary documentation is not available in this corpus version. A correct answer flags this coverage gap.",
            )
        ],
    ),
    EvalScenario(
        scenario_id="eval_unans_02",
        category="unanswerable",
        title="Uncovered Policy: Personal Vehicle Mileage Reimbursement Rate",
        provenance="Out-of-domain query against DOC_06 (Domestic Travel Reimbursement).",
        notes="Corpus covers lodging, airfare, meals, and receipts, but never mentions mileage or per-km rates.",
        turns=[
            EvalTurn(
                turn_number=1,
                user_input="I need to know the per kilometer mileage reimbursement rate for driving a personal car on company business",
                timestamp_s=1.0,
                expected_action="RETRIEVE",
                expected_doc_ids=[],
                expected_chunk_ids=[],
                expected_sub_queries=["I need to know the per kilometer mileage reimbursement rate for driving a personal car on company business"],
                sub_query_gold_chunks={
                    "I need to know the per kilometer mileage reimbursement rate for driving a personal car on company business": []
                },
                gold_claims=[],
                is_answerable=False,
                notes="Grounding failure if the pipeline invents a numerical mileage rate (e.g. ₹10/km).",
            )
        ],
    ),

    # ── Category 5: Contradiction-Sensitive Case (1 case) ─────────────────────
    EvalScenario(
        scenario_id="eval_contra_01",
        category="contradiction",
        title="Internal 30-Day Cancellation Window vs Directly Contracted Vendor 14-Day Terms",
        provenance="Derived from DOC_04 §1 (30-day window) and DOC_10 §2 (14-day vendor cancellation clause).",
        notes="Tests whether the system surfaces the scope difference between internal event policy and external vendor terms.",
        turns=[
            EvalTurn(
                turn_number=1,
                user_input="I need to check the cancellation notice window required for directly contracted catering vendors",
                timestamp_s=1.0,
                expected_action="RETRIEVE",
                expected_doc_ids=["DOC_04", "DOC_10"],
                expected_chunk_ids=["DOC_04_§1", "DOC_10_§2"],
                expected_sub_queries=["I need to check the cancellation notice window required for directly contracted catering vendors"],
                sub_query_gold_chunks={
                    "I need to check the cancellation notice window required for directly contracted catering vendors": [
                        "DOC_04_§1",
                        "DOC_10_§2",
                    ]
                },
                gold_claims=[
                    GoldClaim(
                        claim_id="gc_c01_1",
                        text="Internal corporate policy specifies a 30-day notice window for full event deposit refund under DOC_04.",
                        supporting_chunk_ids=["DOC_04_§1"],
                    ),
                    GoldClaim(
                        claim_id="gc_c01_2",
                        text="Directly contracted event vendors like catering may enforce a 14-day cancellation notice period for full refund eligibility under DOC_10.",
                        supporting_chunk_ids=["DOC_10_§2"],
                    ),
                    GoldClaim(
                        claim_id="gc_c01_3",
                        text="Employees must confirm vendor-specific terms at the time of contracting due to the discrepancy between internal and vendor terms.",
                        supporting_chunk_ids=["DOC_10_§2"],
                    ),
                ],
                is_answerable=True,
                contradiction_pair=("DOC_04_§1", "DOC_10_§2"),
                notes="System must surface both windows without silently suppressing one or picking a side.",
            )
        ],
    ),
]


# ---------------------------------------------------------------------------
# Accessor Functions
# ---------------------------------------------------------------------------

def get_eval_scenarios() -> list[EvalScenario]:
    """Return all held-out evaluation scenarios."""
    return list(HELD_OUT_SCENARIOS)


def get_scenario_by_id(scenario_id: str) -> EvalScenario | None:
    """Retrieve an evaluation scenario by its unique identifier."""
    for s in HELD_OUT_SCENARIOS:
        if s.scenario_id == scenario_id:
            return s
    return None


def get_scenarios_by_category(category: str) -> list[EvalScenario]:
    """Retrieve evaluation scenarios filtered by category."""
    return [s for s in HELD_OUT_SCENARIOS if s.category == category]


def all_gold_chunk_ids() -> set[str]:
    """Return the set of all chunk IDs referenced across all evaluation scenarios."""
    chunks: set[str] = set()
    for s in HELD_OUT_SCENARIOS:
        for t in s.turns:
            chunks.update(t.expected_chunk_ids)
            for gc in t.gold_claims:
                chunks.update(gc.supporting_chunk_ids)
    return chunks


def all_gold_doc_ids() -> set[str]:
    """Return the set of all doc IDs referenced across all evaluation scenarios."""
    docs: set[str] = set()
    for s in HELD_OUT_SCENARIOS:
        for t in s.turns:
            docs.update(t.expected_doc_ids)
    return docs
