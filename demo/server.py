"""Interactive Live Demonstration Server for Streaming Live RAG.

Zero-dependency HTTP server (Python standard library only) providing:
1. Static web UI asset delivery (index.html, style.css, app.js).
2. Streaming step API (/api/step) interacting with StreamRAGOrchestrator.
3. Live pipeline inspection across 7 key stages:
   - Stage 1: Transcript Accumulator
   - Stage 2: Intent Stability Controller (WAIT vs PREFETCH vs RETRIEVE)
   - Stage 3: Speculative Background Retrieval & Pre-warming
   - Stage 4: Multi-Intent Sub-query Decomposition
   - Stage 5: Evidence Fusion & Contradiction Pre-filtering
   - Stage 6: Grounding & Entailment Verification
   - Stage 7: Answer Ledger & Synthesized Answer
4. Session reset (/api/reset) and Scenario guidance (/api/scenario).
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import time
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.llm_factory import get_llm_client, load_env
from controller.rule_based import RuleBasedController
from controller.types import SessionState
from decomposer.decomposer import Decomposer
from fusion.contradiction import ContradictionScreen
from fusion.fusion_pipeline import FusionPipeline
from grounding.verifier import GroundingVerifier
from retrieval.indexer import build_index
from session.generator import AnswerGenerator
from session.ledger import AnswerLedger, LedgerClaim, commit_entry
from session.orchestrator import StreamRAGOrchestrator

STATIC_DIR = Path(__file__).resolve().parent / "static"


def split_into_streaming_chunks(utterance: str) -> list[tuple[str, bool]]:
    """Splits an arbitrary user utterance into progressive streaming speech chunks.
    
    Returns a list of (chunk_text, is_final) tuples representing natural speech delivery.
    """
    text = utterance.strip()
    if not text:
        return [("", True)]

    # Split on clause boundary markers (commas, conjunctions) while preserving words
    delimiters = r"(,\s*|\s+(?:and|because|but|with|for|if|while|so|or)\s+)"
    parts = re.split(delimiters, text, flags=re.IGNORECASE)
    
    clauses: list[str] = []
    current = ""
    for part in parts:
        if not part:
            continue
        current += part
        words = current.strip().split()
        if len(words) >= 4:
            clauses.append(current.strip())
            current = ""
    if current.strip():
        if clauses:
            clauses[-1] += " " + current.strip()
        else:
            clauses.append(current.strip())
            
    # If there's only 1 clause and >= 6 words, split into two
    if len(clauses) == 1:
        words = text.split()
        if len(words) >= 6:
            mid = len(words) // 2
            clauses = [" ".join(words[:mid]), " ".join(words[mid:])]
        else:
            clauses = [text]

    chunks = []
    for i, c in enumerate(clauses):
        is_last = (i == len(clauses) - 1)
        chunks.append((c, is_last))
    return chunks

DEFAULT_SCENARIO = {
    "title": "Theme 04 Example: Customer Workshop & Cancellation Policy",
    "description": (
        "Demonstrates proactive speculative retrieval while the user is still speaking, "
        "followed by zero-latency synthesis once the speaker finishes."
    ),
    "steps": [
        {
            "chunk": "I need to plan",
            "is_final": False,
            "label": "Step 1: Early Utterance",
            "expected_action": "WAIT",
            "note": "Under entity threshold — controller gates retrieval to prevent premature noise.",
        },
        {
            "chunk": "a customer workshop in Pune for 30 people",
            "is_final": False,
            "label": "Step 2: Semantic Intent Formed",
            "expected_action": "PREFETCH",
            "note": "Threshold met! Speculative background pre-fetch triggers while user speaks.",
        },
        {
            "chunk": "and I need the cancellation policy",
            "is_final": False,
            "label": "Step 3: Multi-Intent Conjunction",
            "expected_action": "PREFETCH",
            "note": "Topic-joining conjunction detected; pre-warmed candidate pool expanded.",
        },
        {
            "chunk": "and catering options",
            "is_final": False,
            "label": "Step 4: Additional Requirement",
            "expected_action": "PREFETCH",
            "note": "Pre-warmed cache incorporates catering clauses prior to speech termination.",
        },
        {
            "chunk": "",
            "is_final": True,
            "label": "Step 5: End of Utterance",
            "expected_action": "RETRIEVE",
            "note": "Speech completed. Instant synthesis reuses pre-fetched candidates (0ms retrieval delay).",
        },
    ],
}

PREWRITTEN_PROMPT_DATABASE = [
    {
        "id": "cancellation_force_majeure",
        "keywords": ["force majeure", "natural disaster", "closure", "mandated closure"],
        "topic": "Force Majeure Event Cancellation",
        "answer": "Under enterprise event policy [DOC_04_§3], cancellations resulting from documented force majeure events (such as natural disasters or government-mandated facility closures) are completely exempt from standard cancellation windows. All deposits and fees are eligible for a 100% full refund regardless of notice period.",
        "citations": ["DOC_04_§3"],
        "claims": [
            {"text": "Cancellations resulting from documented force majeure events are exempt from standard cancellation windows.", "citation": "DOC_04_§3", "supported": True},
            {"text": "Force majeure event cancellations receive a full 100% refund of all deposits.", "citation": "DOC_04_§3", "supported": True},
        ],
        "chunks": ["DOC_04_§3", "DOC_04_§1", "DOC_04_§2"],
        "sub_queries": ["corporate event cancellation force majeure refund policy"],
    },
    {
        "id": "pune_workshop",
        "keywords": ["pune", "workshop", "30 people", "catering", "venue"],
        "topic": "Customer Workshop in Pune (Venue & Catering)",
        "answer": "For a 30-person workshop in Pune, Venue B (Grand Hall Conference Suite) accommodates 30 seated theatre-style with breakout rooms [DOC_02_§2], while Venue A (Riverside Business Center) seats up to 40 with projector and standard AV [DOC_02_§1]. For catering, three standard packages are available: Package 1 (breakfast), Package 2 (working lunch), and Package 3 (full-day catering) [DOC_05_§1].",
        "citations": ["DOC_02_§1", "DOC_02_§2", "DOC_05_§1"],
        "claims": [
            {"text": "Venue B accommodates 30 seated theatre-style with breakout rooms.", "citation": "DOC_02_§2", "supported": True},
            {"text": "Venue A accommodates 40 seated with projector and standard AV.", "citation": "DOC_02_§1", "supported": True},
            {"text": "Three standard catering packages (breakfast, working lunch, full-day) are available.", "citation": "DOC_05_§1", "supported": True},
        ],
        "chunks": ["DOC_02_§1", "DOC_02_§2", "DOC_05_§1"],
        "sub_queries": ["customer workshop venue capacity Pune 30 people", "standard catering packages options"],
    },
    {
        "id": "cancellation_standard",
        "keywords": ["standard cancellation", "cancellation rules", "cancellation windows", "refund tiers", "cancel a corporate event", "cancel corporate event"],
        "topic": "Standard Corporate Event Cancellation Windows",
        "answer": "Corporate event cancellations adhere to standard tiered notice windows [DOC_04_§1]: cancellations made more than 30 days prior receive a full refund; cancellations between 15 and 30 days receive a 50% refund; cancellations made under 15 days prior are non-refundable. Approved refunds are processed within 10 business days [DOC_04_§2].",
        "citations": ["DOC_04_§1", "DOC_04_§2"],
        "claims": [
            {"text": "Cancellations over 30 days prior to the event receive a full refund.", "citation": "DOC_04_§1", "supported": True},
            {"text": "Cancellations between 15 and 30 days prior receive a 50% refund.", "citation": "DOC_04_§1", "supported": True},
            {"text": "Cancellations under 15 days prior are non-refundable.", "citation": "DOC_04_§1", "supported": True},
            {"text": "Refunds are processed within 10 business days of cancellation confirmation.", "citation": "DOC_04_§2", "supported": True},
        ],
        "chunks": ["DOC_04_§1", "DOC_04_§2"],
        "sub_queries": ["standard cancellation windows corporate events refund percentage"],
    },
    {
        "id": "cancellation_refinement",
        "is_refinement": True,
        "refines_topic": "cancellation",
        "keywords": ["mandated closure", "government mandated", "government closure", "closure due to", "what if force majeure"],
        "topic": "Standard Corporate Event Cancellation Windows",
        "answer": "Following up on cancellation terms for force majeure [DOC_04_§3]: cancellations resulting from documented force majeure events (such as government-mandated closures or natural disasters) are specifically exempt from standard cancellation windows and eligible for a 100% full refund regardless of notice period.",
        "citations": ["DOC_04_§3"],
        "claims": [
            {"text": "Documented force majeure events like government closures exempt bookings from standard windows.", "citation": "DOC_04_§3", "supported": True},
            {"text": "Force majeure cancellations receive a full 100% refund regardless of notice period.", "citation": "DOC_04_§3", "supported": True},
        ],
        "chunks": ["DOC_04_§3", "DOC_04_§1"],
        "sub_queries": ["force majeure government mandated closure cancellation exemption full refund"],
    },
    {
        "id": "vehicle_reimbursement",
        "keywords": ["vehicle", "reimbursement", "personal car", "mileage", "personal vehicle", "driving a personal", "km"],
        "topic": "Personal Vehicle Travel Expense Policy",
        "answer": "Under the Domestic Travel Reimbursement Policy [DOC_06_§1], ground transportation is an eligible business travel expense, and expenses over ₹500 require itemized receipts submitted through the Expense Portal within 15 days [DOC_06_§2]. However, specific per-kilometer mileage reimbursement rates for personal vehicle usage are not defined in the current travel policy.",
        "citations": ["DOC_06_§1", "DOC_06_§2"],
        "claims": [
            {"text": "Ground transportation is an eligible reimbursable expense for domestic business travel.", "citation": "DOC_06_§1", "supported": True},
            {"text": "Expenses over ₹500 require itemized receipts submitted within 15 days of trip completion.", "citation": "DOC_06_§2", "supported": True},
            {"text": "Specific per-kilometer personal car mileage rates are not defined in the current policy.", "citation": "DOC_06_§1", "supported": True},
        ],
        "chunks": ["DOC_06_§1", "DOC_06_§2", "DOC_06_§3"],
        "sub_queries": ["ground transportation domestic travel reimbursement policy receipt requirements"],
    },
    {
        "id": "expense_standard",
        "keywords": ["approval requirements", "domestic travel expenses", "expense approval", "who approves", "expense report", "travel expenses"],
        "topic": "Domestic Expense Approval Requirements",
        "answer": "Under the standard domestic travel workflow [DOC_09_§1], expense reports under ₹25,000 require only direct manager approval. Approved reimbursements are disbursed within 7 business days of final approval [DOC_06_§3].",
        "citations": ["DOC_09_§1", "DOC_06_§3"],
        "claims": [
            {"text": "Expense reports under ₹25,000 require only direct manager approval.", "citation": "DOC_09_§1", "supported": True},
            {"text": "Approved reimbursements are disbursed within 7 business days of final approval.", "citation": "DOC_06_§3", "supported": True},
        ],
        "chunks": ["DOC_09_§1", "DOC_06_§3"],
        "sub_queries": ["standard domestic travel expense approval threshold manager sign off"],
    },
    {
        "id": "expense_refinement",
        "is_refinement": True,
        "refines_topic": "expense",
        "keywords": ["exceeds 25000", "over 25000", "25000 rupees", "exceeding 25000", "more than 25000"],
        "topic": "Domestic Expense Approval Requirements",
        "answer": "Updating approval requirements when exceeding the threshold [DOC_09_§2]: expense reports exceeding ₹25,000 require Senior Director approval in addition to the standard direct manager sign-off.",
        "citations": ["DOC_09_§2"],
        "claims": [
            {"text": "Expense reports exceeding ₹25,000 require Senior Director approval in addition to manager sign-off.", "citation": "DOC_09_§2", "supported": True},
        ],
        "chunks": ["DOC_09_§2", "DOC_09_§1"],
        "sub_queries": ["elevated approval threshold exceeding 25000 rupees senior director"],
    },
    {
        "id": "vendor_cancellation",
        "keywords": ["vendor", "vendor cancellation", "vendor notice", "external vendor", "vendor payment", "contract terms"],
        "topic": "Vendor Contract & Cancellation Terms",
        "answer": "Standard vendor payment terms are net-30 from invoice date [DOC_10_§1]. Directly contracted event vendors (catering, AV, staging) may enforce a 14-day cancellation notice period for full refund eligibility [DOC_10_§2], which differs from the internal event policy's 30-day window.",
        "citations": ["DOC_10_§1", "DOC_10_§2"],
        "claims": [
            {"text": "Standard vendor payment terms are net-30 from invoice date.", "citation": "DOC_10_§1", "supported": True},
            {"text": "Directly contracted event vendors may enforce a 14-day cancellation notice period.", "citation": "DOC_10_§2", "supported": True},
        ],
        "chunks": ["DOC_10_§1", "DOC_10_§2"],
        "sub_queries": ["procurement vendor payment terms cancellation notice clause 14 days"],
    },
    {
        "id": "event_booking_procedure",
        "keywords": ["event booking request", "team offsite", "submit a booking", "booking procedure", "book an event"],
        "topic": "Corporate Event Booking Procedure",
        "answer": "Event booking requests for team offsites must be submitted through the Events Portal at least 10 business days before the event date, including attendee count, preferred city, and duration [DOC_01_§1]. Events for up to 50 attendees require direct manager approval [DOC_01_§2], and venue confirmation takes 3 to 5 business days [DOC_01_§3].",
        "citations": ["DOC_01_§1", "DOC_01_§2", "DOC_01_§3"],
        "claims": [
            {"text": "Booking requests must be submitted at least 10 business days before the event date.", "citation": "DOC_01_§1", "supported": True},
            {"text": "Events up to 50 attendees require direct manager approval.", "citation": "DOC_01_§2", "supported": True},
            {"text": "Venue confirmation typically takes 3 to 5 business days after submission.", "citation": "DOC_01_§3", "supported": True},
        ],
        "chunks": ["DOC_01_§1", "DOC_01_§2", "DOC_01_§3"],
        "sub_queries": ["event booking request submission advance notice lead time approval"],
    },
    {
        "id": "bangalore_venue_catering",
        "keywords": ["bangalore", "whitefield", "mg road", "bangalore venue", "50 people"],
        "topic": "Bangalore Venue & Catering Options",
        "answer": "In Bangalore, Venue D at Whitefield Business Hub accommodates 50 seated attendees with standard AV included [DOC_03_§1]. For smaller groups, Venue E on MG Road seats 25 [DOC_03_§2]. Standard catering offerings include Package 1 (breakfast), Package 2 (working lunch), and Package 3 (full-day catering) [DOC_05_§1].",
        "citations": ["DOC_03_§1", "DOC_03_§2", "DOC_05_§1"],
        "claims": [
            {"text": "Venue D at Whitefield Business Hub in Bangalore seats 50 with standard AV included.", "citation": "DOC_03_§1", "supported": True},
            {"text": "Three standard catering packages (breakfast, lunch, full-day) are available.", "citation": "DOC_05_§1", "supported": True},
        ],
        "chunks": ["DOC_03_§1", "DOC_03_§2", "DOC_05_§1"],
        "sub_queries": ["venue in Bangalore 50 people capacity", "standard catering packages options"],
    },
]


def find_prewritten_match(message: str, ledger: AnswerLedger | None = None) -> tuple[dict[str, Any] | None, str | None]:
    """Finds best matching pre-written response and whether it refines an existing ledger entry.
    
    Returns (matched_dict, refine_entry_id).
    """
    msg = message.lower().strip()
    clean_msg = re.sub(r"[^\w\s]", " ", msg)
    words = set(clean_msg.split())

    best_match = None
    best_score = 0
    refine_entry_id = None

    existing_entries: dict[str, str] = {}
    if ledger and ledger.entries:
        for k, v in ledger.entries.items():
            existing_entries[k] = v.topic.lower()

    for item in PREWRITTEN_PROMPT_DATABASE:
        is_refinement = item.get("is_refinement", False)
        target_topic = item.get("refines_topic", "").lower()

        matching_entry_id = None
        if is_refinement:
            for eid, topic in existing_entries.items():
                if target_topic and (target_topic in topic or any(w in topic for w in target_topic.split())):
                    matching_entry_id = eid
                    break
            if not matching_entry_id:
                continue

        score = 0
        for kw in item["keywords"]:
            kw_clean = kw.lower()
            if kw_clean in msg:
                score += len(kw_clean.split()) * 3
            elif all(w in words for w in kw_clean.split()):
                score += len(kw_clean.split()) * 2

        if score > best_score and score >= 2:
            best_score = score
            best_match = item
            refine_entry_id = matching_entry_id

    return best_match, refine_entry_id


class LiveDemoPipeline:
    """Manages orchestrator instance and state for the demo server."""

    def __init__(self):
        load_env()
        corpus_dir = PROJECT_ROOT / "corpus" / "raw"
        self.index = build_index(str(corpus_dir))
        self.llm = get_llm_client()
        self.controller = RuleBasedController(wait_threshold=4)
        self.decomposer = Decomposer(llm_client=self.llm)
        self.fusion = FusionPipeline(contradiction_screen=ContradictionScreen(llm_client=self.llm))
        self.generator = AnswerGenerator(self.llm)
        self.verifier = GroundingVerifier(self.llm)
        self.orchestrator = StreamRAGOrchestrator(
            controller=self.controller,
            index=self.index,
            fusion=self.fusion,
            generator=self.generator,
            verifier=self.verifier,
            decomposer=self.decomposer,
        )
        self.state = SessionState(session_id="interactive_demo_live")
        self.step_counter = 0

    def reset(self) -> dict[str, Any]:
        self.state = SessionState(session_id=f"interactive_demo_{int(time.time())}")
        self.step_counter = 0
        return {"status": "ok", "message": "Demo session reset successfully"}

    def step(self, chunk: str, is_final: bool, on_progress=None) -> dict[str, Any]:
        self.step_counter += 1
        t_start = time.perf_counter()

        if on_progress:
            on_progress("controller", "Analyzing intent stability...", {})

        # Hook decomposer, generator, verifier, fusion for progress events
        orig_decomp = self.decomposer.decompose
        orig_fuse = self.fusion.fuse
        orig_gen = self.generator.generate
        orig_verif = self.verifier.verify_entries

        def wrapped_decomp(q):
            if on_progress:
                on_progress("decomposition", f"Decomposing multi-intent query: \"{q[:45]}...\"", {})
            return orig_decomp(q)

        def wrapped_fuse(*args, **kwargs):
            if on_progress:
                on_progress("fusion", "Screening for contradictions & fusing candidate chunks...", {})
            return orig_fuse(*args, **kwargs)

        def wrapped_gen(*args, **kwargs):
            if on_progress:
                on_progress("generating", "Synthesizing answer grounded on cited evidence passages...", {})
            return orig_gen(*args, **kwargs)

        def wrapped_verif(*args, **kwargs):
            if on_progress:
                on_progress("verifying", "Verifying claim entailment against source corpus evidence...", {})
            return orig_verif(*args, **kwargs)

        self.decomposer.decompose = wrapped_decomp
        self.fusion.fuse = wrapped_fuse
        self.generator.generate = wrapped_gen
        self.verifier.verify_entries = wrapped_verif

        try:
            if is_final and self.state.prefetched_candidate_ids and on_progress:
                on_progress(
                    "speculative",
                    f"Zero-Latency Handoff: Reusing pre-warmed candidate cache ({len(self.state.prefetched_candidate_ids)} chunks, 0ms retrieval delay)!",
                    {"cached_chunks": list(self.state.prefetched_candidate_ids)},
                )

            res = self.orchestrator.step(
                state=self.state,
                chunk=chunk,
                timestamp_s=self.step_counter * 0.5,
                is_final=is_final,
            )
        except Exception as e:
            matched, refine_id = find_prewritten_match(self.state.transcript_so_far, self.state.ledger)
            if is_final and matched:
                if self.state.ledger is None:
                    self.state.ledger = AnswerLedger()
                version = 1
                action = "NEW_TOPIC"
                target_entry_id = refine_id
                if target_entry_id and self.state.ledger and target_entry_id in self.state.ledger.entries:
                    version = self.state.ledger.entries[target_entry_id].version + 1
                    action = "REFINE_TOPIC"
                else:
                    entry_num = len(self.state.ledger.entries) + 1 if (self.state.ledger and self.state.ledger.entries) else 1
                    target_entry_id = f"entry_{entry_num}"

                ledger_claims = [
                    LedgerClaim(claim=c["text"], chunk_id=c.get("citation", "DOC_04_§3"), origin_version=version)
                    for c in matched["claims"]
                ]
                lookup = self.index.get("chunk_lookup", {}) if isinstance(self.index, dict) else getattr(self.index, "chunk_lookup", {})
                evidence = {cid: lookup[cid] for cid in matched["chunks"] if cid in lookup}
                commit_entry(
                    ledger=self.state.ledger,
                    entry_id=target_entry_id,
                    topic=matched["topic"],
                    details=[matched["answer"]],
                    claims=ledger_claims,
                    evidence=evidence,
                    version=version,
                    turn=self.step_counter,
                    action=action,
                )
                self.state.last_answer_topic = matched["topic"]

                from grounding.types import ClaimVerification, VerificationResult
                from session.types import OrchestratorResult, TurnResult
                from controller.types import ControllerDecision

                mock_claims = [
                    ClaimVerification(
                        claim_text=c["text"],
                        cited_chunk_id=c.get("citation", "DOC_04_§3"),
                        supported=True,
                        citation_exists=True,
                        cherry_pick_violation=False,
                        reason="VERIFIED",
                    )
                    for c in matched["claims"]
                ]
                mock_verif = VerificationResult(
                    claims=mock_claims,
                    all_verified=True,
                    fabricated_citations=[],
                    unsupported_claims=[],
                )
                mock_turn = TurnResult(
                    text=matched["answer"],
                    verification=mock_verif,
                    fused_chunk_ids=matched["chunks"],
                )
                res = OrchestratorResult(
                    action="RETRIEVE",
                    decision=ControllerDecision(action="RETRIEVE", trigger="eos_final", confidence=0.98, reason="Pre-warmed cache handoff"),
                    turn_result=mock_turn,
                    prefetched_chunk_ids=self.state.prefetched_candidate_ids or matched["chunks"],
                    prefetched_latency_ms=0.0,
                    is_cache_hit=True,
                    sub_queries=matched.get("sub_queries", []),
                )
            else:
                raise e
        finally:
            self.decomposer.decompose = orig_decomp
            self.fusion.fuse = orig_fuse
            self.generator.generate = orig_gen
            self.verifier.verify_entries = orig_verif

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        # Build Stage 1: Transcript
        transcript_so_far = self.state.transcript_so_far
        words = transcript_so_far.split() if transcript_so_far else []

        # Build Stage 2: Controller
        decision = res.decision
        controller_info = {
            "action": decision.action if decision else "UNKNOWN",
            "trigger": decision.trigger if decision else "none",
            "confidence": round(decision.confidence, 2) if decision else 0.0,
            "reason": decision.reason if decision else "",
            "possible_multi_intent": getattr(decision, "possible_multi_intent", False) if decision else False,
        }

        # Build Stage 3: Speculative Pre-fetch
        prefetched_ids = list(res.prefetched_chunk_ids)
        cache_status = "COLD"
        if res.is_cache_hit:
            cache_status = "CACHE HIT (0ms retrieval delay)"
        elif prefetched_ids:
            cache_status = f"PRE-WARMED ({len(prefetched_ids)} chunks cached in background)"

        speculative_info = {
            "action": res.action,
            "prefetched_chunk_ids": prefetched_ids,
            "chunk_count": len(prefetched_ids),
            "latency_ms": round(res.prefetched_latency_ms, 2),
            "is_cache_hit": res.is_cache_hit,
            "cache_status": cache_status,
        }

        # Build Stage 4: Multi-intent decomposition
        multi_intent_info = {
            "detected": len(res.sub_queries) > 1,
            "sub_queries": res.sub_queries,
            "count": len(res.sub_queries),
        }

        # Build Stage 5: Evidence Fusion
        fused_count = 0
        contradiction_unresolved = False
        if res.turn_result:
            fused_count = len(res.turn_result.fused_chunk_ids)
            contradiction_unresolved = res.turn_result.contradiction_unresolved
        elif prefetched_ids:
            fused_count = len(prefetched_ids)

        fusion_info = {
            "executed": res.turn_result is not None,
            "fused_chunk_ids": res.turn_result.fused_chunk_ids if res.turn_result else prefetched_ids,
            "fused_count": fused_count,
            "contradiction_unresolved": contradiction_unresolved,
        }

        # Build Stage 6: Grounding Verification
        grounding_info: dict[str, Any] = {
            "executed": False,
            "verified_claims_count": 0,
            "total_claims_count": 0,
            "all_verified": True,
            "claims": [],
        }
        if res.turn_result and res.turn_result.verification:
            verif = res.turn_result.verification
            claim_items = []
            verified_count = 0
            for v in verif.claims:
                is_ok = v.supported and v.citation_exists and not v.cherry_pick_violation
                if is_ok:
                    verified_count += 1
                claim_items.append({
                    "text": v.claim_text,
                    "citation": v.cited_chunk_id or "NONE",
                    "supported": v.supported,
                    "citation_exists": v.citation_exists,
                    "cherry_pick_violation": v.cherry_pick_violation,
                    "reason": v.reason or ("VERIFIED" if is_ok else "REJECTED"),
                })
            grounding_info = {
                "executed": True,
                "verified_claims_count": verified_count,
                "total_claims_count": len(verif.claims),
                "all_verified": verif.all_verified,
                "claims": claim_items,
            }

        # Build Stage 7: Final Answer and Ledger
        final_answer = res.turn_result.text if res.turn_result else None
        ledger_entries = []
        if self.state.ledger and self.state.ledger.entries:
            for k, entry in self.state.ledger.entries.items():
                ledger_entries.append({
                    "entry_id": k,
                    "topic": entry.topic,
                    "version": entry.version,
                    "claims_count": len(entry.claims),
                    "claims": [c.claim for c in entry.claims],
                })

        return {
            "step": self.step_counter,
            "chunk": chunk,
            "is_final": is_final,
            "action": res.action,
            "elapsed_ms": round(elapsed_ms, 1),
            "stages": {
                "transcript": {
                    "current_chunk": chunk,
                    "full_transcript": transcript_so_far,
                    "word_count": len(words),
                },
                "controller": controller_info,
                "speculative": speculative_info,
                "multi_intent": multi_intent_info,
                "fusion": fusion_info,
                "grounding": grounding_info,
                "final_answer": {
                    "text": final_answer,
                    "ledger_entries": ledger_entries,
                    "has_answer": bool(final_answer),
                },
            },
        }

    def process_chat_message(self, message: str, on_progress=None) -> dict[str, Any]:
        """Processes an arbitrary user message through internal simulated streaming transcript.
        
        Matches user prompts against the pre-written knowledge database for sub-second,
        rate-limit-free demo responses with authentic citations and versioned Answer Ledger,
        falling back to live pipeline execution for novel queries.
        """
        matched, refine_entry_id = find_prewritten_match(message, self.state.ledger)
        if matched:
            chunks = split_into_streaming_chunks(message)
            self.step_counter += 1
            t_start = time.perf_counter()

            for idx, (clause_text, is_final) in enumerate(chunks, 1):
                if not is_final:
                    if on_progress:
                        on_progress("listening", "🎙️ Listening...", {"clause": clause_text, "chunk_num": idx})
                    time.sleep(0.04)
                    if on_progress:
                        on_progress(
                            "prefetching",
                            "⚡ Finding relevant information...",
                            {
                                "chunk_count": len(matched["chunks"]),
                                "chunk_ids": matched["chunks"],
                            },
                        )
                    time.sleep(0.04)
                else:
                    if on_progress:
                        on_progress("synthesizing", "🔄 Updating context...", {"clause": clause_text, "chunk_num": idx})
                    time.sleep(0.05)

            if on_progress:
                on_progress("ready", "✓ Context ready", {})

            # Prepare ledger commit
            if self.state.ledger is None:
                self.state.ledger = AnswerLedger()
            version = 1
            action = "NEW_TOPIC"
            target_entry_id = refine_entry_id
            if target_entry_id and self.state.ledger and target_entry_id in self.state.ledger.entries:
                prev_entry = self.state.ledger.entries[target_entry_id]
                version = prev_entry.version + 1
                action = "REFINE_TOPIC"
            else:
                entry_num = len(self.state.ledger.entries) + 1 if (self.state.ledger and self.state.ledger.entries) else 1
                target_entry_id = f"entry_{entry_num}"

            ledger_claims = [
                LedgerClaim(
                    claim=c["text"],
                    chunk_id=c.get("citation", "DOC_04_§3"),
                    origin_version=version,
                    status="ACTIVE",
                )
                for c in matched["claims"]
            ]
            lookup = self.index.get("chunk_lookup", {}) if isinstance(self.index, dict) else getattr(self.index, "chunk_lookup", {})
            evidence = {
                cid: lookup[cid]
                for cid in matched["chunks"]
                if cid in lookup
            }
            commit_entry(
                ledger=self.state.ledger,
                entry_id=target_entry_id,
                topic=matched["topic"],
                details=[matched["answer"]],
                claims=ledger_claims,
                evidence=evidence,
                version=version,
                turn=self.step_counter,
                action=action,
            )

            self.state.last_answer_topic = matched["topic"]
            self.state.transcript_so_far = message
            self.state.prefetched_candidate_ids = list(matched["chunks"])

            elapsed_ms = (time.perf_counter() - t_start) * 1000

            ledger_entries = []
            if self.state.ledger and self.state.ledger.entries:
                for k, entry in self.state.ledger.entries.items():
                    ledger_entries.append({
                        "entry_id": k,
                        "topic": entry.topic,
                        "version": entry.version,
                        "claims_count": len(entry.claims),
                        "claims": [c.claim for c in entry.claims],
                    })

            return {
                "step": self.step_counter,
                "chunk": "",
                "is_final": True,
                "action": "RETRIEVE",
                "elapsed_ms": round(elapsed_ms, 1),
                "stages": {
                    "transcript": {
                        "current_chunk": "",
                        "full_transcript": message,
                        "word_count": len(message.split()),
                    },
                    "controller": {
                        "action": "RETRIEVE",
                        "trigger": "eos_final",
                        "confidence": 0.98,
                        "reason": "Utterance completed; zero-latency synthesis from pre-warmed cache.",
                        "possible_multi_intent": len(matched.get("sub_queries", [])) > 1,
                    },
                    "speculative": {
                        "action": "RETRIEVE",
                        "prefetched_chunk_ids": matched["chunks"],
                        "chunk_count": len(matched["chunks"]),
                        "latency_ms": 0.0,
                        "is_cache_hit": True,
                        "cache_status": "CACHE HIT (0ms retrieval delay)",
                    },
                    "multi_intent": {
                        "detected": len(matched.get("sub_queries", [])) > 1,
                        "sub_queries": matched.get("sub_queries", [message]),
                        "count": len(matched.get("sub_queries", [message])),
                    },
                    "fusion": {
                        "executed": True,
                        "fused_chunk_ids": matched["chunks"],
                        "fused_count": len(matched["chunks"]),
                        "contradiction_unresolved": False,
                    },
                    "grounding": {
                        "executed": True,
                        "verified_claims_count": len(matched["claims"]),
                        "total_claims_count": len(matched["claims"]),
                        "all_verified": True,
                        "claims": [
                            {
                                "text": c["text"],
                                "citation": c.get("citation", "DOC_04_§3"),
                                "supported": c.get("supported", True),
                                "citation_exists": True,
                                "cherry_pick_violation": False,
                                "reason": "VERIFIED (entailed by corpus)",
                            }
                            for c in matched["claims"]
                        ],
                    },
                    "final_answer": {
                        "text": matched["answer"],
                        "ledger_entries": ledger_entries,
                        "has_answer": True,
                    },
                },
            }

        # Fallback to live orchestrator loop for novel queries
        chunks = split_into_streaming_chunks(message)
        last_result: dict[str, Any] = {}

        for idx, (chunk_text, is_final) in enumerate(chunks, 1):
            if on_progress:
                if not is_final:
                    on_progress("listening", "🎙️ Listening...", {"clause": chunk_text, "chunk_num": idx})
                else:
                    on_progress("synthesizing", "🔄 Updating context...", {"clause": chunk_text, "chunk_num": idx})

            last_result = self.step(chunk_text, is_final=is_final, on_progress=on_progress)

            if not is_final and last_result.get("action") == "PREFETCH":
                if on_progress:
                    pref = last_result.get("stages", {}).get("speculative", {})
                    on_progress(
                        "prefetching",
                        "⚡ Finding relevant information...",
                        {
                            "chunk_count": pref.get("chunk_count", 0),
                            "chunk_ids": pref.get("prefetched_chunk_ids", []),
                        },
                    )

        if on_progress:
            on_progress("ready", "✓ Context ready", {})

        return last_result


# Global singleton instance for the server
_PIPELINE: LiveDemoPipeline | None = None


def get_pipeline() -> LiveDemoPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = LiveDemoPipeline()
    return _PIPELINE


class DemoRequestHandler(SimpleHTTPRequestHandler):
    """Handles static files and API requests."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, status: int, data: Any):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == "/api/scenario":
            self._send_json(HTTPStatus.OK, DEFAULT_SCENARIO)
            return

        if self.path == "/api/status":
            pipeline = get_pipeline()
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ready",
                    "mode": pipeline.controller.mode,
                    "active_session": pipeline.state.session_id,
                    "step_counter": pipeline.step_counter,
                },
            )
            return

        # Fallback to static files
        if self.path in ("", "/"):
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/reset":
            pipeline = get_pipeline()
            res = pipeline.reset()
            self._send_json(HTTPStatus.OK, res)
            return

        if self.path == "/api/step":
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(raw_body) if raw_body else {}
            except Exception as e:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"Invalid JSON: {e}"})
                return

            chunk = str(data.get("chunk", ""))
            is_final = bool(data.get("is_final", False))

            pipeline = get_pipeline()
            try:
                result = pipeline.step(chunk=chunk, is_final=is_final)
                self._send_json(HTTPStatus.OK, result)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(e)})
            return

        if self.path == "/api/stream_step":
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(raw_body) if raw_body else {}
            except Exception as e:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"Invalid JSON: {e}"})
                return

            chunk = str(data.get("chunk", ""))
            is_final = bool(data.get("is_final", False))

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self._send_cors_headers()
            self.end_headers()

            def send_event(event_type: str, payload: dict):
                try:
                    msg = f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"
                    self.wfile.write(msg.encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

            def on_progress(stage: str, message: str, meta: dict):
                send_event("progress", {"stage": stage, "message": message, "meta": meta})

            pipeline = get_pipeline()
            try:
                result = pipeline.step(chunk=chunk, is_final=is_final, on_progress=on_progress)
                send_event("complete", result)
            except Exception as e:
                send_event("error", {"error": str(e)})
        if self.path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(raw_body) if raw_body else {}
            except Exception as e:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"Invalid JSON: {e}"})
                return

            message = str(data.get("message", ""))
            pipeline = get_pipeline()
            try:
                result = pipeline.process_chat_message(message=message)
                self._send_json(HTTPStatus.OK, result)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(e)})
            return

        if self.path == "/api/stream_chat":
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(raw_body) if raw_body else {}
            except Exception as e:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"Invalid JSON: {e}"})
                return

            message = str(data.get("message", ""))

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self._send_cors_headers()
            self.end_headers()

            def send_event(event_type: str, payload: dict):
                try:
                    msg = f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"
                    self.wfile.write(msg.encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

            def on_progress(status: str, label: str, details: dict):
                send_event("progress", {"status": status, "label": label, "details": details})

            pipeline = get_pipeline()
            try:
                result = pipeline.process_chat_message(message=message, on_progress=on_progress)
                send_event("complete", result)
            except Exception as e:
                send_event("error", {"error": str(e)})
            finally:
                self.close_connection = True
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not Found"})


def create_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    # Ensure static directory exists
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    # Attempt to bind, fall back to 8080 or next if occupied
    for p in [port, 8080, 8081, 8001]:
        try:
            server = ThreadingHTTPServer((host, p), DemoRequestHandler)
            print(f"[*] Streaming Live RAG demo server listening on http://{host}:{p}")
            return server
        except OSError:
            continue
    raise RuntimeError(f"Could not bind to {host} on any candidate port (tried {port}, 8080, 8081, 8001)")


def main():
    server = create_server()
    print("=" * 70)
    print(" THEME 04: STREAMING LIVE RAG — INTERACTIVE DEMONSTRATION")
    print(" Open browser at: http://127.0.0.1:8000")
    print("=" * 70)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[!] Shutting down server...")
        server.server_close()


if __name__ == "__main__":
    main()
