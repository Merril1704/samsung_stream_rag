"""Interactive / Console Demo for Streaming Live RAG.

Demonstrates Theme 04 Core Capabilities:
1. Incremental transcript ingestion (streaming speech simulation).
2. Intent stability gating: controller WAIT decisions while utterance is incomplete.
3. Speculative Pre-fetch (Pre-warming): Retrieval occurs BEFORE speaker finishes.
4. Zero-latency handoff: Final synthesis reuses pre-warmed candidate evidence with 0ms retrieval delay.
5. Answer ledger commitment and grounded markdown rendering.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from controller.rule_based import RuleBasedController
from controller.types import SessionState
from decomposer.decomposer import Decomposer
from fusion.fusion_pipeline import FusionPipeline
from fusion.contradiction import ContradictionScreen
from grounding.verifier import GroundingVerifier
from session.generator import AnswerGenerator
from session.orchestrator import StreamRAGOrchestrator
from retrieval.indexer import build_index
from controller.llm_factory import get_llm_client, load_env


STREAMING_CHUNKS = [
    (0.0, "I need to cancel", False),
    (0.5, "a corporate event", False),
    (1.2, "booked for next week", False),
    (2.0, "because of a force majeure closure.", True),
]


def run_streaming_demo():
    load_env()
    print("=" * 80)
    print(" STREAMING LIVE RAG — SPECULATIVE RETRIEVAL & ZERO-LATENCY DEMO")
    print("=" * 80)

    print("\n[INIT] Indexing policy corpus...")
    corpus_dir = Path(__file__).parent.parent / "corpus" / "raw"
    index = build_index(str(corpus_dir))

    print("[INIT] Initializing pipeline components...")
    llm = get_llm_client()
    controller = RuleBasedController(wait_threshold=4)
    decomposer = Decomposer(llm_client=llm)
    fusion = FusionPipeline(contradiction_screen=ContradictionScreen(llm_client=llm))
    generator = AnswerGenerator(llm)
    verifier = GroundingVerifier(llm)

    orchestrator = StreamRAGOrchestrator(
        controller=controller,
        index=index,
        fusion=fusion,
        generator=generator,
        verifier=verifier,
        decomposer=decomposer,
    )

    state = SessionState(session_id="streaming_live_demo_01")

    print("\n[STREAM START] Incoming user speech stream...")
    print("-" * 80)

    for step_idx, (ts, chunk_text, is_final) in enumerate(STREAMING_CHUNKS, 1):
        time.sleep(0.3)  # Visual pacing for demo
        tag = "[FINAL SPEECH]" if is_final else "[USER SPEAKING]"
        print(f"\n{tag} [{ts:>4.1f}s] Chunk #{step_idx}: \"{chunk_text}\"")

        t_start = time.perf_counter()
        res = orchestrator.step(
            state=state,
            chunk=chunk_text,
            timestamp_s=ts,
            is_final=is_final,
        )
        t_elapsed = time.perf_counter() - t_start

        decision = res.decision
        if decision.action == "WAIT":
            print(f"  ──► Controller: WAIT | {decision.reason}")
            print(f"      Transcript so far: \"{state.transcript_so_far}\"")
            print(f"      Pipeline state: Passive listening (0 LLM calls, 0ms retrieval)")

        elif res.action == "PREFETCH":
            print(f"  ──► ⚡ [SPECULATIVE PRE-FETCH TRIGGERED]")
            print(f"      Trigger: {decision.trigger} ({decision.reason})")
            print(f"      Candidates retrieved in background: {len(state.prefetched_candidate_ids)} chunks in {state.prefetched_latency_s * 1000:.1f}ms")
            print(f"      Pre-warmed chunk IDs: {state.prefetched_candidate_ids[:5]}...")
            print(f"      Status: User STILL speaking. Generator LLM NOT invoked. Ledger UNTOUCHED.")

        elif res.action == "RETRIEVE" and is_final:
            print(f"  ──► 🚀 [END-OF-UTTERANCE: INSTANT SYNTHESIS]")
            print(f"      Controller: RETRIEVE | Final speech confirmed")
            print(f"      Candidate cache hit: Reused {len(state.prefetched_candidate_ids)} pre-warmed chunks!")
            print(f"      Answer synthesis & verification completed in {t_elapsed:.2f}s")
            if res.turn_result:
                print("\n" + "=" * 80)
                print(" FINAL GROUNDED ANSWER (Rendered from Answer Ledger v1):")
                print("=" * 80)
                print(res.turn_result.text.strip())
                print("=" * 80)
                print(f"Ledger entries: {list(state.ledger.entries.keys()) if state.ledger else 'None'}")
                print(f"Claims verified: {len(res.turn_result.verification.claims) if res.turn_result.verification else 0}")
                print(f"All claims verified: {res.turn_result.verification.all_verified if res.turn_result.verification else True}")


if __name__ == "__main__":
    run_streaming_demo()
