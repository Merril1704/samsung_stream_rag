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
from session.orchestrator import StreamRAGOrchestrator

STATIC_DIR = Path(__file__).resolve().parent / "static"

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

    def step(self, chunk: str, is_final: bool) -> dict[str, Any]:
        self.step_counter += 1
        t_start = time.perf_counter()

        res = self.orchestrator.step(
            state=self.state,
            chunk=chunk,
            timestamp_s=self.step_counter * 0.5,
            is_final=is_final,
        )
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
