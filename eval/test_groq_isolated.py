"""Isolated comparison benchmark: Ollama vs Groq on real contradiction & grounding prompts.

Measures:
- Latency (wall-clock seconds)
- Response validity & schema validation via real _extract_first_json_object parser
- Token usage (when exposed by provider)
- Strictly avoids logging or exposing API keys
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controller.llm_factory import load_env
from controller.model_based import OpenAICompatibleLLMClient, _extract_first_json_object
from fusion.contradiction import CONTRADICTION_SYSTEM_PROMPT
from grounding.entailment import ENTAILMENT_SYSTEM_PROMPT


# Real contradiction test data (DOC_04_§1 vs DOC_10_§2)
SUB_QUERY = "cancellation notice period and refund terms for event vendors"
CHUNK_04_TEXT = (
    "Bookings cancelled more than 30 days before the event date receive a full refund of any deposit paid. "
    "Cancellations made 15-30 days before the event receive a 50% refund. "
    "Cancellations made fewer than 15 days before the event are non-refundable."
)
CHUNK_10_TEXT = (
    "Event vendors contracted directly (catering, AV, staging) may enforce a 14-day cancellation notice period "
    "for full refund eligibility, which may differ from the internal Event Cancellation & Refund Policy's 30-day "
    "window; employees should confirm vendor-specific terms at time of contracting."
)
CONTRADICTION_USER_PROMPT = (
    f'Sub-query: "{SUB_QUERY}"\n'
    f'Passage A: "{CHUNK_04_TEXT}"\n'
    f'Passage B: "{CHUNK_10_TEXT}"'
)

# Real grounding test data
GROUNDING_CLAIM = "Cancellations made more than 30 days before the event receive a full refund."
GROUNDING_PASSAGE = (
    "Bookings cancelled more than 30 days before the event date receive a full refund of any deposit paid."
)
GROUNDING_USER_PROMPT = (
    f'Claim: "{GROUNDING_CLAIM.strip()}"\n'
    f'Passage: "{GROUNDING_PASSAGE.strip()}"'
)


def validate_contradiction_response(raw: str) -> tuple[bool, bool, dict | None]:
    """Uses real project parser to validate contradiction schema."""
    data = _extract_first_json_object(raw)
    if data is None:
        return False, False, None
    valid_schema = (
        "is_contradiction" in data
        and isinstance(data["is_contradiction"], bool)
        and "reason" in data
    )
    return True, valid_schema, data


def validate_grounding_response(raw: str) -> tuple[bool, bool, dict | None]:
    """Uses real project parser to validate grounding entailment schema."""
    data = _extract_first_json_object(raw)
    if data is None:
        return False, False, None
    valid_schema = (
        "supported" in data
        and isinstance(data["supported"], bool)
        and "reason" in data
    )
    return True, valid_schema, data


def run_benchmark():
    load_env()
    sys.stdout.reconfigure(encoding="utf-8")

    results = {}

    print("=" * 70)
    print("ISOLATED PROVIDER BENCHMARK: Ollama vs Groq")
    print("=" * 70)

    # 1. Ollama Test (Contradiction)
    print("\n[1/3] Testing Ollama (phi35-4k:latest) on Contradiction Screen...")
    ollama_base_url = os.environ.get("OLLAMA_BASE_URL", os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"))
    ollama_model = os.environ.get("OLLAMA_MODEL", os.environ.get("LLM_MODEL", "phi35-4k:latest"))
    ollama_client = OpenAICompatibleLLMClient(base_url=ollama_base_url, api_key="ollama", model=ollama_model)

    try:
        t0 = time.perf_counter()
        ollama_raw = ollama_client.complete(CONTRADICTION_SYSTEM_PROMPT, CONTRADICTION_USER_PROMPT)
        ollama_latency = time.perf_counter() - t0
        ollama_json_ok, ollama_schema_ok, ollama_data = validate_contradiction_response(ollama_raw)
        results["ollama_contradiction"] = {
            "latency_s": ollama_latency,
            "json_parsed": ollama_json_ok,
            "schema_valid": ollama_schema_ok,
            "data": ollama_data,
            "raw": ollama_raw.strip(),
            "usage": ollama_client.last_usage,
        }
        print(f"  ✓ Latency: {ollama_latency:.3f}s")
        print(f"  ✓ JSON Parsed: {ollama_json_ok}, Schema Valid: {ollama_schema_ok}")
        print(f"  ✓ Parsed Result: {ollama_data}")
    except Exception as e:
        print(f"  ✗ Ollama test failed: {type(e).__name__}: {e}")
        results["ollama_contradiction"] = {"error": str(e)}

    # Check Groq Key
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        print("\n[!] GROQ_API_KEY is not set in environment or .env.")
        print("    Please set GROQ_API_KEY in your environment or .env to run Groq tests.")
        print("    (The key will never be printed or logged).")
        return results

    groq_model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
    groq_base_url = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    groq_client = OpenAICompatibleLLMClient(base_url=groq_base_url, api_key=groq_key, model=groq_model)

    # 2. Groq Test (Contradiction)
    print(f"\n[2/3] Testing Groq ({groq_model}) on Contradiction Screen...")
    try:
        t0 = time.perf_counter()
        groq_raw = groq_client.complete(CONTRADICTION_SYSTEM_PROMPT, CONTRADICTION_USER_PROMPT)
        groq_latency = time.perf_counter() - t0
        groq_json_ok, groq_schema_ok, groq_data = validate_contradiction_response(groq_raw)
        results["groq_contradiction"] = {
            "latency_s": groq_latency,
            "json_parsed": groq_json_ok,
            "schema_valid": groq_schema_ok,
            "data": groq_data,
            "raw": groq_raw.strip(),
            "usage": groq_client.last_usage,
        }
        print(f"  ✓ Latency: {groq_latency:.3f}s")
        print(f"  ✓ JSON Parsed: {groq_json_ok}, Schema Valid: {groq_schema_ok}")
        print(f"  ✓ Parsed Result: {groq_data}")
        print(f"  ✓ Token Usage: {groq_client.last_usage}")
        if not groq_schema_ok:
            print(f"  [!] Raw Groq Output:\n{groq_raw}")
    except Exception as e:
        print(f"  ✗ Groq contradiction test failed: {type(e).__name__}: {e}")
        results["groq_contradiction"] = {"error": str(e)}

    # 3. Groq Test (Grounding Entailment)
    print(f"\n[3/3] Testing Groq ({groq_model}) on Grounding Entailment...")
    try:
        t0 = time.perf_counter()
        groq_g_raw = groq_client.complete(ENTAILMENT_SYSTEM_PROMPT, GROUNDING_USER_PROMPT)
        groq_g_latency = time.perf_counter() - t0
        groq_g_json_ok, groq_g_schema_ok, groq_g_data = validate_grounding_response(groq_g_raw)
        results["groq_grounding"] = {
            "latency_s": groq_g_latency,
            "json_parsed": groq_g_json_ok,
            "schema_valid": groq_g_schema_ok,
            "data": groq_g_data,
            "raw": groq_g_raw.strip(),
            "usage": groq_client.last_usage,
        }
        print(f"  ✓ Latency: {groq_g_latency:.3f}s")
        print(f"  ✓ JSON Parsed: {groq_g_json_ok}, Schema Valid: {groq_g_schema_ok}")
        print(f"  ✓ Parsed Result: {groq_g_data}")
        print(f"  ✓ Token Usage: {groq_client.last_usage}")
    except Exception as e:
        print(f"  ✗ Groq grounding test failed: {type(e).__name__}: {e}")
        results["groq_grounding"] = {"error": str(e)}

    # Comparison Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    if "ollama_contradiction" in results and "groq_contradiction" in results:
        o = results["ollama_contradiction"]
        g = results["groq_contradiction"]
        if "latency_s" in o and "latency_s" in g:
            speedup = o["latency_s"] / g["latency_s"] if g["latency_s"] > 0 else 0
            print(f"Contradiction Latency: Ollama = {o['latency_s']:.3f}s | Groq = {g['latency_s']:.3f}s ({speedup:.1f}x speedup)")
            print(f"Contradiction Schema:  Ollama = {'VALID' if o['schema_valid'] else 'INVALID'} | Groq = {'VALID' if g['schema_valid'] else 'INVALID'}")
            if g.get("usage"):
                print(f"Groq Contradiction Tokens: {g['usage']}")
    if "groq_grounding" in results:
        gg = results["groq_grounding"]
        if "latency_s" in gg:
            print(f"Groq Grounding Latency: {gg['latency_s']:.3f}s | Schema: {'VALID' if gg['schema_valid'] else 'INVALID'}")
            if gg.get("usage"):
                print(f"Groq Grounding Tokens: {gg['usage']}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_benchmark()
