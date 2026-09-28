"""Model-based Retrieval Controller.

Approximates Self-RAG's reflection-token idea (Asai et al. 2023) via prompting
a general-purpose instruction-tuned model instead of fine-tuning one — we have
no annotated reflection-token training data and no time to produce it. This is
a deliberate substitution, not a reproduction of Self-RAG: expect weaker
calibration, offset by zero training cost.

The prompt is domain-agnostic and contains no corpus-specific or scenario-
specific content (no-hardcoding constraint) — it describes the WAIT / RETRIEVE
/ SUPPRESS decision abstractly and receives only the live transcript.
"""
import json
from typing import Protocol, Optional
from .types import SessionState, RetrievalDecision

CONTROLLER_SYSTEM_PROMPT = """You are the retrieval-timing controller in a streaming RAG system.
Given the transcript of a user's utterance so far (which may be incomplete), decide ONE action:

- WAIT: the intent is still unstable or too incomplete to search on.
- RETRIEVE: a stable, meaningful, search-worthy intent (or sub-intent) is present, even if the
  utterance may not be finished. Set trigger to "multi_intent" if the transcript now contains
  two or more distinct, separately-searchable requests; otherwise "provisional" or "final".
- SUPPRESS: the request only asks to reformat, shorten, repeat, or otherwise transform an
  existing prior answer, and needs no new evidence.

Before deciding, count the number of distinct, independently-answerable requests in the transcript (e.g., 'book a venue' and 'tell me catering options' = 2). If that count is ≥ 2, possible_multi_intent must be true and trigger must be "multi_intent", regardless of overall confidence.

Example:
Transcript so far: "what are the library hours on weekends and how do I renew a book online"
{"action": "RETRIEVE", "trigger": "multi_intent", "confidence": 0.95, "reason": "Two distinct requests: library weekend hours and online book renewal", "possible_multi_intent": true}

Respond with ONLY compact JSON, no prose:
{"action": "WAIT|RETRIEVE|SUPPRESS", "trigger": "provisional|final|multi_intent|suppression|none",
 "confidence": 0.0-1.0, "reason": "...", "possible_multi_intent": true|false}"""


class LLMClient(Protocol):
    def complete(self, system: str, user: str, *, max_tokens: int | None = None) -> str: ...


class MockLLMClient:
    """Deterministic offline stand-in so the pipeline is runnable without API
    credentials. NOT a substitute for the real ablation run — swap in
    AnthropicLLMClient (or any LLMClient) before reporting G2/G3 numbers."""
    def complete(self, system: str, user: str, *, max_tokens: int | None = None) -> str:
        text = user.lower()
        if any(k in text for k in ("repeat", "bullet", "shorter", "rephrase")):
            return json.dumps({"action": "SUPPRESS", "trigger": "suppression",
                                "confidence": 0.8, "reason": "mock: formatting request",
                                "possible_multi_intent": False})
        if len(text.split()) < 6:
            return json.dumps({"action": "WAIT", "trigger": "none", "confidence": 0.5,
                                "reason": "mock: too short/unstable", "possible_multi_intent": False})
        multi = " and the " in text or " and catering" in text
        return json.dumps({"action": "RETRIEVE",
                            "trigger": "multi_intent" if multi else "provisional",
                            "confidence": 0.7, "reason": "mock: stable request detected",
                            "possible_multi_intent": multi})


class AnthropicLLMClient:
    """Real client. Requires ANTHROPIC_API_KEY in the environment."""
    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def complete(self, system: str, user: str, *, max_tokens: int | None = None) -> str:
        tokens = 200 if max_tokens is None else max_tokens
        resp = self.client.messages.create(
            model=self.model, max_tokens=tokens, temperature=0, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")


class OpenAICompatibleLLMClient:
    """Works with any provider that speaks the OpenAI chat-completions wire
    format: OpenAI itself, Ollama (local), vLLM, LM Studio, OpenRouter, Groq,
    Together, etc. This is the client to use for "bring your own API key /
    local model" — provider identity is just (base_url, api_key, model).

    Ollama example (no real key needed, Ollama ignores it):
        OpenAICompatibleLLMClient(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model="llama3.1:8b",
        )

    Hosted provider example:
        OpenAICompatibleLLMClient(
            base_url="https://api.openai.com/v1",
            api_key=os.environ["OPENAI_API_KEY"],
            model="gpt-4o-mini",
        )
    """
    def __init__(self, base_url: str, api_key: str, model: str):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def complete(self, system: str, user: str, *, max_tokens: int | None = None) -> str:
        tokens = 200 if max_tokens is None else max_tokens
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=tokens,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


def _extract_first_json_object(raw: str) -> Optional[dict]:
    """Finds and parses the first balanced top-level JSON object in raw text.
    Handles markdown fences, leading/trailing prose, and duplicate blocks.
    """
    start = raw.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(raw)):
            char = raw[i]
            if escape:
                escape = False
                continue
            if in_string:
                if char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = raw[start : i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        break
                    break
        start = raw.find("{", start + 1)
    return None


class ModelBasedController:
    mode = "model_based"

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or MockLLMClient()

    def decide(self, state: SessionState, new_chunk: str, timestamp_s: float) -> RetrievalDecision:
        state.append_chunk(new_chunk)
        user_prompt = f'Transcript so far: "{state.transcript_so_far}"'
        raw = self.llm.complete(CONTROLLER_SYSTEM_PROMPT, user_prompt)
        print(f"\n--- RAW LLM OUTPUT ---\n{raw}\n----------------------\n")
        data = _extract_first_json_object(raw)
        if data is None:
            # Fail-safe: never fabricate a retrieval trigger on a parse failure.
            return RetrievalDecision("WAIT", "none", 0.0, "controller parse failure -> defaulting to WAIT")
        return RetrievalDecision(
            action=data.get("action", "WAIT"),
            trigger=data.get("trigger", "none"),
            confidence=float(data.get("confidence", 0.5)),
            reason=data.get("reason", ""),
            possible_multi_intent=bool(data.get("possible_multi_intent", False)),
        )

