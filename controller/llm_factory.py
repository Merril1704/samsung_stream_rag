"""Single point of provider selection, driven entirely by environment
variables. Nothing in the controller/pipeline code should import a concrete
LLMClient directly — import get_llm_client() instead, so switching between
Ollama, Anthropic, or a hosted OpenAI-compatible provider is a config change,
not a code change.

Env vars:
  LLM_PROVIDER   "anthropic" | "openai_compatible" | "mock"   (default: mock)

  # anthropic
  ANTHROPIC_API_KEY
  ANTHROPIC_MODEL          (default: claude-haiku-4-5-20251001)

  # openai_compatible (covers Ollama, vLLM, LM Studio, OpenRouter, real OpenAI, etc.)
  LLM_BASE_URL             e.g. http://localhost:11434/v1  for Ollama
  LLM_API_KEY              any non-empty string for Ollama; real key for hosted providers
  LLM_MODEL                e.g. llama3.1:8b  or  gpt-4o-mini
  LLM_CONTEXT_TOKENS       e.g. 4096
"""
import os
from pathlib import Path
from .model_based import MockLLMClient, AnthropicLLMClient, OpenAICompatibleLLMClient, LLMClient
from .llm_wrapper import CountingLLMClient


def load_env() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def get_llm_client(counting: bool = False) -> LLMClient:
    load_env()
    provider = os.environ.get("LLM_PROVIDER", "mock").lower()

    if provider == "mock":
        client = MockLLMClient()
    elif provider == "anthropic":
        model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        client = AnthropicLLMClient(model=model)
    elif provider == "openai_compatible":
        base_url = os.environ["LLM_BASE_URL"]
        api_key = os.environ.get("LLM_API_KEY", "unused")
        model = os.environ["LLM_MODEL"]
        client = OpenAICompatibleLLMClient(base_url=base_url, api_key=api_key, model=model)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")

    if counting:
        ctx_env = os.environ.get("LLM_CONTEXT_TOKENS")
        context_tokens = int(ctx_env.strip()) if ctx_env and ctx_env.strip() else None
        return CountingLLMClient(client, context_tokens=context_tokens)

    return client
