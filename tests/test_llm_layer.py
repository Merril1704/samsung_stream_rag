"""Tests for the LLM layer: CountingLLMClient, ContextBudgetExceeded, and llm_factory."""
import pytest
from controller.llm_wrapper import CountingLLMClient, ContextBudgetExceeded
from controller.model_based import MockLLMClient, OpenAICompatibleLLMClient
from controller.llm_factory import get_llm_client


class DummyInnerClient:
    def __init__(self):
        self.calls_received = []

    def complete(self, system: str, user: str, **kwargs) -> str:
        self.calls_received.append((system, user, kwargs))
        return "response_text"


def test_counting_client_counts_and_forwards_max_tokens_only_when_passed():
    inner = DummyInnerClient()
    client = CountingLLMClient(inner, context_tokens=1000)

    # Call 1: without max_tokens kwargs
    out1 = client.complete("system instruction", "user message")
    assert out1 == "response_text"
    assert client.calls == 1
    assert len(client.log) == 1
    assert client.log[0]["system_chars"] == len("system instruction")
    assert client.log[0]["user_chars"] == len("user message")
    assert client.log[0]["out_chars"] == len("response_text")
    assert client.log[0]["latency_s"] >= 0
    assert inner.calls_received[0] == ("system instruction", "user message", {})

    # Call 2: with explicit max_tokens
    out2 = client.complete("sys2", "usr2", max_tokens=150)
    assert out2 == "response_text"
    assert client.calls == 2
    assert len(client.log) == 2
    assert inner.calls_received[1] == ("sys2", "usr2", {"max_tokens": 150})

    # Test reset
    client.reset()
    assert client.calls == 0
    assert client.log == []


def test_context_guard_raises_and_does_not_count_call():
    inner = DummyInnerClient()
    # context_tokens set to 50
    # prompt chars: 30 sys + 30 user = 60 chars -> 20 tokens
    # default max_tokens: 200 -> total 220 > 50 -> ContextBudgetExceeded
    client = CountingLLMClient(inner, context_tokens=50)

    with pytest.raises(ContextBudgetExceeded) as exc_info:
        client.complete("a" * 30, "b" * 30)

    assert "exceeds context budget" in str(exc_info.value)
    assert client.calls == 0
    assert len(client.log) == 0
    assert len(inner.calls_received) == 0


def test_factory_returns_mock_client(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    client = get_llm_client()
    assert isinstance(client, MockLLMClient)


def test_factory_returns_counting_wrapper(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("LLM_CONTEXT_TOKENS", "2048")
    client = get_llm_client(counting=True)
    assert isinstance(client, CountingLLMClient)
    assert client.context_tokens == 2048
    assert isinstance(client.inner, MockLLMClient)


def test_factory_returns_openai_compatible_client(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    client = get_llm_client()
    assert isinstance(client, OpenAICompatibleLLMClient)
    assert client.model == "test-model"


def test_factory_returns_ollama_client(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    client = get_llm_client()
    assert isinstance(client, OpenAICompatibleLLMClient)
    assert client.model == "phi35-4k:latest"


def test_factory_returns_groq_client(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test_mock_groq_key")
    client = get_llm_client()
    assert isinstance(client, OpenAICompatibleLLMClient)
    assert client.model == "openai/gpt-oss-20b"


def test_factory_groq_missing_key_raises_value_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setattr("controller.llm_factory.load_env", lambda: None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError) as exc:
        get_llm_client()
    assert "GROQ_API_KEY environment variable is required" in str(exc.value)


def test_counting_client_captures_usage_from_inner():
    class UsageMockClient:
        def __init__(self):
            self.last_usage = {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23}

        def complete(self, system: str, user: str, **kwargs) -> str:
            return "ok"

    inner = UsageMockClient()
    counting = CountingLLMClient(inner)
    counting.complete("sys", "usr")
    assert len(counting.log) == 1
    assert counting.log[0]["usage"] == {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23}
