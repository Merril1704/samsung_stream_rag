"""LLM wrapper providing usage accounting and context-window guardrails."""
import time


class ContextBudgetExceeded(Exception):
    """Raised when estimated input tokens plus max output tokens exceed the context budget."""
    pass


class CountingLLMClient:
    """Wraps an LLMClient to track call counts, payload sizes, and latencies,
    and enforce an estimated context window limit.
    """
    def __init__(self, inner, context_tokens: int | None = None):
        self.inner = inner
        self.context_tokens = context_tokens
        self.calls: int = 0
        self.log: list[dict] = []

    def reset(self) -> None:
        self.calls = 0
        self.log = []

    def complete(self, system: str, user: str, **kwargs) -> str:
        estimate = (len(system) + len(user)) / 3
        max_output_tokens = kwargs.get("max_tokens") or 200

        if self.context_tokens is not None and (estimate + max_output_tokens) > self.context_tokens:
            raise ContextBudgetExceeded(
                f"Estimated prompt tokens ({estimate:.1f}) + output tokens ({max_output_tokens}) "
                f"exceeds context budget ({self.context_tokens})"
            )

        t0 = time.perf_counter()
        if kwargs:
            out = self.inner.complete(system, user, **kwargs)
        else:
            out = self.inner.complete(system, user)
        latency = time.perf_counter() - t0

        self.calls += 1
        self.log.append({
            "system_chars": len(system),
            "user_chars": len(user),
            "out_chars": len(out),
            "latency_s": latency,
        })
        return out
