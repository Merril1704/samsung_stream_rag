from __future__ import annotations
import time
from typing import Callable
from telemetry.hooks import stage_context, get_current_stage, infer_stage_from_prompt


class ContextBudgetExceeded(Exception):
    """Raised when estimated input tokens plus max output tokens exceed the context budget."""
    pass


class CountingLLMClient:
    """Wraps an LLMClient to track call counts, payload sizes, and latencies,
    and enforce an estimated context window limit.
    """
    def __init__(
        self,
        inner,
        context_tokens: int | None = None,
        on_call: Callable[[str, dict], None] | None = None,
    ):
        self.inner = inner
        self.context_tokens = context_tokens
        self.on_call = on_call
        self.calls: int = 0
        self.log: list[dict] = []

    def reset(self) -> None:
        self.calls = 0
        self.log = []

    def stage_context(self, stage: str):
        """Context manager for setting the active stage for LLM stage attribution."""
        return stage_context(stage)

    def complete(self, system: str, user: str, stage: str | None = None, **kwargs) -> str:
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
        entry = {
            "system_chars": len(system),
            "user_chars": len(user),
            "out_chars": len(out),
            "latency_s": latency,
        }
        inner_usage = getattr(self.inner, "last_usage", None)
        if inner_usage:
            entry["usage"] = inner_usage
        self.log.append(entry)

        if self.on_call is not None:
            call_stage = stage
            if not call_stage:
                inferred = infer_stage_from_prompt(system)
                if inferred != "unknown":
                    call_stage = inferred
                else:
                    ambient = get_current_stage()
                    call_stage = ambient if ambient != "unknown" else "unknown"
            self.on_call(call_stage, entry)

        return out


