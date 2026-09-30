from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from telemetry.schema import TurnTrace

LLMCallHook = Callable[[str, dict], None]

_active_stage_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "telemetry_active_stage",
    default="unknown",
)


@contextmanager
def stage_context(stage: str):
    """Scoped context manager for setting the active stage for LLM stage attribution."""
    token = _active_stage_var.set(stage)
    try:
        yield
    finally:
        _active_stage_var.reset(token)


def get_current_stage() -> str:
    """Return the currently active stage from the ambient context."""
    return _active_stage_var.get()


def infer_stage_from_prompt(system_prompt: str) -> str:
    """Infer the pipeline stage from known system prompt contents."""
    s = (system_prompt or "").lower()
    if "you generate factual claims" in s:
        return "generator"
    if "genuinely contradict each other" in s:
        return "contradiction"
    if "splitting an answer into atomic factual claims" in s or "atomic factual claims" in s:
        return "decomposer"
    if "checking whether a specific claim is actually supported" in s or "entails" in s:
        return "verifier"
    if "evaluating the quality of a query decomposition" in s:
        return "verifier"
    if "retrieval-timing controller" in s:
        return "controller"
    return "unknown"


def make_llm_hook(trace: TurnTrace) -> LLMCallHook:
    """Return a provider-agnostic callback that records LLM calls into the given trace."""
    def hook(stage: str, log_entry: dict) -> None:
        trace.record_llm_call(stage=stage, log_entry=log_entry)
    return hook
