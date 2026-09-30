from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import time
from typing import Any

CHARS_PER_TOKEN = 3.5  # conservative estimate for English text


def estimate_tokens(char_count: int) -> int:
    """Estimate token count from character count."""
    if char_count <= 0:
        return 0
    return max(1, int(char_count / CHARS_PER_TOKEN))


@dataclass
class StageEvent:
    """Timing record for a single pipeline stage within a turn."""
    stage: str                    # "retrieval", "fusion", "generator", "verifier", "render", etc.
    wall_start: float             # time.time() - wall-clock timestamp
    wall_end: float               # time.time() - wall-clock timestamp
    duration_s: float             # time.perf_counter() delta (monotonic duration)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class LLMCallEvent:
    """Record of a single LLM provider call."""
    stage: str                    # which pipeline stage triggered this call
    wall_timestamp: float         # time.time() at call completion
    system_chars: int
    user_chars: int
    out_chars: int
    latency_s: float              # perf_counter delta
    estimated_input_tokens: int   # computed from chars
    estimated_output_tokens: int  # computed from chars

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class RetrievalTriggerEvent:
    """Record of a single retrieval operation or controller decision."""
    wall_timestamp: float         # time.time()
    transcript_timestamp_s: float = 0.0 # stream time from controller.decide()
    action: str = "RETRIEVE"      # "RETRIEVE" | "WAIT" | "SUPPRESS"
    trigger: str = "none"         # "provisional" | "final" | "multi_intent" | "suppression" | "none"
    confidence: float = 1.0
    query: str = ""
    reason: str = ""
    chunk_ids_returned: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class VersionTransitionEvent:
    """Record of an answer version change."""
    entry_id: str
    old_version: int
    new_version: int
    claims_added: list[str] = field(default_factory=list)
    claims_retired: list[str] = field(default_factory=list)
    cited_chunk_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class _StageTimer:
    """Context manager that records a StageEvent on exit and manages scoped stage context."""
    def __init__(self, trace: TurnTrace, stage: str, metadata: dict):
        self._trace = trace
        self._stage = stage
        self._metadata = metadata
        self._wall_start = 0.0
        self._perf_start = 0.0
        self._ctx = None

    def __enter__(self):
        from telemetry.hooks import stage_context
        self._wall_start = time.time()
        self._perf_start = time.perf_counter()
        self._ctx = stage_context(self._stage)
        self._ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self._ctx is not None:
                self._ctx.__exit__(exc_type, exc_val, exc_tb)
        finally:
            wall_end = time.time()
            duration = time.perf_counter() - self._perf_start
            self._trace.stage_events.append(
                StageEvent(
                    stage=self._stage,
                    wall_start=self._wall_start,
                    wall_end=wall_end,
                    duration_s=duration,
                    metadata=self._metadata,
                )
            )
        return False


@dataclass
class TurnTrace:
    """Collects all telemetry events for a single pipeline turn."""
    turn_number: int
    session_id: str
    path: str = ""                # "NEW_TOPIC" | "REFINEMENT", set at finalize

    wall_start: float = 0.0
    wall_end: float = 0.0
    duration_s: float = 0.0

    retrieval_events: list[RetrievalTriggerEvent] = field(default_factory=list)
    stage_events: list[StageEvent] = field(default_factory=list)
    llm_calls: list[LLMCallEvent] = field(default_factory=list)
    version_transition: VersionTransitionEvent | None = None

    # Aggregates (computed at finalize)
    total_llm_calls: int = 0
    total_estimated_input_tokens: int = 0
    total_estimated_output_tokens: int = 0
    total_llm_latency_s: float = 0.0

    _perf_start: float = field(default=0.0, repr=False)

    def begin(self) -> None:
        self.wall_start = time.time()
        self._perf_start = time.perf_counter()

    def start_stage(self, stage: str, **metadata) -> _StageTimer:
        return _StageTimer(self, stage, metadata)

    def record_retrieval(
        self,
        action: str = "RETRIEVE",
        trigger: str = "none",
        confidence: float = 1.0,
        query: str = "",
        reason: str = "",
        transcript_timestamp_s: float = 0.0,
        chunk_ids_returned: list[str] | None = None,
        wall_timestamp: float | None = None,
    ) -> RetrievalTriggerEvent:
        event = RetrievalTriggerEvent(
            wall_timestamp=wall_timestamp if wall_timestamp is not None else time.time(),
            transcript_timestamp_s=transcript_timestamp_s,
            action=action,
            trigger=trigger,
            confidence=confidence,
            query=query,
            reason=reason,
            chunk_ids_returned=list(chunk_ids_returned or []),
        )
        self.retrieval_events.append(event)
        return event

    def record_llm_call(
        self,
        stage: str,
        log_entry: dict,
        wall_timestamp: float | None = None,
    ) -> LLMCallEvent:
        sys_chars = int(log_entry.get("system_chars", 0))
        usr_chars = int(log_entry.get("user_chars", 0))
        out_chars = int(log_entry.get("out_chars", 0))
        latency = float(log_entry.get("latency_s", 0.0))

        event = LLMCallEvent(
            stage=stage,
            wall_timestamp=wall_timestamp if wall_timestamp is not None else time.time(),
            system_chars=sys_chars,
            user_chars=usr_chars,
            out_chars=out_chars,
            latency_s=latency,
            estimated_input_tokens=estimate_tokens(sys_chars + usr_chars),
            estimated_output_tokens=estimate_tokens(out_chars),
        )
        self.llm_calls.append(event)
        return event

    def record_version_transition(
        self,
        entry_id: str,
        old_version: int,
        new_version: int,
        claims_added: list[str] | None = None,
        claims_retired: list[str] | None = None,
        cited_chunk_ids: list[str] | None = None,
    ) -> VersionTransitionEvent:
        event = VersionTransitionEvent(
            entry_id=entry_id,
            old_version=old_version,
            new_version=new_version,
            claims_added=list(claims_added or []),
            claims_retired=list(claims_retired or []),
            cited_chunk_ids=list(cited_chunk_ids or []),
        )
        self.version_transition = event
        return event

    def finalize(self, path: str = "") -> None:
        if path:
            self.path = path
        self.wall_end = time.time()
        if self._perf_start > 0.0:
            self.duration_s = time.perf_counter() - self._perf_start
        self.total_llm_calls = len(self.llm_calls)
        self.total_estimated_input_tokens = sum(c.estimated_input_tokens for c in self.llm_calls)
        self.total_estimated_output_tokens = sum(c.estimated_output_tokens for c in self.llm_calls)
        self.total_llm_latency_s = sum(c.latency_s for c in self.llm_calls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_number": self.turn_number,
            "session_id": self.session_id,
            "path": self.path,
            "wall_start": self.wall_start,
            "wall_end": self.wall_end,
            "duration_s": self.duration_s,
            "retrieval_events": [e.to_dict() for e in self.retrieval_events],
            "stage_events": [e.to_dict() for e in self.stage_events],
            "llm_calls": [e.to_dict() for e in self.llm_calls],
            "version_transition": self.version_transition.to_dict() if self.version_transition else None,
            "total_llm_calls": self.total_llm_calls,
            "total_estimated_input_tokens": self.total_estimated_input_tokens,
            "total_estimated_output_tokens": self.total_estimated_output_tokens,
            "total_llm_latency_s": self.total_llm_latency_s,
        }
