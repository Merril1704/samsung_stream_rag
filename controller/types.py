from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional, Protocol

Action = Literal["WAIT", "RETRIEVE", "SUPPRESS"]
Trigger = Literal["provisional", "final", "multi_intent", "suppression", "none"]


@dataclass
class RetrievalDecision:
    action: Action
    trigger: Trigger
    confidence: float
    reason: str
    possible_multi_intent: bool = False


@dataclass
class SessionState:
    """Session-bound state only (hard constraint D: no cross-session leakage).
    A fresh SessionState must be constructed per independent test case."""
    session_id: str
    transcript_so_far: str = ""
    chunks_seen: list[str] = field(default_factory=list)
    last_retrieved_entity_hash: Optional[str] = None
    last_answer_topic: Optional[str] = None  # set by the synthesis stage after an answer is produced
    decision_log: list[dict] = field(default_factory=list)

    def append_chunk(self, chunk: str) -> None:
        if chunk:
            self.chunks_seen.append(chunk)
            self.transcript_so_far = (self.transcript_so_far + " " + chunk).strip()


class Controller(Protocol):
    mode: str

    def decide(self, state: SessionState, new_chunk: str, timestamp_s: float) -> RetrievalDecision:
        ...
