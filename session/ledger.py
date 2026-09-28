from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field
from typing import Literal
from retrieval.types import EvidenceChunk


@dataclass
class LedgerClaim:
    claim: str
    chunk_id: str
    origin_version: int
    status: Literal["ACTIVE", "RETIRED"] = "ACTIVE"


@dataclass
class LedgerEntry:
    entry_id: str
    topic: str
    details: list[str]
    claims: list[LedgerClaim]
    evidence: dict[str, EvidenceChunk]
    version: int = 1


@dataclass
class AnswerLedger:
    entries: dict[str, LedgerEntry] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)


def active_claims(entry: LedgerEntry) -> list[LedgerClaim]:
    return [c for c in entry.claims if c.status == "ACTIVE"]


def commit_entry(
    ledger: AnswerLedger,
    entry_id: str,
    topic: str,
    details: list[str],
    claims: list[LedgerClaim],
    evidence: dict[str, EvidenceChunk],
    version: int = 1,
    turn: int = 1,
    action: str = "NEW_TOPIC",
    retired: list[str] | None = None,
) -> LedgerEntry:
    evidence_copies = {k: dataclasses.replace(v) for k, v in evidence.items()}
    entry = LedgerEntry(
        entry_id=entry_id,
        topic=topic,
        details=details,
        claims=claims,
        evidence=evidence_copies,
        version=version,
    )
    ledger.entries[entry_id] = entry
    retired_list = retired or []
    ledger.history.append({
        "turn": turn,
        "action": action,
        "added": [c.claim for c in claims],
        "retired": retired_list,
    })
    return entry
