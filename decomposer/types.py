from dataclasses import dataclass, field


@dataclass
class DecompositionResult:
    is_compound: bool
    sub_queries: list[str]
    method: str
    rejected_splits: list[str] = field(default_factory=list)
    reason: str = ""
