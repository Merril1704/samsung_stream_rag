from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence
from telemetry.schema import TurnTrace


def write_event_log(traces: Sequence[TurnTrace] | TurnTrace, path: str | Path) -> None:
    """Append TurnTrace records as JSON Lines to the given file.

    Guarantees exactly one valid JSON object per turn line.
    """
    if isinstance(traces, TurnTrace):
        traces = [traces]

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace.to_dict(), default=str) + "\n")


def format_summary(trace: TurnTrace) -> str:
    """Produce a single-line human-readable summary for console/log output."""
    return (
        f"Turn {trace.turn_number} [{trace.path}] "
        f"duration={trace.duration_s:.3f}s "
        f"llm_calls={trace.total_llm_calls} "
        f"tokens_in={trace.total_estimated_input_tokens} "
        f"tokens_out={trace.total_estimated_output_tokens} "
        f"stages={len(trace.stage_events)}"
    )
