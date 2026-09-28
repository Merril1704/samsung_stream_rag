from __future__ import annotations
from session.ledger import LedgerEntry, active_claims


def render(entry: LedgerEntry) -> str:
    """
    Renders active claims deterministically without using an LLM.
    Groups claims by origin_version:
      - version 1 under 'Baseline'
      - later versions under 'Refinement: <detail>'
    Each active claim is followed by ' [chunk_id]'.
    """
    claims = active_claims(entry)
    if not claims:
        return ""

    versions: list[int] = []
    by_version: dict[int, list] = {}
    for c in claims:
        if c.origin_version not in by_version:
            versions.append(c.origin_version)
            by_version[c.origin_version] = []
        by_version[c.origin_version].append(c)

    sections: list[str] = []
    for v in sorted(versions):
        if v == 1:
            header = "Baseline"
        else:
            detail_idx = v - 2
            detail_str = entry.details[detail_idx] if (entry.details and 0 <= detail_idx < len(entry.details)) else f"v{v}"
            header = f"Refinement: {detail_str}"

        lines = [header]
        for c in by_version[v]:
            lines.append(f"- {c.claim} [{c.chunk_id}]")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)
