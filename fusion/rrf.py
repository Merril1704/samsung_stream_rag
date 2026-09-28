"""Reciprocal Rank Fusion (RRF).

Implements Cormack et al. (2009) reciprocal rank fusion over multiple ranked
lists of chunk identifiers. Pure function without I/O or state.
"""


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],   # each inner list = chunk_ids in rank order, from one retriever/source
    k: int = 60,                    # literature-standard default (Cormack et al. 2009); do not hand-tune without eval data
) -> list[tuple[str, float]]:
    """
    Returns [(chunk_id, fused_score), ...] sorted descending by fused_score.
    fused_score for a chunk = sum over every ranked_list it appears in of 1/(k + rank),
    where rank is 0-indexed position in that list.
    A chunk absent from a given list contributes 0 from that list.
    """
    scores: dict[str, float] = {}

    for r_list in ranked_lists:
        for rank, chunk_id in enumerate(r_list):
            if chunk_id not in scores:
                scores[chunk_id] = 0.0
            scores[chunk_id] += 1.0 / (k + rank)

    # Sort descending by fused_score; stable tie-breaking
    sorted_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_items
