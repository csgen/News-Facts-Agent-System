"""Retrieval reranking: Reciprocal Rank Fusion.

Merges vector DB results and graph entity-claim results into one ranked list
without needing score normalisation.
"""

import logging

logger = logging.getLogger(__name__)

_RRF_K = 60  # standard constant — dampens high rank outliers


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    id_key: str = "claim_id",
) -> list[dict]:
    """Merge multiple ranked lists into one using Reciprocal Rank Fusion.

    Each list must be ordered best-first. Returns a deduplicated list ordered
    by descending RRF score. The original item dict is preserved; an rrf_score
    key is added.
    """
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            key = item[id_key]
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
            if key not in items:
                items[key] = item

    merged = sorted(items.values(), key=lambda x: scores[x[id_key]], reverse=True)
    for item in merged:
        item["rrf_score"] = scores[item[id_key]]
    return merged


def rerank_candidates(
    query: str,
    vector_results: list[dict],
    graph_results: list[dict],
    top_k: int,
) -> list[dict]:
    """Rerank pipeline: RRF merge of vector and graph results.

    Args:
        vector_results: Ranked list from ChromaDB similarity search.
        graph_results:  Ranked list from Neo4j entity-claim traversal.
        top_k: Maximum number of results to return.

    Returns:
        Deduplicated, reranked list of claim dicts.
    """
    lists_to_merge = [i for i in [vector_results, graph_results] if i]

    if not lists_to_merge:
        return []

    if len(lists_to_merge) == 1:
        return lists_to_merge[0][:top_k]

    merged = reciprocal_rank_fusion(lists_to_merge)
    logger.info(
        "RRF merged %d vector + %d graph → %d unique candidates",
        len(vector_results),
        len(graph_results),
        len(merged),
    )
    return merged[:top_k]
