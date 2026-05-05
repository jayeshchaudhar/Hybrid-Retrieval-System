from __future__ import annotations
import math
from typing import List, Dict


def dcg(gains: List[float], k: int) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg(retrieved_ids: List[str], relevance_map: Dict[str, int], k: int) -> float:
    gains = [relevance_map.get(aid, 0) for aid in retrieved_ids[:k]]
    actual_dcg = dcg(gains, k)

    ideal_gains = sorted(relevance_map.values(), reverse=True)
    ideal_dcg = dcg(ideal_gains, k)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def average_precision(retrieved_ids: List[str], relevant_set: set, k: int) -> float:
    hits = 0
    precision_sum = 0.0
    for rank, aid in enumerate(retrieved_ids[:k], start=1):
        if aid in relevant_set:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(len(relevant_set), k) if relevant_set else 0.0


def reciprocal_rank(retrieved_ids: List[str], relevant_set: set) -> float:
    for rank, aid in enumerate(retrieved_ids, start=1):
        if aid in relevant_set:
            return 1.0 / rank
    return 0.0


def precision_at_k(retrieved_ids: List[str], relevant_set: set, k: int) -> float:
    hits = sum(1 for aid in retrieved_ids[:k] if aid in relevant_set)
    return hits / k


def recall_at_k(retrieved_ids: List[str], relevant_set: set, k: int) -> float:
    if not relevant_set:
        return 0.0
    hits = sum(1 for aid in retrieved_ids[:k] if aid in relevant_set)
    return hits / len(relevant_set)


def compute_all(
    retrieved_ids: List[str],
    relevance_map: Dict[str, int],
) -> Dict[str, float]:
    relevant_set = {aid for aid, grade in relevance_map.items() if grade >= 1}
    return {
        "ndcg@5":       ndcg(retrieved_ids, relevance_map, k=5),
        "ndcg@10":      ndcg(retrieved_ids, relevance_map, k=10),
        "map@10":       average_precision(retrieved_ids, relevant_set, k=10),
        "mrr":          reciprocal_rank(retrieved_ids, relevant_set),
        "precision@5":  precision_at_k(retrieved_ids, relevant_set, k=5),
        "recall@10":    recall_at_k(retrieved_ids, relevant_set, k=10),
    }
