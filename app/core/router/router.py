
"""
Confidence-based query router.
entity queries (high NE density) - BM25  (exact matching wins)
factual / NL questions - DenseQA (QA-tuned model)
purely semantic / vague -  Semantic
mixed / ambiguous - Hybrid  (RRF)
low confidence in all signals -  fallback (configurable, default: Hybrid)
"""
from __future__ import annotations
import logging 
from typing import Dict, Optional
from dataclasses import dataclass
from app.core.query_processor.processor import ProcessedQuery
from config.config import ROUTER_CFG, RouterConfig

logger = logging.getLogger(__name__)

@dataclass
class RoutingDecision:
    method: str
    confidence: float
    reason: str


class QueryRouter:

    def __init__(self, cfg: RouterConfig = ROUTER_CFG):
        self.cfg = cfg

    def route(self, pq: ProcessedQuery, force_method: Optional[str] = None) -> RoutingDecision:
        if force_method:
            return RoutingDecision(
                method=force_method,
                confidence=1.0,
                reason="forced by caller",
            )

        scores: Dict[str, float] = self._score_methods(pq)
        best_method = max(scores, key=lambda m: scores[m])
        best_score = scores[best_method]

        # Fall-through if confidence is too low, use fallback
        if best_score < self.cfg.semantic_fallback_threshold:
            return RoutingDecision(
                method=self.cfg.fallback_method,
                confidence=best_score,
                reason=f"all scores below threshold → fallback({self.cfg.fallback_method})",
            )

        return RoutingDecision(
            method=best_method,
            confidence=best_score,
            reason=f"query_type={pq.query_type}, entity_density={pq.entity_density:.2f}",
        )

    def _score_methods(self, pq: ProcessedQuery) -> Dict[str, float]:
        """
        Heuristic scoring — replace with a trained classifier once you have
        enough labelled routing examples from the eval harness.
        """
        d = pq.entity_density
        is_q = pq.is_question
        qtype = pq.query_type

        return {
            "bm25": (
                0.85 if qtype == "entity"
                else 0.55 if qtype == "factual"
                else 0.30
            ),
            "tfidf": (
                0.65 if qtype == "entity"
                else 0.40
            ),
            "semantic": (
                0.85 if qtype == "semantic"
                else 0.50 if is_q
                else 0.35
            ),
            "dense_qa": (
                0.90 if qtype == "factual"
                else 0.60 if is_q
                else 0.25
            ),
            "hybrid": (
                0.80 if qtype == "hybrid"
                else 0.55 if d > 0.1
                else 0.45
            ),
        }

