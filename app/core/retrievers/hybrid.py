# Hybrid Retriever
from __future__ import annotations
import logging
from typing import List, Optional, Dict
from collections import defaultdict
from app.core.retrievers.base import BaseRetriever
from app.core.retrievers.bm25 import BM25Retriever
from app.core.retrievers.semantic import SemanticRetriever
from app.models.schemas import Article, RetrievedDoc
from config.config import HYBRID_CFG, BM25_CFG, SEMANTIC_CFG

logger = logging.getLogger(__name__)

def reciprocal_rank_fusion(
        ranked_lists :  List[List[RetrievedDoc]],
        weights: List[float],
        k: int = 60,
) -> List[RetrievedDoc]:
    rrf_scores: Dict[str, float] = defaultdict(float)
    doc_meta :  Dict[str, RetrievedDoc] = {}

    for ranked_list, w in zip(ranked_lists, weights):
        for rank, doc in enumerate(ranked_list, start=1):
            rrf_scores[doc.article_id] += w/(k+rank)
            if doc.article_id not in doc_meta:
                doc_meta[doc.article_id] = doc
    fused = []

    for aid, score, in sorted(rrf_scores.items(), key = lambda x: -x[1]):
        d = doc_meta[aid].model_copy(update={"score": score, "retriever": "hybrid"})
        fused.append(d)
    return fused

class HybridRetriever(BaseRetriever):
    name = "hybrid"

    def __init__(self, cfg = HYBRID_CFG, bm25_cfg = BM25_CFG, semantic_cfg = SEMANTIC_CFG):
        super().__init__()
        self.cfg = cfg
        self.bm25 = BM25Retriever(bm25_cfg)
        self.semantic = SemanticRetriever(semantic_cfg)
    
    #index
    def build_index(self, articles: List[Article]) -> None:
        logger.info("Hybrid: Delegating index build to BM25 + Semantic...")
        self.bm25.build_index(articles)
        self.semantic.build_index(articles)
        self._indexed = True
    
    # Retrieval

    def retrieve(self, query: str, top_k: int = 10, sport_filter: Optional[str] = None) -> List[RetrievedDoc]:
        fetch_k = top_k*2 
        self.bm25_results = self.bm25.retrieve(query, fetch_k, sport_filter)
        sem_result = self.semantic.retrieve(query, fetch_k, sport_filter)

        fused = reciprocal_rank_fusion(
            [self.bm25_results, sem_result],
            weights=[self.cfg.bm25_weight, self.cfg.semantic_weight],
            k = self.cfg.rrf_k,
        )
        return fused[:top_k]
    
    #Persistence

    def save(self) -> None:
        self.bm25.save()
        self.semantic.save()

    def load(self) -> None:
        self.bm25.load()
        self.semantic.load()
        self._indexed = True

        

