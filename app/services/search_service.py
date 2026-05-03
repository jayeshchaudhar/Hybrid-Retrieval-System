from __future__ import annotations
import logging
import time
from typing import Optional, Dict
from app.core.query_processor.processor import QueryProcessor
from app.core.router.router import QueryRouter
from app.core.retrievers.base import BaseRetriever
from app.core.retrievers.tfidf import TFIDFRetriever
from app.core.retrievers.bm25 import BM25Retriever
from app.core.retrievers.semantic import SemanticRetriever
from app.core.retrievers.hybrid import HybridRetriever
from app.core.retrievers.dense_qa import DenseQARetriever
from app.core.reranker.reranker import Reranker
from app.core.cache.cache import SearchCache
from app.models.schemas import SearchRequest, SearchResponse, Article

logger = logging.getLogger(__name__)

class SearchService:
    def __init__(self):
        self.processor = QueryProcessor()
        self.router = QueryRouter()
        self.reranker = Reranker()
        self.cache = SearchCache()
        self.retrievers: Dict[str, BaseRetriever] = {
            "tfidf":    TFIDFRetriever(),
            "bm25":     BM25Retriever(),
            "semantic": SemanticRetriever(),
            "dense_qa": DenseQARetriever(),
            "hybrid":   HybridRetriever(),
        }
        self._ready = False
    def build_indexes(self, articles: list[Article]) -> None:
        """Build all indexes. Call once after corpus is loaded."""
        for name, retriever in self.retrievers.items():
            logger.info("Building index: %s…", name)
            retriever.build_index(articles)
            retriever.save()
        self._ready = True
        logger.info("SearchService: all indexes built and saved.")

    def load_indexes(self) -> None:
        """Load pre-built indexes from disk."""
        for name, retriever in self.retrievers.items():
            try:
                retriever.load()
                logger.info("Loaded index: %s", name)
            except FileNotFoundError:
                logger.warning("Index not found for %s — run build_indexes first.", name)
        self._ready = True

    def search(self, req: SearchRequest) -> SearchResponse:
        t0 = time.perf_counter()
        cached = self.cache.get(req.query, req.method or "auto", req.top_k, req.sport_filter)
        if cached:
            return SearchResponse(
                query=req.query,
                results=cached,
                method_used="cached",
                router_confidence=1.0,
                latency_ms=(time.perf_counter() - t0) * 1000,
                cached=True,
            )

        pq = self.processor.process(req.query)
        decision = self.router.route(pq, force_method=req.method)
        retriever = self.retrievers[decision.method]
        results = retriever.retrieve(req.query, req.top_k * 2, req.sport_filter)
        if req.rerank:
            results = self.reranker.rerank(req.query, results)

        results = results[:req.top_k]

        self.cache.set(req.query, decision.method, req.top_k, req.sport_filter, results)

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug("search: method=%s conf=%.2f latency=%.1fms",
                     decision.method, decision.confidence, latency_ms)

        return SearchResponse(
            query=req.query,
            results=results,
            method_used=decision.method,
            router_confidence=decision.confidence,
            latency_ms=latency_ms,
        )
