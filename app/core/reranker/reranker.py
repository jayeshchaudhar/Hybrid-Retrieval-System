#Cross-encoder re-ranker — takes top-K candidates from first-stage retrieval and re-scores them with a cross-encoder
from __future__ import annotations
import logging
import textwrap
from typing import List
from app.models.schemas import RetrievedDoc
from config.config import RERANKER_CFG, RerankerConfig

logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self, cfg: RerankerConfig = RERANKER_CFG):
        self.cfg = cfg
        self._model = None

    def _load(self):
        if self._model is None and self.cfg.enabled:
            from sentence_transformers import CrossEncoder
            logger.info("Reranker: loading %s…", self.cfg.model_name)
            self._model = CrossEncoder(self.cfg.model_name, max_length=512)
            logger.info("Reranker: model loaded and ready")
        elif not self.cfg.enabled:
            logger.info("Reanker: disabled in config-skipping load")

    def rerank(self, query: str, docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
        if not self.cfg.enabled or not docs:
            return docs

        if self._model is None:
            self._load()

        candidates = docs[:self.cfg.top_k]
        pairs = [(query, f"{d.title}. {self._truncate(d.snippet)}")
                 for d in candidates]
        
        scores = self._model.predict(pairs)

        reranked = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )
        out = []
        for score, doc in reranked[: self.cfg.final_top_k]:
            out.append(doc.model_copy(update={"score": float(score), "retriever": doc.retriever + "+rerank"}))
        return out
    
    @staticmethod
    def _truncate(text: str, max_chars: int = 300) -> str:
        return textwrap.shorten(text, width=max_chars, placeholder="...")
    
    @property
    def is_ready(self) -> bool:
        return self._model is not None
