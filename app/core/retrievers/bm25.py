from __future__ import annotations
import pickle
import logging
import textwrap
from typing import List, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from app.core.retrievers.base import BaseRetriever
from app.models.schemas import Article, RetrievedDoc
from config.config import BM25_CFG


logger = logging.getLogger(__name__)
def _tokenize(text: str) -> List[str]:
    return text.lower().split()


class BM25Retriever(BaseRetriever):
    name = "bm25"

    def __init__(self, cfg=BM25_CFG):
        super().__init__()
        self.cfg = cfg
        self.bm25: Optional[BM25Okapi] = None
        self.articles: List[Article] = []
        self.tokenized_corpus: List[List[str]] = []

    # ---------------- INDEX ----------------
    def build_index(self, articles: List[Article]) -> None:
        logger.info("BM25: building index on %d articles...", len(articles))
        self.articles = articles

        self.tokenized_corpus = [
            _tokenize(a.full_text()) for a in articles
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=self.cfg.k1,
            b=self.cfg.b,
        )

        self._indexed = True
        logger.info("BM25: index ready - avg_doc_len = %.1f", self.bm25.avgdl)

    # ---------------- PARAM UPDATE ----------------
    def set_params(self, k1: float, b: float):
        """Update BM25 parameters without re-tokenizing"""
        self.cfg.k1 = k1
        self.cfg.b = b

        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=k1,
            b=b,
        )
        logger.debug("BM25: param updated - k1 = %.2f b = %.2f", k1, b)

    # ---------------- RETRIEVE ----------------
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        sport_filter: Optional[str] = None,
    ) -> List[RetrievedDoc]:

        if not self._indexed:
            raise RuntimeError("Index not built.")

        q_tokens = _tokenize(query)
        scores = np.array(self.bm25.get_scores(q_tokens))

        if sport_filter:
            mask = np.array([a.sport == sport_filter for a in self.articles])
            scores[~mask] = 0.0

        top_k_actual = min(top_k, len(self.articles))
        top_indices = np.argpartition(scores, -top_k_actual)[-top_k_actual:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break

            a = self.articles[idx]

            results.append(RetrievedDoc(
                article_id=a.id,
                title=a.title,
                sport=a.sport,
                score=float(scores[idx]),
                retriever=self.name,
                snippet=self._make_snippet(a.body),
            ))

        return results
    @staticmethod
    def _make_snippet(text: str, max_chars : int = 200) -> str:
        return textwrap.shorten(text, width=max_chars, placeholder="...")

    # ---------------- SAVE ----------------
    def save(self) -> None:
        payload = {
            "bm25": self.bm25,
            "article_ids": [a.id for a in self.articles],
            "article_meta": [(a.title, a.sport, a.body[:300], a.source) for a in self.articles],
        }

        self.cfg.index_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.cfg.index_path, "wb") as f:
            pickle.dump(payload, f)

        logger.info("BM25: index saved to %s", self.cfg.index_path)

    # ---------------- LOAD ----------------
    def load(self) -> None:
        with open(self.cfg.index_path, "rb") as f:
            payload = pickle.load(f)

        self.bm25 = payload["bm25"]

        self.articles = [
            Article(id=aid, title=meta[0], sport=meta[1], body=meta[2], source=meta[3])
            for aid, meta in zip(payload["article_ids"], payload["article_meta"])
        ]

        self._indexed = True
        logger.info("BM25: index loaded from %s", self.cfg.index_path)