from __future__ import annotations
import pickle
import logging
from typing import List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.core.retrievers.base import BaseRetriever
from app.models.schemas import Article, RetrievedDoc
from config.config import TFIDF_CFG

logger = logging.getLogger(__name__)

class TFIDFRetriever(BaseRetriever):
    name = "tfidf"

    def __init__(self, cfg = TFIDF_CFG):
        super().__init__()
        self.cfg = cfg
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None
        self.articles: List[Article] = []
    
    #index

    def build_index(self, articles: List[Article]) -> None:
        logger.info("TF-IDF: building index on %d articles...", len(articles))
        self.articles = articles
        self.vectorizer = TfidfVectorizer(
            max_features=self.cfg.max_features,
            ngram_range=self.cfg.ngram_range,
            sublinear_tf=self.cfg.sublinear_tf,
            strip_accents="unicode",
            analyzer="word",
            min_df=2,
        )

        corpus_texts = [a.full_text() for a in articles]
        self.matrix = self.vectorizer.fit_transform(corpus_texts)
        self._indexed = True
        logger.info("TF-IDF: index ready - vocab=%d", len(self.vectorizer.vocabulary_))

    
    #retrieval

    def retrieve(self,
                query: str,
                top_k: int = 10,
                sport_filter: Optional[str] = None,
                ) -> List[RetrievedDoc]:
        
        if not self._indexed:
            raise RuntimeError("Index not build. call build_index() or load() first.")
        
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).flatten()

        if sport_filter:
            mask = np.array([a.sport == sport_filter for a in self.articles])
            scores[~mask] = 0.0
       
        top_indices = np.argpartition(scores, -min(top_k, len(scores)))[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices]) [::-1]]

        results = []

        for idx in top_indices:
            if scores[idx] == 0.0:
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
    
    #persistence

    def save(self)-> None:
        payload = {
            "vectorizer": self.vectorizer,
            "matrix": self.matrix,
            "article_ids": [a.id for a in self.articles],
            "article.meta":[(a.title, a.sport, a.body[:300]) for a in self.articles],
        }

        self.cfg.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cfg.index_path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("TF-IDF: index saved to %s", self.cfg.index_path)

    def load(self) -> None:
        with open(self.cfg.index_path, "rb") as f:
            payload = pickle.load(f)
        self.vectorizer = payload["vectorizer"]
        self.matrix = payload["matrix"]
        # Reconstruct lightweight Article stubs for metadata
        self.articles = [
            Article(id=aid, title=meta[0], sport=meta[1], body=meta[2])
            for aid, meta in zip(payload["article_ids"], payload["article_meta"])
        ]
        self._indexed = True
        logger.info("TF-IDF: index loaded from %s", self.cfg.index_path)