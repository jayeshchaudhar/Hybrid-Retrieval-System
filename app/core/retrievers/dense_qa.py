from __future__ import annotations
import json
import logging 
import numpy as np
from typing import List, Optional
from app.core.retrievers.base import BaseRetriever
from app.models.schemas import Article, RetrievedDoc
from config.config import COLBERT_CFG

logger = logging.getLogger(__name__)

class DenseQARetriever(BaseRetriever):

    name = "dense_qa"

    def __init__(self, cfg=COLBERT_CFG):
        super().__init__()
        self.cfg = cfg
        self.model = None
        self.index = None
        self.article_ids : List[str] = []
        self.article_meta: dict = {}

    def _load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("DenseQA: loading model %s...", self.cfg.model_name)
            self.model = SentenceTransformer(self.cfg.model_name)
        
    #index

    def build_index(self, articles: List[Article]) -> None:
        self._load_model()
        import faiss
        logger.info("DenseQA: encoding %d articles…", len(articles))
        texts = [a.full_text() for a in articles]
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
 
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.article_ids = [a.id for a in articles]
        self.article_meta = {
            a.id: {"title": a.title, "sport": a.sport, "snippet": a.body[:300], "source": a.source}
            for a in articles
        }

        self._indexed = True
        logger.info("DenseQA: index ready - %d vectors", self.index.ntotal)

    #retrieval

    def retrieve(self, query: str, top_k: int  = 10, sport_filter: Optional[str]  = None) -> List[RetrievedDoc]:

        if not self._indexed:
            raise RuntimeError("Index not build.")
        self._load_model()
        
        fetch_k = top_k * 5 if sport_filter else top_k
        q_emb = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
 
        scores, indices = self.index.search(q_emb, min(fetch_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            aid = self.article_ids[idx]
            meta = self.article_meta[aid]
            if sport_filter and meta["sport"] != sport_filter:
                continue
            results.append(RetrievedDoc(
                article_id=aid,
                title=meta["title"],
                sport=meta["sport"],
                score=float(score),
                retriever=self.name,
                snippet=self._make_snippet(meta["snippet"]),
            ))
            if len(results) >= top_k:
                break
        return results
 
    #percistance
 
    def save(self) -> None:
        import faiss
        self.cfg.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.cfg.index_path))
        with open(self.cfg.ids_path, "w", encoding="utf-8") as f:
            json.dump({"ids": self.article_ids, "meta": self.article_meta}, f)
        logger.info("DenseQA: index saved")
 
    def load(self) -> None:
        import faiss
        self.index = faiss.read_index(str(self.cfg.index_path))
        with open(self.cfg.ids_path, encoding="utf-8") as f:
            data = json.load(f)
        self.article_ids = data["ids"]
        self.article_meta = data["meta"]
        self._indexed = True
        logger.info("DenseQA: index loaded")