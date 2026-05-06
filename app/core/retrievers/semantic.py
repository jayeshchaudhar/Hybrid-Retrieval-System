from __future__ import annotations
import json
import logging
import numpy as np
import os
import tempfile
import textwrap
from pathlib import Path
from typing import List, Optional
from app.core.retrievers.base import BaseRetriever
from app.models.schemas import Article, RetrievedDoc
from config.config import SEMANTIC_CFG
 
logger = logging.getLogger(__name__)

class SemanticRetriever(BaseRetriever):
    name = "semantic"

    def __init__(self, cfg=SEMANTIC_CFG):
        super().__init__()
        self.cfg = cfg
        self.model = None
        self.index = None
        self.article_ids: List[str] = []
        self.article_meta: dict= {}
        self._emb_cache : dict[str, np.ndarray] = {}
        self._MAX_CACHE: int = 50_000

    
    def _load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Semantic: loading model %s...",self.cfg.model_name)
            self.model = SentenceTransformer(self.cfg.model_name)

    #FAISS index builder
    def _build_faiss(self, embeddings: np.ndarray):
        import faiss
        dim = embeddings.shape[1]
        if len(embeddings) < 1_000_000:
            index = faiss.IndexFlatL2(dim)
        else:
            nlist = min(4096, len(embeddings) // 39)
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)

        index.add(embeddings)
        self.index = index
        return index
    
    #index

    def build_index(self,
                    articles: List[Article]) -> None:
        self._load_model()
        logger.info("Semantic: encoding %d articles...", len(articles))
        texts = [a.full_text() for a in articles]
        embeddings = self.model.encode(
            texts,
            batch_size  = self.cfg.batch_size,
            show_progress_bar= True,
            normalize_embeddings= True,
            convert_to_numpy=True,
        ).astype(np.float32)

        self.article_ids = [a.id for a in articles]
        self.article_meta = {
            a.id: {"title": a.title, "sport": a.sport, "snippet": a.body[:300], "source": a.source}
            for a in articles
            }

        self._indexed = self._build_faiss(embeddings)
        self._indexed = True
        logger.info("Semantic: FAISS index build - %d vectors", self.index.ntotal)

    
    # query embedding with LRU cache

    def _get_query_embedding(self, query: str) -> np.ndarray:
        key = query.lower().strip()
        if key in self._emb_cache:
            return self._emb_cache[key].reshape(1,-1)
        
        self._load_model()
        emb = self.model.encode(
            [query],
            normalize_embeddings = True,
            convert_to_numpy = True,
        ).astype(np.float32)

        if len(self._emb_cache) >= self._MAX_CACHE:
            oldest_key = next(iter(self._emb_cache))
            del self._emb_cache[oldest_key]

        self._emb_cache[key] = emb[0]
        return emb
    
    # Retrieve
    def retrieve(self, query: str, top_k: int  = 10, sport_filter: Optional[str] = None)-> List[RetrievedDoc]:
        if not self._indexed:
            raise RuntimeError("Index not built.")
        self._load_model()

        fetch_k = top_k * 5 if sport_filter else top_k
        q_emb = self._get_query_embedding(query)
        
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

    @staticmethod
    def _make_snippet(text, max_chars: int = 200) -> str:
        return textwrap.shorten(text, width=max_chars, placeholder="...")
    
    def save(self) -> None:
        import faiss
        self.cfg.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_index = self.cfg.faiss_index_path.with_suffix(".tmp")
        faiss.write_index(self.index, str(tmp_index))

        os.replace(tmp_index, self.cfg.faiss_index_path)
        logger.info("Semantic: FAISS index saved automatically -> %s",
                    self.cfg.faiss_index_path )
        tmp_meta = self.cfg.ids_path.with_suffix(".tmp")
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump({"ids": self.article_ids, "meta": self.article_meta}, f)
        os.replace(tmp_meta, self.cfg.ids_path)
        logger.info("Semantic: index saved automatically -> %s", self.cfg.ids_path)
    
    #load
    def load(self) -> None:
        import faiss
        self.index = faiss.read_index(str(self.cfg.faiss_index_path))
        with open(self.cfg.ids_path, encoding="utf-8") as f:
            data = json.load(f)
        
        self.article_ids = data["ids"]
        self.article_meta = data["meta"]
        self._indexed = True
        logger.info("Semantic: index loaded - %d vectors", self.index.ntotal)

        



                         