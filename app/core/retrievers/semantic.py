from __future__ import annotations
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional
from app.core.retrievers.base import BaseRetriever
from app.models.schemas import Article, RetrievedDoc
from config.config import SEMANTIC_CFG
 
logger = logging.getLogger(__name__)

class SemanaticRetriever(BaseRetriever):
    name = "semantic"

    def __init__(self, cfg=SEMANTIC_CFG):
        super().__init__()
        self.cfg = cfg
        self.model = None
        self.index = None
        self.article_ids: List[str] = []
        self.article_meta: dict= {}

    
    def _load_model(self):
        if self.model in None:
            from sentence_transformers import SentenceTransformer
            logger.info("Semantic: loading model %s...",self.cfg.model_name)
            self.model = SentenceTransformer(self.cfg.model_name)

    def 
                         