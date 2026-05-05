from __future__ import annotations
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.schemas import Article
from app.core.retrievers.tfidf import TFIDFRetriever
from app.core.retrievers.bm25 import BM25Retriever
from app.core.retrievers.semantic import SemanticRetriever
from app.core.retrievers.dense_qa import DenseQARetriever
from app.core.retrievers.hybrid import HybridRetriever
from config.config import PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    corpus_path = PROCESSED_DIR / "articles_processed.json"
    if not corpus_path.exists():
        logger.error("Processed corpus not found at %s. Run preprocess.py first.", corpus_path)
        sys.exit(1)

    with open(corpus_path) as f:
        raw = json.load(f)
    articles = [Article(**item) for item in raw]
    logger.info("Loaded %d articles for indexing", len(articles))

    retrievers = [
        ("TF-IDF",    TFIDFRetriever()),
        ("BM25",      BM25Retriever()),
        ("Semantic",  SemanticRetriever()),
        ("DenseQA",   DenseQARetriever()),
    ]

    for name, retriever in retrievers:
        t0 = time.perf_counter()
        logger.info("Building %s index…", name)
        retriever.build_index(articles)
        retriever.save()
        elapsed = time.perf_counter() - t0
        logger.info("✓ %s — %.1fs", name, elapsed)

    # Hybrid reuses BM25 + Semantic
    logger.info("Hybrid retriever reuses BM25 + Semantic indexes — no extra build needed.")
    logger.info("All indexes built. Run run_evaluation.py next.")


if __name__ == "__main__":
    main()
