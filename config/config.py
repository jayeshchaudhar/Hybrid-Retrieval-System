import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CORPUS_DIR = DATA_DIR / "corpus"
QUERIES_DIR = DATA_DIR / "queries"
PROCESSED_DIR = DATA_DIR / "processed"
INDEXES_DIR = BASE_DIR / "indexes"
EVALUATION_DIR = BASE_DIR / "evaluation"
LOGS_DIR = BASE_DIR / "logs"

for d in [CORPUS_DIR, QUERIES_DIR, PROCESSED_DIR, INDEXES_DIR, EVALUATION_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok= True)

CORPUS_FILE = CORPUS_DIR / "articles.json"
TARGET_ARTICLES = 500
SPORTS = [
    "football", "cricket", "basketball", "tennis", "athletics",
    "swimming", "cycling", "boxing", "golf", "rugby",
    "baseball", "hockey", "volleyball", "badminton", "table_tennis",
]

ARTICLES_PER_SPORT = TARGET_ARTICLES // len(SPORTS)  


# retrival

@dataclass
class TFIDFConfig:
    max_features: int = 50_000
    ngram_range: tuple = (1,2)
    sublinear_tf: bool = True
    top_k :  int = 10
    index_path : Path = INDEXES_DIR/ "tfidf_index.pkl"


@dataclass
class BM25Config:
    k1: float = 1.5
    b: float = 0.75
    top_k: int = 10
    index_path: Path = INDEXES_DIR/ "bm25_index.pkl"

@dataclass
class SemanticConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size : int = 64
    top_k: int = 10
    embeddings_path: Path = INDEXES_DIR / "embeddings.npy"
    ids_path: Path =INDEXES_DIR/ "embedding_ids.json"
    # faiss index
    faiss_index_path: Path = INDEXES_DIR/ "faiss.index"
    use_gpu: bool = False

@dataclass
class HybridConfig:
    bm25_weight: float = 0.5
    semantic_weight: float = 0.5
    rrf_k : int = 60
    top_k = 10

@dataclass
class DensecolBertConfig:
    model_name : str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    top_k : int = 10
    index_path:Path =  INDEXES_DIR/ "colbert_embeddings.npy"
    ids_path : Path = INDEXES_DIR/ "colbert_ids.json"

# router
@dataclass

class RouterConfig:
    entity_query_threshold :  float = 0.6
    semantic_fallback_threshold : float = 0.4
    hybrid_threshold : float = 0.5
    fallback_method: str = "hybrid"

# re-ranker

@dataclass
class RerankerConfig:
    model_name: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2"  # 4x faster
    top_k: int = 20
    final_top_k: int = 10
    enabled: bool = True

# cache
@dataclass
class CacheConfig:
    backend: str = "memory"
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    ttl_seconds: int = 3600
    max_memory_items: int = 10_000

# evaluation
@dataclass
class EvalConfig:
    queries_files: Path = QUERIES_DIR/ "queries.jsonl"
    result_file: Path = EVALUATION_DIR/ "result.json"
    train_ratio: float = 0.6
    dev_ratio: float = 0.2
    test_ratio: float = 0.2
    relevance_levels: int = 4
    metrics : List[str] = field(default_factory=lambda: [
        "ndcg@5", "ndcg@10", "map@10", "mrr", "precision@5", "recall@10" 
    ])
    latency_percentiles :  List[int] = field(default_factory=lambda: [50, 95])


# API

@dataclass
class APIConfig:
    host : str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", 8000))
    workers: int = int(os.getenv("WORKERS", 4))
    log_level: str = os.getenv("LOG_LEVEL", "info")
    rate_limit: int = 100
    enable_docs: bool = True

# singleton instances

TFIDF_CFG = TFIDFConfig()
BM25_CFG = BM25Config()
SEMANTIC_CFG = SemanticConfig()
HYBRID_CFG  = HybridConfig()
COLBERT_CFG = DensecolBertConfig()
ROUTER_CFG = RouterConfig()
RERANKER_CFG = RerankerConfig()
CACHE_CFG = CacheConfig()
EVAL_CFG = EvalConfig()
API_CFG = APIConfig()

