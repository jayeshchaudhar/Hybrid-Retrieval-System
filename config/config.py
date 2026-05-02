import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

basedir = Path(__file__).resolve().parent.parent
datadir = basedir
corpusdir = datadir
querydir = datadir
processdir = datadir
indexsdir = basedir
evalutiondir = basedir
logsdir = basedir

for d in [corpusdir, querydir, processdir, indexsdir, evalutiondir, logsdir]:
    d.mkdir(parents=True, exist_ok= True)

corpusfile = corpusdir/ "articles.json"
target_articles = 500
sports = [
    "football", "cricket", "basketball", "tennies", "athletics", 
    "swimming", "cycling", "boxing", "golf", "rugby",
    "baseball", "hockey", "volleyball", "badminton", "table_tennies",
]

articles_per_sport = target_articles//len(sports)

# retrival

@dataclass
class TFIDFConfig:
    max_features: int = 50_000
    ngram_range: tuple = (1,2)
    sublinear_tf: bool = True
    top_k :  int = 10
    index_path : Path = indexsdir/ "tfidf_index.pkl"


@dataclass
class BM25Config:
    k1: float = 1.5
    b: float = 0.75
    top_k: int = 10
    index_path: Path = indexsdir/ "bm25_index.pkl"

@dataclass
class SemanticConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size : int = 64
    top_k: int = 10
    emneddings_path: Path = indexsdir/ "embeddings.npy"
    ids_path: Path = indexsdir/ "embedding_ids.json"
    # faiss index
    faiss_index_path: Path = indexsdir/ "faiss.index"
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
    index_path:Path =  indexsdir/ "colbert_embeddings.npy"
    ids_path : Path = indexsdir/ "colbert_ids.json"

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
    model_name : str = "cross-encoder/ms-macro-MiniLM-L-6-v2"
    top_k : int = 20
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
    queries_files: Path = querydir/ "queries.jsonl"
    result_file: Path = evalutiondir/ "result.json"
    train_ratio: float = 0.6
    dev_ration: float= 0.2
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

