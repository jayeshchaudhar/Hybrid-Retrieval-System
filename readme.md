#Project Structure

Hybrid_search
|__ app/
|     |___api/                      FastAPI routes + dependency injection
|     |___core/
|     |    |____retrievers/         5 retrieval methods (TF-IDF, BM25, Semantic, DenseQA, Hybrid)
|     |    |____router/             Confidence-based query router
|     |    |____reranker/           Cross-encoder re-ranker
|     |    |____query_processor/    Query cleaning + type classification
|     |    |____cache/              LRU memory + optional Redis cache
|     |____models/                  Pydantic schemas
|     |____services/                SearchService orchestrator
|     |____main.py                  FastAPI app
|___config/config.py                All hyperparameters in one place
|___ data/                          Corpus, queries, processed data
|____indexes/                       Pre-built FAISS + BM25 + TF-IDF indexes
|____evaluation/                    Metrics, evaluator, results
|____scripts/                       fetch_data, preprocess, build_index, generate_queries
|____tests/                         Pytest unit tests
|____run.py                         Single pipeline entry point
|____MEMO.md                        Technical writeup
|____DECISIONS.md                   Architecture decisions log
|____requirements.txt               Project Requirements

