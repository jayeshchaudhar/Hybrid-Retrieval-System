import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from app.models.schemas import Article
from app.core.retrievers.tfidf import TFIDFRetriever
from app.core.retrievers.bm25 import BM25Retriever
from app.core.retrievers.semantic import SemanticRetriever
from app.core.query_processor.processor import QueryProcessor
from app.core.router.router import QueryRouter
from app.core.cache.cache import SearchCache, MemoryCache
from evaluation.metrics import ndcg, average_precision, reciprocal_rank, compute_all


# Fixtures 

@pytest.fixture
def sample_articles():
    return [
        Article(id="a1", title="Messi scores in Champions League final",
                body="Lionel Messi delivered a stunning hat-trick to win the Champions League for Barcelona.",
                sport="football", source="test"),
        Article(id="a2", title="Kohli century in IPL 2024",
                body="Virat Kohli scored a brilliant century for RCB in the IPL 2024 final.",
                sport="cricket", source="test"),
        Article(id="a3", title="LeBron leads Lakers to NBA Finals",
                body="LeBron James led the LA Lakers to victory in a thrilling NBA Finals game.",
                sport="basketball", source="test"),
        Article(id="a4", title="Djokovic wins Wimbledon again",
                body="Novak Djokovic claimed his record 8th Wimbledon title in a five-set thriller.",
                sport="tennis", source="test"),
        Article(id="a5", title="Bolt sets new world record",
                body="Usain Bolt broke the 100m world record at the World Athletics Championships.",
                sport="athletics", source="test"),
    ]


# TF-IDF

class TestTFIDFRetriever:
    def test_build_and_retrieve(self, sample_articles):
        r = TFIDFRetriever()
        r.build_index(sample_articles)
        results = r.retrieve("Messi Champions League", top_k=3)
        assert len(results) >= 1
        assert results[0].article_id == "a1"

    def test_sport_filter(self, sample_articles):
        r = TFIDFRetriever()
        r.build_index(sample_articles)
        results = r.retrieve("world record", top_k=5, sport_filter="athletics")
        assert all(res.sport == "athletics" for res in results)

    def test_empty_query_returns_results(self, sample_articles):
        r = TFIDFRetriever()
        r.build_index(sample_articles)
        results = r.retrieve("sports", top_k=5)
        assert isinstance(results, list)

    def test_not_indexed_raises(self):
        r = TFIDFRetriever()
        with pytest.raises(RuntimeError):
            r.retrieve("test")


#  BM25

class TestBM25Retriever:
    def test_build_and_retrieve(self, sample_articles):
        r = BM25Retriever()
        r.build_index(sample_articles)
        results = r.retrieve("Kohli IPL century", top_k=3)
        assert len(results) >= 1
        assert results[0].article_id == "a2"

    def test_sport_filter(self, sample_articles):
        r = BM25Retriever()
        r.build_index(sample_articles)
        results = r.retrieve("final game", top_k=5, sport_filter="basketball")
        assert all(res.sport == "basketball" for res in results)



#  Semantic

class TestSemanticRetriever:
    def test_class_name_correct(self):
        r = SemanticRetriever()
        assert r.name == "semantic"

    def test_not_indexed_raises(self):
        r = SemanticRetriever()
        with pytest.raises(RuntimeError):
            r.retrieve("test query")

    def test_emb_cache_starts_empty(self):
        r = SemanticRetriever()
        assert len(r._emb_cache) == 0

    def test_model_is_none_before_load(self):
        r = SemanticRetriever()
        assert r.model is None  

    def test_index_is_none_before_build(self):
        r = SemanticRetriever()
        assert r.index is None   

# Query Processor 

class TestQueryProcessor:
    def test_entity_query(self):
        p = QueryProcessor()
        pq = p.process("Messi Barcelona Champions League")
        assert pq.query_type == "entity"
        assert pq.entity_density > 0

    def test_question_query(self):
        p = QueryProcessor()
        pq = p.process("Who won the Wimbledon title this year?")
        assert pq.is_question is True
        assert pq.query_type in ("factual", "semantic")

    def test_sport_detection(self):
        p = QueryProcessor()
        pq = p.process("cricket world cup final score")
        assert pq.detected_sport == "cricket"

    def test_cleaning(self):
        p = QueryProcessor()
        pq = p.process("  Messi!!!   scored  ")
        assert "  " not in pq.cleaned


# Router

class TestQueryRouter:
    def test_entity_routes_to_bm25(self):
        processor = QueryProcessor()
        router = QueryRouter()
        pq = processor.process("Messi Barcelona Champions League goal")
        decision = router.route(pq)
        assert decision.method in ("bm25", "tfidf")

    def test_question_routes_to_dense_qa(self):
        processor = QueryProcessor()
        router = QueryRouter()
        pq = processor.process("Who is the best tennis player of all time?")
        decision = router.route(pq)
        assert decision.method in ("dense_qa", "semantic", "hybrid")

    def test_force_method(self):
        processor = QueryProcessor()
        router = QueryRouter()
        pq = processor.process("anything")
        decision = router.route(pq, force_method="tfidf")
        assert decision.method == "tfidf"
        assert decision.confidence == 1.0


# Metrics

class TestMetrics:
    def test_ndcg_perfect(self):
        retrieved = ["a", "b", "c"]
        rel_map = {"a": 3, "b": 2, "c": 1}
        score = ndcg(retrieved, rel_map, k=3)
        assert abs(score - 1.0) < 1e-6

    def test_ndcg_zero(self):
        retrieved = ["x", "y", "z"]
        rel_map = {"a": 3, "b": 2}
        score = ndcg(retrieved, rel_map, k=3)
        assert score == 0.0

    def test_reciprocal_rank(self):
        assert reciprocal_rank(["a", "b", "c"], {"b"}) == pytest.approx(0.5)
        assert reciprocal_rank(["a", "b", "c"], {"a"}) == pytest.approx(1.0)
        assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0

    def test_average_precision(self):
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "c"}
        ap = average_precision(retrieved, relevant, k=4)
        assert ap > 0

    def test_compute_all_returns_all_keys(self):
        retrieved = ["a", "b", "c"]
        rel_map = {"a": 3, "b": 1}
        metrics = compute_all(retrieved, rel_map)
        expected_keys = {"ndcg@5", "ndcg@10", "map@10", "mrr", "precision@5", "recall@10"}
        assert set(metrics.keys()) == expected_keys


# Cache

class TestMemoryCache:
    def test_set_get(self):
        cache = MemoryCache(maxsize=100)
        cache.set("k1", ["val"])
        assert cache.get("k1") == ["val"]

    def test_eviction(self):
        cache = MemoryCache(maxsize=3)
        for i in range(4):
            cache.set(f"k{i}", [i])
        assert cache.get("k0") is None   # evicted
        assert cache.get("k3") is not None

    def test_miss(self):
        cache = MemoryCache(maxsize=10)
        assert cache.get("nonexistent") is None
