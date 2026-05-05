from __future__ import annotations
import json
import logging
import time
import statistics
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.schemas import LabeledQuery, MethodMetrics, EvalReport, Article
from app.core.retrievers.tfidf import TFIDFRetriever
from app.core.retrievers.bm25 import BM25Retriever
from app.core.retrievers.semantic import SemanticRetriever
from app.core.retrievers.dense_qa import DenseQARetriever
from app.core.retrievers.hybrid import HybridRetriever
from app.core.reranker.reranker import Reranker
from evaluation.metrics import compute_all
from config.config import EVAL_CFG, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_queries(path: Path, split: str) -> List[LabeledQuery]:
    queries = []
    with open(path) as f:
        for line in f:
            q = LabeledQuery(**json.loads(line))
            if q.split == split:
                queries.append(q)
    return queries


def _percentile(data: List[float], p: int) -> float:
    if not data:
        return 0.0
    data_sorted = sorted(data)
    idx = max(0, int(len(data_sorted) * p / 100) - 1)
    return data_sorted[idx]


def evaluate_retriever(retriever, queries: List[LabeledQuery], top_k: int = 10) -> Dict[str, Any]:
    all_metrics: List[Dict[str, float]] = []
    latencies: List[float] = []
    errors: List[Dict] = []

    for q in queries:
        relevance_map = {j.article_id: j.relevance for j in q.judgments}
        if not relevance_map:
            continue

        t0 = time.perf_counter()
        results = retriever.retrieve(q.text, top_k)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        retrieved_ids = [r.article_id for r in results]
        m = compute_all(retrieved_ids, relevance_map)
        all_metrics.append(m)

        if m["ndcg@10"] < 0.2:
            errors.append({
                "query_id": q.id,
                "query_text": q.text,
                "query_type": q.query_type,
                "ndcg@10": m["ndcg@10"],
                "retrieved_top3": retrieved_ids[:3],
                "relevant": list(relevance_map.keys())[:3],
                "diagnosis": "low_recall" if m["recall@10"] < 0.3 else "low_precision",
            })

    def mean(key):
        vals = [m[key] for m in all_metrics]
        return statistics.mean(vals) if vals else 0.0

    return {
        "metrics": {k: mean(k) for k in ["ndcg@5", "ndcg@10", "map@10", "mrr", "precision@5", "recall@10"]},
        "p50_latency_ms": _percentile(latencies, 50),
        "p95_latency_ms": _percentile(latencies, 95),
        "n_queries": len(all_metrics),
        "errors": errors,
    }


def run_ablation_no_reranker(retriever, queries, top_k=10) -> Dict:
    return evaluate_retriever(retriever, queries, top_k)


def run_ablation_reduced_corpus(retriever, articles, queries, fraction=0.5) -> Dict:
    subset = articles[:int(len(articles) * fraction)]
    logger.info("Ablation: rebuilding BM25 on %d articles (%.0f%% corpus)…", len(subset), fraction*100)
    retriever.build_index(subset)
    return evaluate_retriever(retriever, queries)


def main():
    corpus_path = PROCESSED_DIR / "articles_processed.json"
    with open(corpus_path) as f:
        articles = [Article(**a) for a in json.load(f)]

    queries_path = EVAL_CFG.queries_file
    if not queries_path.exists():
        logger.error("Queries file not found at %s. Run generate_queries.py first.", queries_path)
        sys.exit(1)

    test_queries = _load_queries(queries_path, "test")
    logger.info("Evaluating on %d test queries…", len(test_queries))

    retrievers = {
        "tfidf":    TFIDFRetriever(),
        "bm25":     BM25Retriever(),
        "semantic": SemanticRetriever(),
        "dense_qa": DenseQARetriever(),
        "hybrid":   HybridRetriever(),
    }
    for name, r in retrievers.items():
        try:
            r.load()
        except FileNotFoundError:
            logger.warning("%s index not found — building now…", name)
            r.build_index(articles)
            r.save()

    reranker = Reranker()

    metrics_list: List[MethodMetrics] = []
    all_errors = []

    for name, retriever in retrievers.items():
        logger.info("Evaluating %s…", name)
        result = evaluate_retriever(retriever, test_queries)
        m = result["metrics"]
        metrics_list.append(MethodMetrics(
            method=name,
            ndcg_at_5=m["ndcg@5"],
            ndcg_at_10=m["ndcg@10"],
            map_at_10=m["map@10"],
            mrr=m["mrr"],
            precision_at_5=m["precision@5"],
            recall_at_10=m["recall@10"],
            p50_latency_ms=result["p50_latency_ms"],
            p95_latency_ms=result["p95_latency_ms"],
            n_queries=result["n_queries"],
        ))
        all_errors.extend(result["errors"])
        logger.info(
            "  nDCG@10=%.3f  MAP@10=%.3f  MRR=%.3f  p50=%.1fms  p95=%.1fms",
            m["ndcg@10"], m["map@10"], m["mrr"],
            result["p50_latency_ms"], result["p95_latency_ms"],
        )

    logger.info("Ablation 1: half-corpus BM25…")
    abl_bm25 = BM25Retriever()
    abl1 = run_ablation_reduced_corpus(abl_bm25, articles, test_queries, 0.5)
    ablation_results = {
        "half_corpus_bm25": {
            "description": "BM25 trained on 50% of corpus — tests corpus size sensitivity",
            "ndcg@10": abl1["metrics"]["ndcg@10"],
            "map@10": abl1["metrics"]["map@10"],
        }
    }

    logger.info("Ablation 2: hybrid without reranker…")
    abl2 = run_ablation_no_reranker(retrievers["hybrid"], test_queries)
    ablation_results["hybrid_no_reranker"] = {
        "description": "Hybrid RRF without cross-encoder reranker — measures reranker value",
        "ndcg@10": abl2["metrics"]["ndcg@10"],
        "map@10": abl2["metrics"]["map@10"],
    }

    best = max(metrics_list, key=lambda m: m.ndcg_at_10)
    logger.info("Best method: %s (nDCG@10=%.3f)", best.method, best.ndcg_at_10)

    report = EvalReport(
        metrics_per_method=metrics_list,
        best_method=best.method,
        ablation_results=ablation_results,
        error_analysis=all_errors[:20],   # top 20 failures
    )

    EVAL_CFG.results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_CFG.results_file, "w") as f:
        f.write(report.model_dump_json(indent=2))
    logger.info("Results written to %s", EVAL_CFG.results_file)

    # Human-readable summary
    summary_path = EVAL_CFG.results_file.parent / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("SPORTECH RETRIEVAL BENCHMARK — RESULTS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Method':<12} {'nDCG@5':>8} {'nDCG@10':>8} {'MAP@10':>8} {'MRR':>8} {'P@5':>8} {'R@10':>8} {'p50ms':>7} {'p95ms':>7}\n")
        f.write("-" * 80 + "\n")
        for m in sorted(metrics_list, key=lambda x: -x.ndcg_at_10):
            f.write(
                f"{m.method:<12} {m.ndcg_at_5:>8.3f} {m.ndcg_at_10:>8.3f} "
                f"{m.map_at_10:>8.3f} {m.mrr:>8.3f} {m.precision_at_5:>8.3f} "
                f"{m.recall_at_10:>8.3f} {m.p50_latency_ms:>7.1f} {m.p95_latency_ms:>7.1f}\n"
            )
        f.write(f"\nBest method: {best.method}\n")
        f.write("\nAblation Results:\n")
        for k, v in ablation_results.items():
            f.write(f"  {k}: {v}\n")
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
