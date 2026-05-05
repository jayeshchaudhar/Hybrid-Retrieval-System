from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.query_processor.processor import QueryProcessor
from app.core.router.router import QueryRouter
from app.core.retrievers.tfidf import TFIDFRetriever
from app.core.retrievers.bm25 import BM25Retriever
from app.core.retrievers.semantic import SemanticRetriever
from app.core.retrievers.dense_qa import DenseQARetriever
from app.core.retrievers.hybrid import HybridRetriever
from app.models.schemas import Article, LabeledQuery
from evaluation.metrics import compute_all
from config.config import (
    RouterConfig, ROUTER_CFG,
    PROCESSED_DIR, EVAL_CFG,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


def load_articles() -> list[Article]:
    corpus_path = PROCESSED_DIR / "articles_processed.json"
    if not corpus_path.exists():
        logger.error("Corpus not found — run preprocess.py first.")
        sys.exit(1)
    with open(corpus_path) as f:
        return [Article(**a) for a in json.load(f)]


def load_dev_queries() -> list[LabeledQuery]:
    if not EVAL_CFG.queries_file.exists():
        logger.error("queries.jsonl not found — run generate_queries.py first.")
        sys.exit(1)
    queries = []
    with open(EVAL_CFG.queries_file) as f:
        for line in f:
            q = LabeledQuery(**json.loads(line))
            if q.split == "dev":
                queries.append(q)
    if not queries:
        logger.error("No dev-split queries found.")
        sys.exit(1)
    logger.info("Loaded %d dev queries", len(queries))
    return queries

# Load and index all retrievers
def load_retrievers(articles: list[Article]) -> dict:
    retrievers = {
        "tfidf"    : TFIDFRetriever(),
        "bm25"     : BM25Retriever(),
        "semantic" : SemanticRetriever(),
        "dense_qa" : DenseQARetriever(),
        "hybrid"   : HybridRetriever(),
    }
    for name, r in retrievers.items():
        try:
            r.load()
            logger.info("Loaded index: %s", name)
        except FileNotFoundError:
            logger.warning("%s index not found — building now...", name)
            r.build_index(articles)
            r.save()
    return retrievers


# Find best method per query 

def get_best_method_per_query(
    queries    : list[LabeledQuery],
    retrievers : dict,
    metric     : str = "ndcg@10",
    top_k      : int = 10,
) -> dict[str, str]:
    logger.info("Finding best method per query (ground truth)...")
    best_per_query: dict[str, str] = {}

    for q in queries:
        relevance_map = {j.article_id: j.relevance for j in q.judgments}
        if not relevance_map:
            continue

        scores_per_method = {}
        for name, retriever in retrievers.items():
            try:
                results       = retriever.retrieve(q.text, top_k)
                retrieved_ids = [r.article_id for r in results]
                m             = compute_all(retrieved_ids, relevance_map)
                scores_per_method[name] = m.get(metric, 0.0)
            except Exception as e:
                logger.debug("Retriever %s failed on query %s: %s", name, q.id, e)
                scores_per_method[name] = 0.0

        best_method = max(scores_per_method, key=lambda k: scores_per_method[k])
        best_per_query[q.id] = best_method

    logger.info("Ground truth computed for %d queries", len(best_per_query))
    return best_per_query
def evaluate_thresholds(
    queries           : list[LabeledQuery],
    best_per_query    : dict[str, str],
    entity_thresh     : float,
    semantic_thresh   : float,
    hybrid_thresh     : float,
    fallback          : str = "hybrid",
) -> float:
    processor = QueryProcessor()
    router    = QueryRouter(RouterConfig(
        entity_query_threshold      = entity_thresh,
        semantic_fallback_threshold = semantic_thresh,
        hybrid_threshold            = hybrid_thresh,
        fallback_method             = fallback,
    ))

    correct = 0
    total   = 0

    for q in queries:
        if q.id not in best_per_query:
            continue
        pq       = processor.process(q.text)
        decision = router.route(pq)
        if decision.method == best_per_query[q.id]:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


# Grid search over all threshold combinations and optimises routing accuracy on dev split.

def grid_search(
    queries        : list[LabeledQuery],
    best_per_query : dict[str, str],
    entity_values  : list[float],
    semantic_values: list[float],
    hybrid_values  : list[float],
) -> dict:
    total = len(entity_values) * len(semantic_values) * len(hybrid_values)
    logger.info(
        "Grid search: %d combinations (%d × %d × %d)",
        total,
        len(entity_values),
        len(semantic_values),
        len(hybrid_values),
    )

    results  = []
    best     = {
        "entity_threshold"   : entity_values[0],
        "semantic_threshold" : semantic_values[0],
        "hybrid_threshold"   : hybrid_values[0],
        "accuracy"           : 0.0,
    }
    run_num  = 0

    for et, st, ht in product(entity_values, semantic_values, hybrid_values):
        run_num += 1
        t0       = time.perf_counter()

        accuracy = evaluate_thresholds(queries, best_per_query, et, st, ht)
        elapsed  = (time.perf_counter() - t0) * 1000

        results.append({
            "entity_threshold"   : et,
            "semantic_threshold" : st,
            "hybrid_threshold"   : ht,
            "accuracy"           : round(accuracy, 4),
        })

        logger.info(
            "  [%3d/%d]  entity=%.2f  semantic=%.2f  hybrid=%.2f  "
            "accuracy=%.4f  (%.0fms)",
            run_num, total, et, st, ht, accuracy, elapsed,
        )

        if accuracy > best["accuracy"]:
            best = {
                "entity_threshold"   : et,
                "semantic_threshold" : st,
                "hybrid_threshold"   : ht,
                "accuracy"           : accuracy,
            }
            logger.info("  *** New best! accuracy=%.4f ***", accuracy)

    return {"best": best, "all_results": results}

def update_config(best: dict) -> None:
    config_path = Path(__file__).resolve().parent.parent / "config" / "config.py"
    with open(config_path) as f:
        content = f.read()

    import re

    content = re.sub(
        r"(entity_query_threshold:\s*float\s*=\s*)[\d.]+",
        f"\\g<1>{best['entity_threshold']}",
        content,
    )
    content = re.sub(
        r"(semantic_fallback_threshold:\s*float\s*=\s*)[\d.]+",
        f"\\g<1>{best['semantic_threshold']}",
        content,
    )
    content = re.sub(
        r"(hybrid_threshold:\s*float\s*=\s*)[\d.]+",
        f"\\g<1>{best['hybrid_threshold']}",
        content,
    )

    with open(config_path, "w") as f:
        f.write(content)

    logger.info(
        "config/config.py updated:\n"
        "  entity_query_threshold      = %.2f\n"
        "  semantic_fallback_threshold = %.2f\n"
        "  hybrid_threshold            = %.2f",
        best["entity_threshold"],
        best["semantic_threshold"],
        best["hybrid_threshold"],
    )
def save_results(results: dict, best: dict) -> None:
    out_path = Path(__file__).resolve().parent.parent / "evaluation" / "router_tuning.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "best"       : best,
        "all_results": results["all_results"],
        "timestamp"  : time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Tuning results saved → %s", out_path)

def main():
    parser = argparse.ArgumentParser(
        description="Tune router thresholds on dev split"
    )
    parser.add_argument(
        "--entity", nargs="+", type=float,
        default=[0.3, 0.4, 0.5, 0.6, 0.7],
        help="entity_query_threshold values (default: 0.3 0.4 0.5 0.6 0.7)",
    )
    parser.add_argument(
        "--semantic", nargs="+", type=float,
        default=[0.3, 0.4, 0.5],
        help="semantic_fallback_threshold values (default: 0.3 0.4 0.5)",
    )
    parser.add_argument(
        "--hybrid", nargs="+", type=float,
        default=[0.4, 0.5, 0.6],
        help="hybrid_threshold values (default: 0.4 0.5 0.6)",
    )
    parser.add_argument(
        "--metric", type=str, default="ndcg@10",
        choices=["ndcg@5", "ndcg@10", "map@10", "mrr", "precision@5", "recall@10"],
        help="Metric to determine best method per query (default: ndcg@10)",
    )
    parser.add_argument(
        "--no-update", action="store_true",
        help="Do not update config.py with best thresholds",
    )
    args = parser.parse_args()

    articles = load_articles()
    queries  = load_dev_queries()

    #Find ground truth
    retrievers     = load_retrievers(articles)
    best_per_query = get_best_method_per_query(queries, retrievers, args.metric)

    #Grid search 
    t_start  = time.perf_counter()
    results  = grid_search(
        queries, best_per_query,
        args.entity, args.semantic, args.hybrid,
    )
    elapsed  = time.perf_counter() - t_start

    # Report
    best = results["best"]
    logger.info("\n" + "=" * 55)
    logger.info("ROUTER TUNING COMPLETE in %.1fs", elapsed)
    logger.info("=" * 55)
    logger.info("Best entity_threshold      : %.2f", best["entity_threshold"])
    logger.info("Best semantic_threshold    : %.2f", best["semantic_threshold"])
    logger.info("Best hybrid_threshold      : %.2f", best["hybrid_threshold"])
    logger.info("Routing accuracy           : %.4f", best["accuracy"])
    logger.info("=" * 55)
    save_results(results, best)

    if not args.no_update:
        update_config(best)
        logger.info("\nAdd this to DECISIONS.md:")
        logger.info(
            "  Router tuning: entity=%.2f semantic=%.2f hybrid=%.2f "
            "→ accuracy=%.4f on dev split",
            best["entity_threshold"],
            best["semantic_threshold"],
            best["hybrid_threshold"],
            best["accuracy"],
        )
    else:
        logger.info("\nSkipped config update (--no-update flag).")
        logger.info(
            "Manually set in config/config.py:\n"
            "  entity_query_threshold      = %.2f\n"
            "  semantic_fallback_threshold = %.2f\n"
            "  hybrid_threshold            = %.2f",
            best["entity_threshold"],
            best["semantic_threshold"],
            best["hybrid_threshold"],
        )


if __name__ == "__main__":
    main()
