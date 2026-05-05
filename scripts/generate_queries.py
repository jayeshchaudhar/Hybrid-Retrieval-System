from __future__ import annotations
import json
import random
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.schemas import LabeledQuery, RelevanceJudgment, Article
from config.config import EVAL_CFG, PROCESSED_DIR, SPORTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

random.seed(0)

# ── Query templates per type ──────────────────────────────────────────────────

_ENTITY_QUERIES = [
    "Lionel Messi Barcelona Champions League",
    "Virat Kohli century IPL India",
    "LeBron James Lakers NBA Finals",
    "Novak Djokovic Wimbledon grand slam",
    "Usain Bolt sprint world record Olympics",
    "Tadej Pogacar Tour de France yellow jersey",
    "Tyson Fury heavyweight WBC championship",
    "Rory McIlroy Masters Augusta National",
    "Antoine Dupont Rugby World Cup France",
    "Shohei Ohtani LA Dodgers World Series",
    "Viktor Axelsen BWF badminton championship",
    "Ma Long table tennis Olympics China",
    "Erling Haaland Manchester City Premier League",
    "PV Sindhu India Olympics badminton",
    "Connor McDavid Edmonton Oilers Stanley Cup",
]

_FACTUAL_QUERIES = [
    "Who scored the most goals in the UEFA Champions League?",
    "Which team won the IPL 2023 final?",
    "Who is the current Wimbledon men's singles champion?",
    "How many Grand Slams has Rafael Nadal won?",
    "What is the record for fastest 100m sprint?",
    "Who won the Tour de France most times?",
    "Which country has won the most Olympic swimming medals?",
    "Who holds the heavyweight boxing championship?",
    "What is the highest score in Test cricket?",
    "Which NBA team has the most championships?",
    "Who won the Rugby World Cup in 2023?",
    "Which golfer has won the most majors?",
    "How many World Cups has Brazil won in football?",
    "What is the badminton world record smash speed?",
    "Who is the most decorated Olympian of all time?",
]

_SEMANTIC_QUERIES = [
    "athletic performance under pressure during crucial moments",
    "young talent emerging in competitive sports leagues",
    "tactical evolution in modern team sports strategy",
    "mental health challenges faced by professional athletes",
    "technology and data analytics changing sports performance",
    "underdog teams defeating stronger opponents unexpectedly",
    "comeback stories after career-threatening injuries",
    "financial transformation of sports through broadcasting rights",
    "role of coaching in developing world-class athletes",
    "training methodologies that revolutionised athletic performance",
    "the psychological impact of home crowd advantage",
    "nutrition and recovery science in elite sport",
    "legacy of legendary athletes on younger generations",
    "referee decisions that changed major tournament outcomes",
    "international transfers and their effect on team dynamics",
]

_HYBRID_QUERIES = [
    "Messi retirement impact on Argentina football",
    "Cricket batting techniques against fast bowling",
    "NBA draft prospects and career development",
    "Tennis serve speed versus accuracy trade-off",
    "Tour de France mountain stage performance tactics",
    "Football defensive formations in Champions League",
    "Swimming relay race team coordination strategy",
    "Boxing footwork and ring control techniques",
    "Rugby scrum dominance and set piece strategy",
    "Golf mental game under Major championship pressure",
    "Basketball three-point revolution changing game tactics",
    "Hockey penalty shootout goalkeeper statistics",
    "Volleyball libero defensive role evolution",
    "Badminton deceptive shot placement strategy",
    "Table tennis spin techniques at world level",
]


def _assign_relevance(query: str, articles: list[Article]) -> list[RelevanceJudgment]:
    """
    3 = Highly relevant
    2 = Relevant
    1 = Marginally relevant
    0 = Irrelevant
    """
    q_lower = query.lower()
    q_tokens = set(q_lower.split())

    judgments = []
    for a in articles:
        text = f"{a.title} {a.body}".lower()
        text_tokens = set(text.split())

        # Count matching tokens
        overlap = len(q_tokens & text_tokens)
        sport_match = a.sport.replace("_", " ") in q_lower or any(
            tok in q_lower for tok in a.sport.split("_")
        )
        entity_match = any(e.lower() in q_lower for e in a.entities)

        if entity_match and overlap >= 3:
            grade = 3
        elif entity_match or (sport_match and overlap >= 2):
            grade = 2
        elif sport_match:
            grade = 1
        else:
            grade = 0

        if grade > 0:
            judgments.append(RelevanceJudgment(article_id=a.id, relevance=grade))

    return judgments[:15]   # max 15 judgments per query


def build_eval_set(articles: list[Article]) -> list[LabeledQuery]:
    all_queries_raw = [
        (t, "entity")   for t in _ENTITY_QUERIES
    ] + [
        (t, "factual")  for t in _FACTUAL_QUERIES
    ] + [
        (t, "semantic") for t in _SEMANTIC_QUERIES
    ] + [
        (t, "hybrid")   for t in _HYBRID_QUERIES
    ]

    # Sample to 75
    random.shuffle(all_queries_raw)
    selected = all_queries_raw[:75]

    # Split: 60% train, 20% dev, 20% test 
    n = len(selected)
    splits = (
        ["train"] * int(n * 0.6) +
        ["dev"]   * int(n * 0.2) +
        ["test"]  * (n - int(n * 0.6) - int(n * 0.2))
    )

    queries = []
    for (text, qtype), split in zip(selected, splits):
        judgments = _assign_relevance(text, articles)
        q = LabeledQuery(
            text=text,
            query_type=qtype,
            split=split,
            judgments=judgments,
        )
        queries.append(q)

    return queries


def main():
    corpus_path = PROCESSED_DIR / "articles_processed.json"
    with open(corpus_path) as f:
        articles = [Article(**a) for a in json.load(f)]
    logger.info("Loaded %d articles", len(articles))

    queries = build_eval_set(articles)
    logger.info("Generated %d labeled queries", len(queries))

    split_counts = {}
    for q in queries:
        split_counts[q.split] = split_counts.get(q.split, 0) + 1
    logger.info("Split distribution: %s", split_counts)

    EVAL_CFG.queries_file.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_CFG.queries_file, "w") as f:
        for q in queries:
            f.write(q.model_dump_json() + "\n")
    logger.info("Queries written to %s", EVAL_CFG.queries_file)

    rubric_path = EVAL_CFG.queries_file.parent / "labeling_rubric.md"
    with open(rubric_path, "w") as f:
        f.write("""# Labeling Rubric — SporTech Retrieval Benchmark

## Relevance Grades

| Grade | Label              | Criteria                                                                            |
|-------|--------------------|-------------------------------------------------------------------------------------|
| 3     | Highly Relevant    | Article directly answers or is specifically about the query topic, player, or event |
| 2     | Relevant           | Article discusses the same player, team, or tournament mentioned in the query       |
| 1     | Marginally Relevant| Article covers the same sport; only tangentially related to the specific query      |
| 0     | Irrelevant         | Different sport or completely unrelated content                                     |

## Query Types

- **entity**: Named entity + tournament/team queries (best for lexical methods)
- **factual**: Natural language questions with specific factual answers (best for DenseQA)
- **semantic**: Concept/theme queries without specific named entities (best for semantic)
- **hybrid**: Mixed entity + context queries (best for hybrid RRF)

## Inter-Annotator Agreement
Sanity check: compute Krippendorff's alpha across 10% double-annotated queries.
Target: α > 0.7 before trusting labels.
""")
    logger.info("Rubric written to %s", rubric_path)


if __name__ == "__main__":
    main()
