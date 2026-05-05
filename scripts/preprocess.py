# Loads raw corpus -> cleans text -> computes entity lists -> writes processed corpus.
from __future__ import annotations
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.schemas import Article
from config.config import CORPUS_FILE, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\w\s\.\,\!\?\-\'\"]", " ", text)
    return text

def extract_entities(article: Article) -> list[str]:
    found = set(article.entities)
    for match in re.finditer(r"([A-Z][a-z]+ ){1,3}[A-Z][a-z]+", article.body):
        found.add(match.group().strip())
    return list(found)[:20]   # cap at 20

def preprocess(articles: list[Article]) -> list[Article]:
    processed = []
    for a in articles:
        a = a.model_copy(update={
            "title": clean_text(a.title),
            "body": clean_text(a.body),
            "entities": extract_entities(a),
            "word_count": len(a.body.split()),
        })
        processed.append(a)
    return processed

def main():
    logger.info("Loading corpus from %s…", CORPUS_FILE)
    with open(CORPUS_FILE) as f:
        raw = json.load(f)
    articles = [Article(**item) for item in raw]
    logger.info("Loaded %d articles", len(articles))

    processed = preprocess(articles)

    out_path = PROCESSED_DIR / "articles_processed.json"
    with open(out_path, "w") as f:
        json.dump([a.model_dump(mode="json") for a in processed], f, indent=2, default=str)
    logger.info("Processed corpus written to %s", out_path)


if __name__ == "__main__":
    main()


