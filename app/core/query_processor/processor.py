#Query cleaning + classification
from __future__ import annotations
import json
import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

from config.config import DATA_DIR
ENTITY_FILE = DATA_DIR / "entities.json"

# fallback if entities.json missing
_DEFAULT_ENTITIES = {
    "players": [
        "messi", "ronaldo", "neymar", "mbappe", "haaland", "salah",
        "bellingham", "vinicius", "pedri", "modric",
        "federer", "nadal", "djokovic", "alcaraz", "sinner",
        "serena", "williams", "swiatek", "sabalenka",
        "lebron", "curry", "durant", "james", "jokic", "doncic",
        "kohli", "tendulkar", "dhoni", "kumble", "rohit", "root",
        "bolt", "phelps", "tyson", "ali", "fury", "usyk", "joshua",
        "pogacar", "vingegaard", "mcilroy", "scheffler",
        "mcdavid", "ovechkin", "ohtani", "judge",
    ],
    "teams": [
        "barcelona", "real madrid", "manchester", "liverpool", "chelsea",
        "arsenal", "city", "juventus", "psg", "bayern",
        "lakers", "bulls", "warriors", "celtics", "nuggets", "bucks",
        "india", "australia", "england", "pakistan", "newzealand",
        "dodgers", "yankees", "oilers", "avalanche",
    ],
    "tournaments": [
        "world cup", "champions league", "wimbledon", "us open",
        "french open", "australian open", "olympics", "ipl",
        "premier league", "la liga", "serie a", "bundesliga",
        "nba finals", "super bowl", "stanley cup", "world series",
        "tour de france", "masters", "grand slam", "six nations",
        "ashes", "t20", "copa america", "euros",
    ],
}

def _load_entities() -> dict:
    if ENTITY_FILE.exists():
        try:
            with open(ENTITY_FILE) as f:
                data = json.load(f)
            logger.info(
                "Entities loaded from %s — players=%d  teams=%d  tournaments=%d",
                ENTITY_FILE,
                len(data.get("players", [])),
                len(data.get("teams", [])),
                len(data.get("tournaments", [])),
            )
            return data
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(
                "entities.json parse error (%s) — using built-in defaults", e
            )
    else:
        logger.info(
            "entities.json not found at %s — using built-in defaults. "
            "Create the file to add new players/teams without redeploying.",
            ENTITY_FILE,
        )
    return _DEFAULT_ENTITIES

def _build_entity_set(data: dict) -> set:
    return set(
        e.lower()
        for group in ("players", "teams", "tournaments")
        for e in data.get(group, [])
    )


_ENTITY_DATA  : dict = _load_entities()
_ENTITY_SET   : set  = _build_entity_set(_ENTITY_DATA)


def reload_entities() -> None:
    global _ENTITY_DATA, _ENTITY_SET
    _ENTITY_DATA = _load_entities()
    _ENTITY_SET  = _build_entity_set(_ENTITY_DATA)
    logger.info("Entities reloaded — total=%d", len(_ENTITY_SET))


# Constants

_QUESTION_WORDS   = {
    "who", "what", "where", "when", "how",
    "which", "why", "does", "is", "can", "did", "has",
}
_ENTITY_THRESHOLD = 0.4   # fraction of tokens that are entities


#ProcessedQuery dataclass 

@dataclass
class ProcessedQuery:
    original      : str
    cleaned       : str
    tokens        : List[str]
    entity_density: float          # 0..1  — high → entity/lexical retrieval
    is_question   : bool           # starts with question word
    query_type    : str            # "entity" | "semantic" | "factual" | "hybrid"
    detected_sport: Optional[str]


def _clean(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+",       " ",  text)
    text = re.sub(r"[^\w\s\-\']", " ", text)
    return text


def _entity_density(tokens: List[str]) -> float:
    if not tokens:
        return 0.0

    # Unigram hits
    hits = sum(1 for t in tokens if t in _ENTITY_SET)

    # Bigram hits (catches "real madrid", "world cup", "champions league")
    bigrams = {
        f"{tokens[i]} {tokens[i + 1]}"
        for i in range(len(tokens) - 1)
    }
    hits += sum(1 for b in bigrams if b in _ENTITY_SET)

    return min(hits / len(tokens), 1.0)


def _detect_sport(tokens: List[str], text: str) -> Optional[str]:
    from config.config import SPORTS
    for sport in SPORTS:
        if sport.replace("_", " ") in text or sport in tokens:
            return sport
    return None


def _classify(density: float, is_question: bool) -> str:
    if density >= _ENTITY_THRESHOLD and not is_question:
        return "entity"
    if is_question and density < 0.2:
        return "semantic"
    if is_question and density >= 0.2:
        return "factual"
    return "hybrid"


# Queryprocessor
class QueryProcessor:
    def process(self, query: str) -> ProcessedQuery:
        cleaned = _clean(query)
        tokens  = cleaned.split()
        density = _entity_density(tokens)
        is_q    = bool(tokens) and tokens[0] in _QUESTION_WORDS
        qtype   = _classify(density, is_q)
        sport   = _detect_sport(tokens, cleaned)

        return ProcessedQuery(
            original      = query,
            cleaned       = cleaned,
            tokens        = tokens,
            entity_density= density,
            is_question   = is_q,
            query_type    = qtype,
            detected_sport= sport,
        )



