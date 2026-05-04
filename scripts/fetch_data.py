"""
Fetches real sports articles from 40+ public RSS feeds.
Sources: BBC Sport, ESPN, Sky Sports, Reuters, Goal.com, Cricbuzz, NBA.com, ATP Tour, UEFA, Guardian, Yahoo Sports, and more.

Strategy:
  1. feedparser  - parse RSS/Atom feed XML
  2. newspaper3k - scrape full article body (optional, when RSS body < 100 chars)
  3. Fallback - use RSS <description> if newspaper3k unavailable
  4. Dedup - MD5 hash of normalised body

Install:
  pip install feedparser newspaper3k requests lxml_html_clean
"""

from __future__ import annotations
import argparse, hashlib, json, logging, sys, time
import regex as re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.models.schemas import Article
from config.config import CORPUS_FILE, SPORTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

RSS_FEEDS: dict[str, list[dict]] = {
    "football": [
        {"url": "https://feeds.bbci.co.uk/sport/football/rss.xml",       "source": "BBC Sport Football"},
        {"url": "https://www.skysports.com/rss/12040",                     "source": "Sky Sports Football"},
        {"url": "https://www.goal.com/feeds/en/news",                      "source": "Goal.com"},
        {"url": "https://www.theguardian.com/football/rss",                "source": "Guardian Football"},
        {"url": "https://www.espn.com/espn/rss/soccer/news",               "source": "ESPN Soccer"},
        {"url": "https://www.premierleague.com/news/rss",                  "source": "Premier League"},
        {"url": "https://feeds.reuters.com/reuters/sportsNews",            "source": "Reuters Sports"},
        {"url": "https://www.uefa.com/rssfeed/uefachampionsleague/",       "source": "UEFA"},
    ],
    "cricket": [
        {"url": "https://feeds.bbci.co.uk/sport/cricket/rss.xml",         "source": "BBC Sport Cricket"},
        {"url": "https://www.espncricinfo.com/rss/content/story/feeds/0.xml", "source": "ESPNcricinfo"},
        {"url": "https://www.cricbuzz.com/rss-feeds/cricket-news",         "source": "Cricbuzz"},
        {"url": "https://www.icc-cricket.com/media-releases/rss",          "source": "ICC Cricket"},
        {"url": "https://www.theguardian.com/sport/cricket/rss",           "source": "Guardian Cricket"},
    ],
    "basketball": [
        {"url": "https://www.espn.com/espn/rss/nba/news",                  "source": "ESPN NBA"},
        {"url": "https://feeds.bbci.co.uk/sport/basketball/rss.xml",       "source": "BBC Basketball"},
        {"url": "https://bleacherreport.com/nba.rss",                      "source": "Bleacher Report NBA"},
        {"url": "https://sports.yahoo.com/nba/rss.xml",                    "source": "Yahoo Sports NBA"},
    ],
    "tennis": [
        {"url": "https://feeds.bbci.co.uk/sport/tennis/rss.xml",           "source": "BBC Sport Tennis"},
        {"url": "https://www.espn.com/espn/rss/tennis/news",               "source": "ESPN Tennis"},
        {"url": "https://www.theguardian.com/sport/tennis/rss",            "source": "Guardian Tennis"},
        {"url": "https://www.atptour.com/en/media/rss-feed/xml-feed",      "source": "ATP Tour"},
        {"url": "https://www.wtatennis.com/rss",                           "source": "WTA Tennis"},
    ],
    "athletics": [
        {"url": "https://feeds.bbci.co.uk/sport/athletics/rss.xml",        "source": "BBC Athletics"},
        {"url": "https://www.worldathletics.org/rss/news",                 "source": "World Athletics"},
        {"url": "https://www.theguardian.com/sport/athletics/rss",         "source": "Guardian Athletics"},
    ],
    "swimming": [
        {"url": "https://feeds.bbci.co.uk/sport/swimming/rss.xml",         "source": "BBC Swimming"},
        {"url": "https://www.swimmingworldmagazine.com/feed/",              "source": "Swimming World"},
        {"url": "https://www.worldaquatics.com/rss/news",                  "source": "World Aquatics"},
    ],
    "cycling": [
        {"url": "https://www.cyclingnews.com/rss",                         "source": "CyclingNews"},
        {"url": "https://www.velonews.com/feed",                           "source": "VeloNews"},
        {"url": "https://www.theguardian.com/sport/cycling/rss",           "source": "Guardian Cycling"},
        {"url": "https://feeds.bbci.co.uk/sport/cycling/rss.xml",          "source": "BBC Cycling"},
    ],
    "boxing": [
        {"url": "https://www.espn.com/espn/rss/boxing/news",               "source": "ESPN Boxing"},
        {"url": "https://www.skysports.com/rss/12572",                     "source": "Sky Sports Boxing"},
        {"url": "https://www.theguardian.com/sport/boxing/rss",            "source": "Guardian Boxing"},
    ],
    "golf": [
        {"url": "https://feeds.bbci.co.uk/sport/golf/rss.xml",             "source": "BBC Sport Golf"},
        {"url": "https://www.espn.com/espn/rss/golf/news",                 "source": "ESPN Golf"},
        {"url": "https://www.pgatour.com/rss",                             "source": "PGA Tour"},
        {"url": "https://www.theguardian.com/sport/golf/rss",              "source": "Guardian Golf"},
    ],
    "rugby": [
        {"url": "https://feeds.bbci.co.uk/sport/rugby-union/rss.xml",      "source": "BBC Rugby"},
        {"url": "https://www.espn.com/espn/rss/rugby/news",                "source": "ESPN Rugby"},
        {"url": "https://www.theguardian.com/sport/rugby-union/rss",       "source": "Guardian Rugby"},
        {"url": "https://www.world.rugby/rss",                             "source": "World Rugby"},
    ],
    "baseball": [
        {"url": "https://www.espn.com/espn/rss/mlb/news",                  "source": "ESPN MLB"},
        {"url": "https://sports.yahoo.com/mlb/rss.xml",                    "source": "Yahoo Sports MLB"},
        {"url": "https://bleacherreport.com/mlb.rss",                      "source": "Bleacher Report MLB"},
    ],
    "hockey": [
        {"url": "https://www.espn.com/espn/rss/nhl/news",                  "source": "ESPN NHL"},
        {"url": "https://sports.yahoo.com/nhl/rss.xml",                    "source": "Yahoo Sports NHL"},
        {"url": "https://bleacherreport.com/nhl.rss",                      "source": "Bleacher Report NHL"},
    ],
    "volleyball": [
        {"url": "https://www.volleyball.world/en/rss",                     "source": "Volleyball World"},
        {"url": "https://www.fivb.com/rss/volleyball",                     "source": "FIVB"},
    ],
    "badminton": [
        {"url": "https://bwfbadminton.com/rss",                            "source": "BWF Badminton"},
    ],
    "table_tennis": [
        {"url": "https://www.ittf.com/feed/",                              "source": "ITTF"},
    ],
}

_GENERAL_FEEDS = [
    {"url": "https://www.theguardian.com/sport/rss",   "source": "Guardian Sport"},
    {"url": "https://www.espn.com/espn/rss/news",      "source": "ESPN General"},
    {"url": "https://feeds.bbci.co.uk/sport/rss.xml",  "source": "BBC Sport General"},
    {"url": "https://sports.yahoo.com/rss.xml",        "source": "Yahoo Sports"},
    {"url": "https://bleacherreport.com/sport.rss",    "source": "Bleacher Report"},
]

_SPORT_KEYWORDS: dict[str, list[str]] = {
    "football":    ["football","soccer","premier league","champions league","la liga",
                    "messi","ronaldo","mbappe","haaland","fifa","transfer","penalty"],
    "cricket":     ["cricket","ipl","ashes","test match","t20","odi","wicket","kohli",
                    "batting","bowling","over","six","boundary","stumps"],
    "basketball":  ["basketball","nba","lakers","celtics","warriors","lebron","curry",
                    "dunk","three-pointer","playoffs","draft"],
    "tennis":      ["tennis","wimbledon","us open","french open","australian open",
                    "grand slam","djokovic","alcaraz","serve","ace","break point"],
    "athletics":   ["athletics","sprint","marathon","100m","200m","hurdles","long jump",
                    "pole vault","relay","bolt","world record","track"],
    "swimming":    ["swimming","freestyle","butterfly","backstroke","breaststroke",
                    "relay","pool","phelps","ledecky","olymp"],
    "cycling":     ["cycling","tour de france","giro","vuelta","peloton","breakaway",
                    "time trial","yellow jersey","pogacar","vingegaard"],
    "boxing":      ["boxing","knockout","tko","heavyweight","welterweight","fury",
                    "usyk","joshua","canelo","title fight","bout","ring"],
    "golf":        ["golf","pga","masters","birdie","eagle","par","bogey","mcilroy",
                    "scheffler","fairway","green","bunker"],
    "rugby":       ["rugby","six nations","world cup","scrum","lineout","try",
                    "conversion","all blacks","springboks","maul","ruck"],
    "baseball":    ["baseball","mlb","home run","strikeout","pitcher","batting",
                    "ohtani","yankees","dodgers","world series","inning"],
    "hockey":      ["hockey","nhl","stanley cup","hat trick","power play","mcdavid",
                    "ovechkin","goalie","puck","faceoff"],
    "volleyball":  ["volleyball","spike","block","libero","setter","fivb","beach"],
    "badminton":   ["badminton","shuttle","smash","bwf","axelsen","sindhu","all england"],
    "table_tennis":["table tennis","ping pong","topspin","ittf","ma long","fan zhendong"],
}

TAG_RE = re.compile(r"<[^>]+>")
ENTITY_RE = re.compile(r"&[a-z]+;")
SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\p{L}\p{N}\s]")
def _detect_sport(title: str, body: str) -> Optional[str]:
    text = (title + " " + body).lower()
    best, best_n = None, 0
    for sport, kws in _SPORT_KEYWORDS.items():
        n = sum(1 for k in kws if k in text)
        if n > best_n:
            best, best_n = sport, n
    return best if best_n >= 2 else None


def _clean_html(text: str) -> str:
    text = TAG_RE.sub(" ", text or "")
    text = ENTITY_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def _make_id(title: str, url: str) -> str:
    return hashlib.md5(f"{title.lower()}|{url}".encode()).hexdigest()


def _parse_date(entry) -> Optional[datetime]:
    for attr in ("published_parsed", "updated_parsed"):
        val = getattr(entry, attr, None)
        if val:
            try:
                return datetime(*val[:6], tzinfo=timezone.utc)
            except Exception:
                pass
    return None

#Deduplicator
class Deduplicator:
    def __init__(self):
        self._seen: set[str] = set()
        self.dropped = 0

    def is_duplicate(self, body: str) -> bool:
        norm = SPACE_RE.sub(" ", body.lower().strip())
        norm = PUNCT_RE.sub("", norm)   # better than [^\w\s]
        h = hashlib.md5(norm.encode()).hexdigest()

        if h in self._seen:
            self.dropped += 1
            return True

        self._seen.add(h)
        return False

#RSS fetch
def fetch_rss(feed_info: dict, sport: str) -> list[dict]:
    try:
        import feedparser
    except ImportError:
        logger.error("feedparser not installed — pip install feedparser")
        return []

    url, source = feed_info["url"], feed_info["source"]
    logger.info("  Fetching %-40s", source)
    try:
        feed = feedparser.parse(url, request_headers={
            "User-Agent": "SporTechBot/1.0 (academic research)",
            "Accept": "application/rss+xml, application/xml, text/xml",
        })
    except Exception as e:
        logger.warning("  Error on %s: %s", source, e)
        return []

    results = []
    for entry in feed.entries:
        title = _clean_html(getattr(entry, "title", ""))
        body  = _clean_html(getattr(entry, "summary", "") or getattr(entry, "description", ""))
        link  = getattr(entry, "link", "")
        if not title or len(title) < 10:
            continue
        results.append({
            "title": title, "description": body,
            "url": link, "source": source,
            "sport": sport, "pub_date": _parse_date(entry),
        })

    logger.info("  → %d entries", len(results))
    return results

#newspaper3k scraper
def scrape_body(url: str) -> Optional[str]:
    if not url or not url.startswith("http"):
        return None
    try:
        from newspaper import Article as NP
        art = NP(url)
        art.download()
        art.parse()
        body = art.text.strip()
        return body if len(body) > 100 else None
    except Exception:
        return None
    
# Build Article
def build_article(raw: dict, full_body: Optional[str] = None) -> Optional[Article]:
    body = (full_body or raw["description"]).strip()
    if len(body) < 50:
        return None
    sport = raw["sport"] or _detect_sport(raw["title"], body)
    if not sport or sport not in SPORTS:
        return None
    return Article(
        id=_make_id(raw["title"], raw.get("url", "")),
        title=raw["title"],
        body=body,
        sport=sport,
        source=raw["source"],
        url=raw.get("url", ""),
        published_at=raw.get("pub_date"),
        entities=[],
        tags=[sport],
        word_count=len(body.split()),
    )

#Corpus fetch
def fetch_corpus(target=500, scrape=True, sport_filter=None, delay=0.5) -> list[Article]:
    dedup    = Deduplicator()
    articles: list[Article] = []
    per_sport = max(10, target // len(SPORTS))
    sports    = [sport_filter] if sport_filter else SPORTS

    for sport in sports:
        feeds = RSS_FEEDS.get(sport, [])
        sport_count = 0
        logger.info("── %s (target %d) ──", sport.upper(), per_sport)

        for feed_info in feeds:
            if sport_count >= per_sport:
                break
            for raw in fetch_rss(feed_info, sport):
                if sport_count >= per_sport:
                    break
                full_body = None
                if scrape and len(raw.get("description", "")) < 100:
                    full_body = scrape_body(raw.get("url", ""))
                    if full_body:
                        time.sleep(0.2)
                art = build_article(raw, full_body)
                if art and not dedup.is_duplicate(art.body):
                    articles.append(art)
                    sport_count += 1
            time.sleep(delay)

        logger.info("  Collected: %d", sport_count)

    # Top up from general feeds if short
    if len(articles) < target * 0.8:
        logger.info("Topping up from general feeds (have %d/%d)…", len(articles), target)
        for feed_info in _GENERAL_FEEDS:
            if len(articles) >= target:
                break
            for raw in fetch_rss(feed_info, sport=""):
                if len(articles) >= target:
                    break
                raw["sport"] = _detect_sport(raw["title"], raw["description"])
                art = build_article(raw)
                if art and not dedup.is_duplicate(art.body):
                    articles.append(art)
            time.sleep(delay)

    logger.info("Total: %d articles  |  Duplicates dropped: %d",
                len(articles), dedup.dropped)
    return articles

#Manifest

def write_manifest(articles: list[Article]) -> None:
    from config.config import CORPUS_DIR
    manifest = {
        "generated_at": datetime.utcnow().isoformat(),
        "total_articles": len(articles),
        "dedup_method": "MD5 of lowercased, punctuation-stripped body",
        "sources": {},
        "sports": {},
    }
    for a in articles:
        manifest["sources"][a.source] = manifest["sources"].get(a.source, 0) + 1
        manifest["sports"][a.sport]   = manifest["sports"].get(a.sport, 0) + 1

    with open(CORPUS_DIR / "sourcing_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    with open(CORPUS_DIR / "deduplication_report.txt", "w") as f:
        f.write("DEDUPLICATION REPORT\n" + "="*40 + "\n\n")
        f.write(f"Total after dedup : {len(articles)}\n")
        f.write(f"Method            : MD5 hash of normalised body\n\n")
        f.write("Per source:\n")
        for src, n in sorted(manifest["sources"].items(), key=lambda x:-x[1]):
            f.write(f"  {src:<40} {n}\n")
        f.write("\nPer sport:\n")
        for sp, n in sorted(manifest["sports"].items(), key=lambda x:-x[1]):
            f.write(f"  {sp:<20} {n}\n")
    logger.info("Manifest + dedup report written to data/corpus/")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit",     type=int,   default=500)
    p.add_argument("--no-scrape", action="store_true")
    p.add_argument("--sport",     type=str,   default=None, choices=SPORTS)
    p.add_argument("--delay",     type=float, default=0.5)
    args = p.parse_args()

    # Check feedparser
    try:
        import feedparser  # noqa
    except ImportError:
        logger.error("Missing: pip install feedparser")
        sys.exit(1)
    if not args.no_scrape:
        try:
            import newspaper  # noqa
        except ImportError:
            logger.warning("newspaper3k not found — using RSS descriptions only. "
                           "Install with: pip install newspaper3k lxml_html_clean")
            args.no_scrape = True

    articles = fetch_corpus(
        target=args.limit,
        scrape=not args.no_scrape,
        sport_filter=args.sport,
        delay=args.delay,
    )

    if not articles:
        logger.error("No articles fetched. Check network access.")
        sys.exit(1)

    CORPUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        json.dump([a.model_dump(mode="json") for a in articles],
                  f, indent=2, default=str, ensure_ascii=False)
    logger.info("Corpus → %s  (%d articles)", CORPUS_FILE, len(articles))

    write_manifest(articles)

    # Distribution summary
    sport_counts: dict[str, int] = {}
    for a in articles:
        sport_counts[a.sport] = sport_counts.get(a.sport, 0) + 1
    logger.info("\nFinal distribution:")
    for sp in SPORTS:
        n = sport_counts.get(sp, 0)
        logger.info("  %-20s %3d  %s", sp, n, "-" * min(n, 35))


if __name__ == "__main__":
    main()

