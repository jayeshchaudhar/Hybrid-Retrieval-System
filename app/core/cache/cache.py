from __future__ import annotations
import hashlib
import json
import logging
from collections import OrderedDict
from typing import Optional, List
from app.models.schemas import RetrievedDoc
from config.config import CACHE_CFG, CacheConfig

logger = logging.getLogger(__name__)

def _cache_key(query: str, method: str, top_k: int, sport: Optional[str]) -> str:
    raw = f"{query}|{method}|{top_k}|{sport or ''}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]

class MemoryCache:
    def __init__(self, maxsize: int):
        self._store: OrderedDict[str, List] = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: str) -> Optional[List[RetrievedDoc]]:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def set(self, key: str, value: List[RetrievedDoc]) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self._maxsize:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


class SearchCache:
    def __init__(self, cfg: CacheConfig = CACHE_CFG):
        self.cfg = cfg
        self._memory = MemoryCache(cfg.max_memory_items)
        self._redis = None
        if cfg.backend == "redis":
            self._init_redis()

    def _init_redis(self):
        try:
            import redis
            self._redis = redis.from_url(self.cfg.redis_url, decode_responses=True)
            self._redis.ping()
            logger.info("Cache: Redis connected at %s", self.cfg.redis_url)
        except Exception as e:
            logger.warning("Cache: Redis unavailable (%s) — memory-only mode", e)
            self._redis = None

    def get(self, query: str, method: str, top_k: int, sport: Optional[str]) -> Optional[List[RetrievedDoc]]:
        key = _cache_key(query, method, top_k, sport)

        # Memory
        hit = self._memory.get(key)
        if hit is not None:
            return hit

        # Redis
        if self._redis:
            try:
                raw = self._redis.get(key)
                if raw:
                    docs = [RetrievedDoc(**d) for d in json.loads(raw)]
                    self._memory.set(key, docs)   # warm local cache
                    return docs
            except Exception as e:
                logger.debug("Cache Redis get error: %s", e)
        return None

    def set(self, query: str, method: str, top_k: int, sport: Optional[str], docs: List[RetrievedDoc]) -> None:
        key = _cache_key(query, method, top_k, sport)
        self._memory.set(key, docs)
        if self._redis:
            try:
                raw = json.dumps([d.model_dump() for d in docs])
                self._redis.setex(key, self.cfg.ttl_seconds, raw)
            except Exception as e:
                logger.debug("Cache Redis set error: %s", e)

    def stats(self) -> dict:
        return {"memory_items": len(self._memory), "redis_connected": self._redis is not None}
