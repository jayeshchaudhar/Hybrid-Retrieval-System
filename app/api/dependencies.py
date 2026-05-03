from functools import lru_cache
from app.services.search_service import SearchService
@lru_cache(maxsize=1)
def get_search_service() -> SearchService:
    svc = SearchService()
    svc.load_indexes()
    return svc
