from functools import lru_cache
import logging
import traceback
from app.services.search_service import SearchService

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_search_service() -> SearchService:
    try:
        logger.info("Initializing SearchService...")
        svc = SearchService()
        svc.load_indexes()
        logger.info("SearchService ready")
        return svc
    except Exception as e:
        logger.error("SearchService init failed: %s", e)
        traceback.print_exc()
        raise