from __future__ import annotations
import time
from fastapi import APIRouter, Depends, HTTPException, Query as QParam
from app.models.schemas import SearchRequest, SearchResponse
from app.services.search_service import SearchService
from app.api.dependencies import get_search_service
from config.config import INDEXES_DIR

router = APIRouter(prefix="/api/v1")
@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search sports articles",
)
async def search(
    req: SearchRequest,
    svc: SearchService = Depends(get_search_service),
) -> SearchResponse:
    try:
        return svc.search(req)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

@router.get("/health", summary="Health check + index staleness")
async def health(svc: SearchService = Depends(get_search_service)):
    faiss_path = INDEXES_DIR / "faiss.index"

    if faiss_path.exists():
        age_hours = (time.time() - faiss_path.stat().st_mtime) / 3600
        index_info = {
            "index_age_hours": round(age_hours, 1),
            "stale":           age_hours > 24,
            "stale_threshold": "24 hours",
        }
    else:
        # Index file not found — not built yet
        index_info = {
            "index_age_hours": None,
            "stale":           True,
            "stale_threshold": "24 hours",
        }

    return {
        "status":        "ok" if svc._ready else "degraded",
        "indexes_ready": svc._ready,
        **index_info,
        "cache_stats":   svc.cache.stats(),
    }
@router.get("/methods", summary="List available retrieval methods")
async def list_methods(svc: SearchService = Depends(get_search_service)):
    """Returns all available retrieval methods and the active routing strategy."""
    return {
        "methods":        list(svc.retrievers.keys()),
        "default_router": "confidence-based",
        "reranker":       svc.reranker.cfg.enabled,
    }

@router.get(
    "/search",
    response_model=SearchResponse,
    summary="Search via GET (query params)",
)
async def search_get(
    q      : str       = QParam(...,        description="Search query"),
    top_k  : int       = QParam(default=10, ge=1, le=100),
    sport  : str | None= QParam(default=None),
    method : str | None= QParam(default=None),
    rerank : bool      = QParam(default=True),
    svc    : SearchService = Depends(get_search_service),
) -> SearchResponse:
    req = SearchRequest(
        query       = q,
        top_k       = top_k,
        sport_filter= sport,
        method      = method,
        rerank      = rerank,
    )
    try:
        return svc.search(req)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
