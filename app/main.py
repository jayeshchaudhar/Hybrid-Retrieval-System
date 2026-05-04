"""
FastAPI entry point
Run:
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
"""
from __future__ import annotations
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from app.api.routes import router
from config.config import API_CFG

logging.basicConfig(
    level= logging.INFO,
    format= "%(asctime)S  %(levelname-8s)  %(name)  %s(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("="*50)
    logger.info("SportTech Retrival API - Starting up...")
    logger.info("="*50)

    # load indexes
    try:
        from app.api.dependencies import get_search_service
        svc = get_search_service()
        logger.info("Indexes loaded - System ready")
    except FileNotFoundError as e:
        logger.critical(
            "Index file missing: %s\n"
            "Run: python scripts/build_index.py",
            e
        )
        svc = None

    if svc is not None:
        try:
            svc.reranker._load()          
            logger.info("Reranker model pre-loaded — no cold-start on first request")
        except Exception as e:
            logger.warning(
                "Reranker pre-load failed (%s) — will load lazily on first request",
                e
            )

    logger.info("SporTech Retrieval API — ready to serve requests")
    logger.info("=" * 50)
    yield
    logger.info("SporTech Retrieval API — shutting down.")

app = FastAPI(
    title       = "SporTech Retrieval System",
    description = (
        "Production-grade sports article retrieval with 5 methods "
        "(TF-IDF, BM25, Semantic, DenseQA, Hybrid) and a confidence-based router. "
        "Designed to scale to 100M articles."
    ),
    version  = "1.0.0",
    docs_url = "/docs"  if API_CFG.enable_docs else None,
    redoc_url= "/redoc" if API_CFG.enable_docs else None,
    lifespan = lifespan,
)


app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["GET", "POST"],
    allow_headers = ["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    t0       = time.perf_counter()
    response = await call_next(request)
    ms       = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-Ms"] = f"{ms:.1f}"
    return response

app.include_router(router)
@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "SporTech Retrieval System v1.0",
        "docs"   : "/docs",
        "health" : "/api/v1/health",
    }

