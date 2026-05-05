from __future__ import annotations
import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
 
BASE     = Path(__file__).resolve().parent
LOG_FILE = BASE / "logs" / "run_log.txt"
sys.path.insert(0, str(BASE))
 
# Create required folders if missing 
for _folder in ["logs", "data/corpus", "data/queries", "data/processed",
                "indexes", "evaluation"]:
    (BASE / _folder).mkdir(parents=True, exist_ok=True)
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
logger = logging.getLogger(__name__)
 
 
def run_step(name: str, module: str):
    logger.info("=" * 60)
    logger.info("STEP: %s", name.upper())
    logger.info("=" * 60)
    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, module],
        cwd=BASE,
    )
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        logger.error("Step %s FAILED (exit %d)", name, result.returncode)
        sys.exit(result.returncode)
    logger.info("Step %s completed in %.1fs", name, elapsed)
 
 
STEPS = {
    "fetch":      ("Fetch corpus",          "scripts/fetch_data.py"),
    "preprocess": ("Preprocess corpus",     "scripts/preprocess.py"),
    "index":      ("Build indexes",         "scripts/build_index.py"),
    "queries":    ("Generate eval queries", "scripts/generate_queries.py"),
    "eval":       ("Run evaluation",        "evaluation/evaluator.py"),
}
 
 
def serve():
    from config.config import API_CFG
    logger.info("Starting API server on %s:%d", API_CFG.host, API_CFG.port)
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", API_CFG.host,
        "--port", str(API_CFG.port),
        "--workers", str(API_CFG.workers),
        "--log-level", API_CFG.log_level,
    ], cwd=BASE)
 
 
def main():
    parser = argparse.ArgumentParser(description="SporTech Retrieval System")
    parser.add_argument("--step", choices=list(STEPS.keys()) + ["serve"], default=None,
                        help="Run a specific step only")
    args = parser.parse_args()
 
    if args.step == "serve":
        serve()
        return
 
    if args.step:
        name, module = STEPS[args.step]
        run_step(name, module)
        return
 
    # Full pipeline
    logger.info("Running full SporTech Retrieval pipeline…")
    total_t0 = time.perf_counter()
 
    for step_key, (name, module) in STEPS.items():
        run_step(name, module)
 
    total = time.perf_counter() - total_t0
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE in %.1fs", total)
    logger.info("Results: evaluation/results.json")
    logger.info("Summary: evaluation/summary.txt")
    logger.info("To start API: python run.py --step serve")
    logger.info("=" * 60)
 
 
if __name__ == "__main__":
    main()