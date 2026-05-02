from __future__ import annotations
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional
from app.core.retrievers.base import BaseRetriever
from app.models.schemas import Article, RetrievedDoc
from config.config import SEMANTIC_CFG
 
logger = logging.getLogger(__name__)