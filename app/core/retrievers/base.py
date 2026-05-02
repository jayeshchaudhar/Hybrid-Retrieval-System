from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional
import time 
import logging
from app.models.schemas import Article, RetrievedDoc

logger = logging.getLogger(__name__)

class BaseRetriever(ABC):

    name: str = "base"

    def __inti__(self):
        self._indexed = False

    @abstractmethod
    def build_index(self, article: List[Article]) -> None:
        pass

    
    @abstractmethod
    def retrieve(
        self, 
        query: str,
        top_k: int = 10,
        sport_filter : Optional[str] = None,
    ) -> List[RetrievedDoc] : 
        pass
        
    @abstractmethod
    def save(self) -> None:
        pass

    @abstractmethod
    def load(self) -> None:
        pass

    def timed_retrive(
            self,
            query: str,
            top_k: int = 10,
            sport_filter: Optional[str] = None,
    ) -> tuple[List[RetrievedDoc], float]:
        t0 = time.perf_counter()
        results = self.retrieve(query, top_k,sport_filter)
        ms = (time.perf_counter() - t0)*100
        return results, ms

    @staticmethod

    def _make_snippet(text: str, max_chars: int = 200) -> str:
        return text[:max_chars].rsplit(" ",1)[0]+ "…" if len(text) > max_chars else text 
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} indexed = {self._indexed}>"
    



        
