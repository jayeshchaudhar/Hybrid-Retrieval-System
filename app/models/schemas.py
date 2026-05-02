from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


#corpus

class Article(BaseModel):
    id : str = Field(default_factory= lambda: str(uuid.uuid4()))
    title: str
    body: str
    sport: str
    source: str
    url: Optional[str] = None
    published_art: Optional[datetime] = None
    entities: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    word_count: int = 0

    def full_text(self) -> str:
        return f"{self.title}. {self.body}"
    

#query & relevance

class RelevanceJudgement(BaseModel):
    article_id = str
    relevance: int = Field(ge=0, le=3,
                           description= "0=irrelevant, 1 = marginally, 2: relevant, 3 = highly relevant")
    

class LabeledQuery(BaseModel):
    id: str = Field(default_factory= lambda: str(uuid.uuid4()))
    text: str
    sport : Optional[str] = None
    query_type : str = Field(description="entity|factual|semantic|hybrid",)
    split: str = Field(description="train|dev|test")
    judgement: List[RelevanceJudgement]  = Field(default_factory=list)

    def relevant_ids(self, min_grade: int  =  1) -> List[str]:
        return [j.article_id for j in self.judgement if j.relevance >= min_grade]
    

#retrivals

class RetrievedDoc(BaseModel):
    article_id : str
    title: str
    sport: str
    score: float
    retriever : str
    snippet : str = ""


class Searchrequest(BaseModel):
    query: str = Field(min_length= 1, max_length= 512)
    top_k : int = Field(default=10, ge=1, le=100)
    sport_filter: Optional[str] = None
    method: Optional[str] = None
    rerank: bool = True

class SearchResponse(BaseModel):
    query: str
    result: List[RetrievedDoc]
    method_used: str
    router_confidence : float
    latency_ms: float
    cached: bool = False

#Evaluation

class Methodmetrics(BaseModel):
    method: str
    ndcg_at_5: float
    ndcg_at_10 : float
    map_at_10 : float
    mrr : float
    percision_at_5 : float
    recall_at_10: float
    p50_latency_ms: float
    p95_latency_ms: float
    n_queries: int

class Evalreport(BaseModel):
    run_id: str = Field(default_factory= lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metrics_per_method: List[Methodmetrics]
    base_model: str
    ablation_results : Dict[str, Any] = Field(default_factory= dict)
    error_analysis : List[Dict[str, Any]] = Field(default_factory=list)



    