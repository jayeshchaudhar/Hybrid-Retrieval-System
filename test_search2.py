from app.services.search_service import SearchService
from app.models.schemas import SearchRequest

svc = SearchService()
svc.load_indexes()
req = SearchRequest(query='cricket', top_k=3, method='bm25', rerank=False)
resp = svc.search(req)
print(resp.model_dump_json())