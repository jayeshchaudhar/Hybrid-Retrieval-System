from app.services.search_service import SearchService
from app.models.schemas import SearchRequest

svc = SearchService()
svc.load_indexes()
req = SearchRequest(query='Virat Kohli IPL', top_k=3, method='bm25', rerank=False)
result = svc.search(req)
print(result)