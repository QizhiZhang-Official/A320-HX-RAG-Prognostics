import threading

from fastapi import APIRouter, Depends
from ragnostics.api.dependencies import get_hybrid, get_lock
from ragnostics.api.schemas import RetrieveRequest, RetrieveResponse, Source
from ragnostics.retrieval.hybrid import HybridRetriever


router = APIRouter(prefix="/retrieve", tags=["retrieve"])


@router.post(path="", response_model=RetrieveResponse)
def retrieve(
    body: RetrieveRequest,
    hybrid: HybridRetriever = Depends(get_hybrid),
    lock: threading.Lock = Depends(get_lock),
) -> RetrieveResponse:
    with lock:
        documents = hybrid.invoke(body.query)
    results = []
    for doc in documents:
        src = Source(
            chunk_id=doc.metadata["chunk_id"],
            section=doc.metadata["section"],
            title=doc.metadata["title"],
            score=doc.metadata["score"],
            text=doc.page_content,
        )
        results.append(src)

    return RetrieveResponse(results=results)
