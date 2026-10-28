import threading
from uuid import uuid4

from fastapi import APIRouter, Depends

from ragnostics.api.dependencies import get_graph, get_lock
from ragnostics.api.schemas import AskRequest, AskResponse, Source


router = APIRouter(prefix='/ask', tags=['ask'])

@router.post(path='', response_model=AskResponse)
def ask(body: AskResponse, graph=Depends(get_graph), lock: threading.Lock = Depends(get_lock)) -> AskResponse:
    with lock:
        result = graph.invoke({
            'question': body.question,
            'top_k'
        })