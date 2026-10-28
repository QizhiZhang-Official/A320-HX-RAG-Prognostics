from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from collections.abc import AsyncGenerator
import threading

from ragnostics.retrieval.dense import DenseRetriever
from ragnostics.retrieval.lexical import LexicalRetriever
from ragnostics.retrieval.hybrid import HybridRetriever
from ragnostics.graphs.rag_agent import build_rag_graph


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    dense = DenseRetriever(k=20, device='cpu')
    lexical = LexicalRetriever(k=20)
    dense.connect_database()
    lexical.connect_database()
    hybrid = HybridRetriever(dense=dense, lexical=lexical, k=5)
    
    app.state.dense = dense
    app.state.lexical = lexical
    app.state.hybrid = hybrid
    app.state.graph = build_rag_graph(retriever=hybrid)
    app.state.lock = threading.Lock()
    
    try:
        yield
    finally:
        dense.close_connection()
        lexical.close_connection()

def get_hybrid(request: Request) -> HybridRetriever:
    return request.app.state.hybrid

def get_graph(request: Request):
    return request.app.state.graph

def get_lock(request: Request) -> threading.Lock:
    return request.app.state.lock