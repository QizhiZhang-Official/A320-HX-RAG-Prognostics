from typing import TypedDict


class RAGState(TypedDict):
    question: str
    chunk_ids: list[str]
    contexts: str
    answer: str