from pydantic import BaseModel, Field


class Source(BaseModel):
    chunk_id: str
    section: str
    title: str
    score: float
    text: str

class AskRequest(BaseModel):
    question: str
    top_k: int = Field(default=3, ge=1, le=8)

class AskResponse(BaseModel):
    answer: str
    sources: list
    request_id: str

class RetrieveRequest(BaseModel):
    query: str
    k: int = 3

class RetrieveResponse(BaseModel):
    results: list

class HealthResponse(BaseModel):
    status: str