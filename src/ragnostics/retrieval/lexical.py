import os

import psycopg
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from psycopg import sql
from pydantic import Field
from rank_bm25 import BM25Okapi


class LexicalRetriever(BaseRetriever):
    k: int = 5
    docs: list[Document] = Field(default_factory=list)
    k1: float = 1.5
    b: float = 0.75
    connection: None = None
    cursor: None = None
    bm25: BM25Okapi | None = None
    chunks: list = Field(default_factory=list)
    query: str = ""
    RETRIEVER_SQL: str = """
        SELECT chunk_id, section, title, embedding_text FROM chunks
    """

    def connect_database(self) -> None:
        connection = psycopg.connect(os.environ["DATABASE_URL"])
        cursor = connection.cursor()
        self.connection = connection
        self.cursor = cursor

    def close_connection(self) -> None:
        self.cursor.close()
        self.cursor = None
        self.connection.close()
        self.connection = None

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def _build_chunk_index(self) -> None:
        tokenized_corpus = [chunk["tokens"] for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)

    def _score_chunks(self, query: str) -> None:
        if self.bm25 == None:
            raise RuntimeError("BM25 Index was not built.")
        scores = self.bm25.get_scores(query=self._tokenize(text=query))
        for chunk, score in zip(self.chunks, scores):
            chunk["score"] = float(score)
        self.chunks.sort(key=lambda chunk: (chunk["score"], chunk["chunk_id"]), reverse=True)

    def _get_chunks(self) -> None:
        if self.connection == None or self.cursor == None:
            raise RuntimeError("Database was not connected.")
        self.cursor.execute(sql.SQL(self.RETRIEVER_SQL))
        self.chunks = []
        for chunk_id, section, title, embedding_text in self.cursor.fetchall():
            self.chunks.append(
                {
                    "chunk_id": chunk_id,
                    "section": section,
                    "title": title,
                    "embedding_text": embedding_text,
                    "tokens": self._tokenize(text=embedding_text),
                    "score": 0.0,
                }
            )

    def _chunks_to_documents(self) -> None:
        self.docs = []
        for chunk in self.chunks:
            # print(f"{chunk['score']:.4f}  [{chunk['section']}] {chunk['title']}")
            self.docs.append(
                Document(
                    page_content=chunk["embedding_text"],
                    metadata={
                        "chunk_id": chunk["chunk_id"],
                        "section": chunk["section"],
                        "title": chunk["title"],
                        "score": chunk["score"],
                    },
                )
            )

    def _get_relevant_documents(self, query: str):
        self.query = query
        self._get_chunks()
        if self.chunks == []:
            raise RuntimeError("Failed to get chunks.")
        self._build_chunk_index()
        self._score_chunks(query=query)
        self._chunks_to_documents()
        return self.docs[: self.k]
