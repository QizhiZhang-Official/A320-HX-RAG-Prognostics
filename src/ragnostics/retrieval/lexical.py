from langchain_core.retrievers import BaseRetriever
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from psycopg import sql

import os
import psycopg


class LexicalRetriever(BaseRetriever):
    k: int = 5
    docs: list[Document] = []
    kl: float = 105
    b: float = 0.75
    connection: None = None
    cursor: None = None
    chunks: None = None
    RETRIEVER_SQL: str = """
        SELECT chunk_id, embedding_text FROM chunks
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
    
    def _get_chunks(self) -> None:
        if self.connection == None or self.cursor == None:
            raise RuntimeError("Database was not connected.")
        self.cursor.execute(sql.SQL(self.RETRIEVER_SQL))
        chunks = self.cursor.fetchall()
        self.chunks = chunks
    
    def _tokenize_chunks(self) -> None:
        