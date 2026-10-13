from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
import psycopg
import os
from pgvector.psycopg import register_vector
from pgvector import Vector
from psycopg import sql
from ragnostics.indexing.embeddings import EmbeddingSpec, build_embeddings
from ragnostics.indexing.psql_store import vector_table_name


class DenseRetriever(BaseRetriever):
    k: int = 5
    docs: list[Document] = []
    device: str = "cpu"
    connection: None = None
    cursor: None = None
    embeddings: None = None
    table_name: None = None
    RETRIEVER_SQL: str = """
        SELECT c.chunk_id, c.section, c.title,
        c. text, 1 - (v.embedding <=> %(query_vector)s) AS score
        FROM {} v JOIN chunks c USING (chunk_id)
        ORDER BY score DESC
        LIMIT %(k)s
    """

    def connect_database(self) -> None:
        connection = psycopg.connect(os.environ["DATABASE_URL"])
        register_vector(connection)
        cursor = connection.cursor()
        self.connection = connection
        self.cursor = cursor

    def close_connection(self) -> None:
        self.cursor.close()
        self.cursor = None
        self.connection.close()
        self.connection = None

    def _build_embeddings(self) -> None:
        if self.embeddings == None:
            spec = EmbeddingSpec()
            table_name = vector_table_name(slug=spec.slug)
            embeddings = build_embeddings(spec=spec, device=self.device)
            self.table_name = table_name
            self.embeddings = embeddings

    def _query_to_vector(self, query: str) -> Vector:
        if self.embeddings == None:
            raise RuntimeError("Embedding Module was not Built.")
        return Vector(self.embeddings.embed_query(query))

    def _get_relevant_documents(self, query: str):
        self._build_embeddings()
        query_vector = self._query_to_vector(query=query)
        if self.connection == None or self.cursor == None:
            raise RuntimeError("Database was not connected.")
        self.cursor.execute(
            sql.SQL(self.RETRIEVER_SQL).format(sql.Identifier(self.table_name)),
            {"query_vector": query_vector, "k": self.k},
        )
        for chunk_id, section, title, text, score in self.cursor.fetchall():
            print(f"{score:.4f}  [{section}] {title}")
            self.docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "chunk_id": chunk_id,
                        "section": section,
                        "title": title,
                        "score": score,
                    },
                )
            )
        return self.docs[: self.k]
