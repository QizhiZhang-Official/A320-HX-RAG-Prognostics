import json
import os
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.rows import dict_row

from ragnostics.indexing.embeddings import EmbeddingSpec, build_embeddings


def vector_table_name(slug: str) -> str:
    return "chunk_vectors_" + slug.replace("-", "_")


def distance_operator_class(spec: EmbeddingSpec) -> str:
    return "vector_cosine_ops" if spec.normalize else "vector_l2_ops"


class PostgresChunkStore:
    def __init__(self) -> None:
        self.INSERT_SQL = """
            INSERT INTO chunks (
                chunk_id, manual_id, task_id, section, title,
                subtask_id, part, text, embedding_text, char_count
            )
            VALUES (
                %(chunk_id)s, %(manual_id)s, %(task_id)s, %(section)s,
                %(title)s, %(subtask_id)s, %(part)s, %(text)s,
                %(embedding_text)s, %(char_count)s
            )
            ON CONFLICT (chunk_id) DO UPDATE SET
                manual_id = EXCLUDED.manual_id,
                task_id = EXCLUDED.task_id,
                section = EXCLUDED.section,
                title = EXCLUDED.title,
                subtask_id = EXCLUDED.subtask_id,
                part = EXCLUDED.part,
                text = EXCLUDED.text,
                embedding_text = EXCLUDED.embedding_text,
                char_count = EXCLUDED.char_count
        """
        self.chunks = None
        self.raws = None

    def _load_chunks_jsonl(self, chunks_path: str) -> None:
        lines = Path(chunks_path).read_text(encoding="utf-8").splitlines()
        chunks = [json.loads(line) for line in lines if line.strip()]
        self.chunks = chunks

    def _chunks_to_raws(self) -> None:
        raws = []
        for chunk in self.chunks:
            raw = {
                "chunk_id": chunk["chunk_id"],
                "manual_id": chunk["metadata"]["manual_id"],
                "task_id": chunk["metadata"]["task_id"],
                "section": chunk["metadata"]["section"],
                "title": chunk["metadata"]["title"],
                "subtask_id": chunk["metadata"]["subtask_id"],
                "part": chunk["metadata"].get("part"),
                "text": chunk["text"],
                "embedding_text": chunk["embedding_text"],
                "char_count": chunk["char_count"],
            }
            raws.append(raw)
        self.raws = raws

    def _insert_chunk_to_psql(self) -> None:
        connection = psycopg.connect(os.environ["DATABASE_URL"])
        cursor = connection.cursor()
        for raw in self.raws:
            cursor.execute(self.INSERT_SQL, raw)
        connection.commit()
        cursor.close()
        connection.close()

    def run(self, chunks_path: str) -> None:
        self._load_chunks_jsonl(chunks_path=chunks_path)
        self._chunks_to_raws()
        self._insert_chunk_to_psql()


class PostgresVectorStore:
    def __init__(self) -> None:
        self.spec = EmbeddingSpec()
        self.chunks = None
        self.vectors = None

        self.CREATE_TABLE_SQL = """
            CREATE TABLE IF NOT EXISTS {} (
                chunk_id TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                embedding VECTOR({}) NOT NULL
            )
        """
        self.INSERT_VECTOR_SQL = """
            INSERT INTO {} (chunk_id, embedding) VALUES (
                %(chunk_id)s, %(vector)s
            ) ON CONFLICT (chunk_id) DO UPDATE SET
                embedding = EXCLUDED.embedding
        """
        self.CREATE_INDEX_SQL = """
            CREATE INDEX IF NOT EXISTS {} ON {}
            USING hnsw (embedding {})
            WITH (m = 16, ef_construction = 64)
        """
        self.WRITE_MANIFEST_SQL = """
            INSERT INTO index_manifests (slug, manifest)
            VALUES (%(slug)s, %(manifest)s)
            ON CONFLICT (slug) DO UPDATE
                manifest = EXCLUDED.manifest,
                built_at = now()
        """

    def create_vector_table(self) -> None:
        table_name = vector_table_name(slug=self.spec.slug)
        connection = psycopg.connect(os.environ["DATABASE_URL"])
        cursor = connection.cursor()
        cursor.execute(
            sql.SQL(self.CREATE_TABLE_SQL).format(
                sql.Identifier(table_name), sql.Literal(self.spec.dimension)
            )
        )
        connection.commit()
        cursor.close()
        connection.close()

    def load_chunks(self) -> None:
        connection = psycopg.connect(os.environ["DATABASE_URL"])
        cursor = connection.cursor(row_factory=dict_row)
        cursor.execute("SELECT chunk_id, embedding_text FROM chunks ORDER BY chunk_id")
        rows = cursor.fetchall()
        self.chunks = rows
        cursor.close()
        connection.close()

    def embed_chunks_to_vectors(self) -> None:
        embeddings = build_embeddings(spec=self.spec, device="cpu")
        embedding_texts = [chunk["embedding_text"] for chunk in self.chunks]
        vectors = embeddings.embed_documents(texts=embedding_texts)
        self.vectors = vectors

    def insert_vectors(self) -> None:
        table_name = vector_table_name(slug=self.spec.slug)
        chunk_ids = [chunk["chunk_id"] for chunk in self.chunks]
        connection = psycopg.connect(os.environ["DATABASE_URL"])
        register_vector(connection)
        cursor = connection.cursor()
        assert len(chunk_ids) == len(self.vectors)
        for i, vector in enumerate(self.vectors):
            chunk_id = chunk_ids[i]
            cursor.execute(
                sql.SQL(self.INSERT_VECTOR_SQL).format(sql.Identifier(table_name)),
                {"chunk_id": chunk_id, "vector": vector},
            )
        connection.commit()
        cursor.close()
        connection.close()

    def create_vector_index(self) -> None:
        table_name = vector_table_name(slug=self.spec.slug)
        index_name = table_name + "_hnsw"
        connection = psycopg.connect(os.environ["DATABASE_URL"])
        cursor = connection.cursor()
        cursor.execute(
            sql.SQL(self.CREATE_INDEX_SQL).format(
                sql.Identifier(index_name),
                sql.Identifier(index_name),
                sql.Identifier(distance_operator_class(spec=self.spec)),
            )
        )
        connection.commit()
        cursor.close()
        connection.close()
    
    # def _generate_manifest(self) -> dict:
                
        
    #     manifest = {
    #         'slug': self.spec.slug,
    #         'model_id': self.spec.model_id,
    #         'revision': self.spec.revision,
    #         'dimension': self.spec.dimension,
    #         'distance': self.spec.distance,
    #         'max_seq_length': self.spec.max_seq_length,
    #         'query_prefix': self.spec.query_prefix,
    #         'passage_prefix': self.spec.passage_prefix,
    #         'chunk_count': len(self.chunks),
    #         'chunker_hash':
    #     }
    
    # def write_manifest(self) -> None:
        
    #     connection = psycopg.connect(os.environ["DATABASE_URL"])
    #     cursor = connection.cursor()
        

    def run(self) -> None:
        self.create_vector_table()
        self.load_chunks()
        self.embed_chunks_to_vectors()
        self.insert_vectors()
        self.create_vector_index()
