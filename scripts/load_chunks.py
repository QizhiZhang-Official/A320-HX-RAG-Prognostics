import json
import os
from pathlib import Path

import psycopg
from psycopg import sql

from ragnostics.indexing.embeddings import EmbeddingSpec
from ragnostics.indexing.psql_store import vector_table_name

CHUNKS_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "chunks.jsonl"

INSERT_CHUNK = """
INSERT INTO chunks (
    chunk_id, manual_id, task_id, section, title,
    subtask_id, part, text, embedding_text, char_count
) VALUES (
    %(chunk_id)s, %(manual_id)s, %(task_id)s, %(section)s, %(title)s,
    %(subtask_id)s, %(part)s, %(text)s, %(embedding_text)s, %(char_count)s
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


def to_row(chunk: dict) -> dict:
    metadata = chunk["metadata"]
    return {
        "chunk_id": chunk["chunk_id"],
        "manual_id": metadata["manual_id"],
        "task_id": metadata["task_id"],
        "section": metadata["section"],
        "title": metadata["title"],
        "subtask_id": metadata["subtask_id"],
        "part": metadata.get("part"),
        "text": chunk["text"],
        "embedding_text": chunk["embedding_text"],
        "char_count": chunk["char_count"],
    }


def main() -> None:
    spec = EmbeddingSpec()
    vector_table = vector_table_name(spec.slug)

    lines = CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
    chunks = [json.loads(line) for line in lines if line.strip()]
    rows = [to_row(chunk) for chunk in chunks]

    conn = psycopg.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()

    cur.execute(
        sql.SQL(
            "CREATE TABLE IF NOT EXISTS {} ("
            "  chunk_id TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,"
            "  embedding vector({}) NOT NULL"
            ")"
        ).format(sql.Identifier(vector_table), sql.Literal(spec.dimension))
    )

    cur.executemany(INSERT_CHUNK, rows)

    conn.commit()
    cur.close()
    conn.close()

    print(f"写入 {len(rows)} 行到 chunks 表")
    print(f"向量表已就绪: {vector_table}")


if __name__ == "__main__":
    main()
