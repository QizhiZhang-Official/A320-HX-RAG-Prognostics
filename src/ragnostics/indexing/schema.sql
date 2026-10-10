CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    manual_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    section TEXT NOT NULL,
    title TEXT NOT NULL,
    subtask_id TEXT,
    part INTEGER,
    text TEXT NOT NULL,
    embedding_text TEXT NOT NULL,
    char_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS index_manifests (
    slug TEXT PRIMARY KEY,
    manifest JSONB NOT NULL,
    built_at TIMESTAMPTZ NOT NULL DEFAULT now()
);