import json

from ragnostics.indexing import embeddings


def test_token_lengths():
    spec = embeddings.EmbeddingSpec()

    with open("artifacts/chunks.jsonl", "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
        embedded_texts = [chunk["embedding_text"] for chunk in chunks]

    token_lengths = embeddings.token_lengths(spec=spec, passages=embedded_texts)

    assert max(token_lengths) <= spec.max_seq_length
