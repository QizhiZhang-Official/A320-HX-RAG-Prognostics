from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer


@dataclass(frozen=True, slots=True)
class EmbeddingSpec:
    model_id: str = "BAAI/bge-m3"
    revision: str = "5617a9f61b028005a4858fdac845db406aefb181"
    dimension: int = 1024
    max_seq_length: int = 8192
    query_prefix: str = ""
    passage_prefix: str = ""
    normalize: bool = True

    @property
    def slug(self) -> str:
        slug = f"{self.model_id.split('/')[-1]}-{self.dimension}"
        if self.normalize:
            slug += "-cos"
        else:
            slug += "-l2"
        return slug

    def prefix_passages(self, passages: Sequence[str]) -> list[str]:
        return [self.passage_prefix + passage for passage in passages]

    def prefix_query(self, query: str) -> str:
        return self.query_prefix + query


class PrefixedEmbeddings(Embeddings):
    def __init__(self, inner: Embeddings, spec: EmbeddingSpec) -> None:
        self.inner = inner
        self.spec = spec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents(self.spec.prefix_passages(texts))

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(self.spec.prefix_query(text))


def build_embeddings(spec: EmbeddingSpec, *, device: str) -> Embeddings:
    inner = HuggingFaceEmbeddings(
        model_name=spec.model_id,
        model_kwargs={
            "device": device,
            "revision": spec.revision,
            "local_files_only": True,
        },
        encode_kwargs={"normalize_embeddings": spec.normalize, "batch_size": 16},
    )

    return PrefixedEmbeddings(inner=inner, spec=spec)


def verify_embeddings(embeddings: Embeddings, spec: EmbeddingSpec) -> None:
    probe = "heat exchanger leak test procedure"
    query_vector = embeddings.embed_query(text=probe)
    document_vector = embeddings.embed_documents(texts=[probe])[0]

    if len(query_vector) != spec.dimension:
        raise ValueError(
            f"Dimension not matched: spec.dimension={spec.dimension}, embedded query vector dimension={len(query_vector)}."
        )

    if len(document_vector) != spec.dimension:
        raise ValueError(
            f"Dimension not matched: spec.dimension={spec.dimension}, embedded document vector dimension={len(document_vector)}."
        )

    if spec.normalize:
        query_norm = float(np.linalg.norm(query_vector))
        if abs(query_norm - 1.0) > 1e-3:
            raise ValueError(
                f"Not normalised: spec.normalize=True but the query vector has norm {query_norm:.6f}."
            )

        document_norm = float(np.linalg.norm(document_vector))
        if abs(document_norm - 1.0) > 1e-3:
            raise ValueError(
                f"Not normalised: spec.normalize=True but the document vector has norm {document_norm:.6f}."
            )


def token_lengths(spec: EmbeddingSpec, passages: Sequence[str]) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=spec.model_id,
        revision=spec.revision,
        local_files_only=True,
    )

    return [len(tokenizer.encode(passage)) for passage in spec.prefix_passages(passages)]
