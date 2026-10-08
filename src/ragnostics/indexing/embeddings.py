# src/ragnostics/indexing/embeddings.py
from dataclasses import dataclass
from collections.abc import Sequence

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings


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


class PrefixedEmbeddings(Embeddings):
    def __init__(self, inner: Embeddings, spec: EmbeddingSpec) -> None:
        self.inner = inner
        self.spec = spec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed_texts = [self.spec.passage_prefix + text for text in texts]
        embedded_text = self.inner.embed_documents(prefixed_texts)

        return embedded_text

    def embed_query(self, text: str) -> list[float]:
        embedded_text = self.inner.embed_query(self.spec.query_prefix + text)

        return embedded_text


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
    