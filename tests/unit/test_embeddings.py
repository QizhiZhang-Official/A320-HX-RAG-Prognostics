from langchain_core.embeddings import Embeddings
from ragnostics.indexing.embeddings import EmbeddingSpec, PrefixedEmbeddings


class RecordingEmbeddings(Embeddings):
    def __init__(self):
        self.seen_documents = None
        self.seen_query = None

    def embed_documents(self, texts):
        self.seen_documents = texts
        return [[0.0]] * len(texts)

    def embed_query(self, text):
        self.seen_query = text
        return [0.0]


def make_prefixed_embeddings(query_prefix: str = "", passage_prefix: str = ""):
    spec = EmbeddingSpec(query_prefix=query_prefix, passage_prefix=passage_prefix)
    recorder = RecordingEmbeddings()
    return PrefixedEmbeddings(inner=recorder, spec=spec), recorder


def test_query_prefix_is_applied():
    emb, recorder = make_prefixed_embeddings(query_prefix="query: ")
    emb.embed_query("abc")
    assert recorder.seen_query == "query: abc"


def test_documents_prefix_is_applied():
    emb, recorder = make_prefixed_embeddings(passage_prefix="passage: ")
    emb.embed_documents(["abc", "def"])
    assert recorder.seen_documents == ["passage: abc", "passage: def"]


def test_documents_are_not_mutated():
    emb, recorder = make_prefixed_embeddings(passage_prefix="passage: ")
    texts = ["abc", "def"]
    emb.embed_documents(texts)
    assert texts == ["abc", "def"]
