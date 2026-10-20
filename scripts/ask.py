from ragnostics.retrieval.dense import DenseRetriever
from ragnostics.retrieval.lexical import LexicalRetriever
from ragnostics.retrieval.hybrid import HybridRetriever
from ragnostics.graphs.rag_agent import build_rag_graph


def main() -> None:
    dense = DenseRetriever(k=20, device="cpu")
    lexical = LexicalRetriever(k=20)
    dense.connect_database()
    lexical.connect_database()
    hybrid = HybridRetriever(dense=dense, lexical=lexical, k=3)
    graph = build_rag_graph(retriever=hybrid)

    result = graph.invoke({"question": "热交换器泄露测试的压力是多少"})
    print(result["answer"])
    print(result["chunk_ids"])

    dense.close_connection()
    lexical.close_connection()


if __name__ == "__main__":
    main()
