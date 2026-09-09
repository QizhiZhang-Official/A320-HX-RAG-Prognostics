# scripts/test_hybrid_retrieval.py
import os
import sys

# 确保能导入 src 包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.indexing.vector_index import VectorIndex
from src.rag.indexing.bm25_index import BM25Index
from src.rag.retrieval.hybrid_retriever import HybridRetriever

def main():
    print("--- Loading Indexes ---")
    # 加载已构建好的索引
    v_index = VectorIndex(persist_directory="outputs/chroma_db")
    
    # BM25 需要重新加载 chunks 来构建，或者你可以把 BM25 的构建结果也持久化
    # 这里为了简单，我们重新读取 chunks 构建 BM25
    from src.rag.common.io import read_jsonl
    chunks = read_jsonl("outputs/chunks.jsonl")
    b_index = BM25Index()
    b_index.build(chunks)

    retriever = HybridRetriever(v_index, b_index, k=60)

    query = "What should I do if internal leakage is out of tolerance?"
    print(f"\n🔍 Query: {query}\n")

    results = retriever.search(query, top_k=4)

    print("--- Hybrid Search Results (RRF Fused) ---")
    for i, res in enumerate(results):
        print(f"[{i+1}] ID: {res['chunk_id']} | RRF Score: {res['rrf_score']:.4f}")
        print(f"    Section: {res['section']} | Pages: {res['manual_page_start']}-{res['manual_page_end']}")
        print(f"    Text Preview: {res['text'][:100].replace(chr(10), ' ')}...")
        print("-" * 50)

if __name__ == "__main__":
    main()