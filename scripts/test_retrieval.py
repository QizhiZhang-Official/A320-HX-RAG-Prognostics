# scripts/test_retrieval.py
import os
import sys
import json

# 确保能导入 src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.indexing.vector_index import VectorIndex
from src.rag.indexing.bm25_index import BM25Index
from src.rag.common.io import read_jsonl

def main():
    # 1. 加载 chunks
    chunks_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "chunks.jsonl")
    chunks = read_jsonl(chunks_path)
    print(f"Loaded {len(chunks)} chunks.\n")

    # 2. 构建 Vector Index (这会花一点时间，因为要跑 GPU 编码)
    print("--- Building Vector Index ---")
    v_index = VectorIndex(persist_directory="outputs/chroma_db")
    # 如果已经构建过，可以注释掉下面这行 build，直接 search
    v_index.build(chunks) 

    # 3. 构建 BM25 Index
    print("\n--- Building BM25 Index ---")
    b_index = BM25Index()
    b_index.build(chunks)

    # 4. 测试查询
    query = "What should I do if internal leakage is out of tolerance?"
    print(f"\n🔍 Query: {query}\n")

    # Vector Search
    print("--- Vector Search Results ---")
    v_results = v_index.search(query, top_k=2)
    # v_results 结构比较复杂，我们需要解析一下
    for i in range(len(v_results['ids'][0])):
        chunk_id = v_results['ids'][0][i]
        text = v_results['documents'][0][i]
        section = v_results['metadatas'][0][i]['section']
        print(f"[{i+1}] ID: {chunk_id} | Section: {section}")
        print(f"    Text: {text[:100]}...\n")

    # BM25 Search
    print("--- BM25 Search Results ---")
    b_results = b_index.search(query, top_k=2)
    for i, res in enumerate(b_results):
        print(f"[{i+1}] ID: {res['chunk_id']} | Score: {res['score']:.4f} | Section: {res['metadata']['section']}")
        print(f"    Text: {res['text'][:100]}...\n")

if __name__ == "__main__":
    main()