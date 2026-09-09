# scripts/test_full_rag.py
import os
import sys

# 确保能导入 src 包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.indexing.vector_index import VectorIndex
from src.rag.indexing.bm25_index import BM25Index
from src.rag.retrieval.hybrid_retriever import HybridRetriever
from src.rag.generation.generator import ManualGenerator
from src.rag.common.io import read_jsonl

def main():
    print("--- 1. Loading Indexes ---")
    chunks = read_jsonl("outputs/chunks.jsonl")
    
    v_index = VectorIndex(persist_directory="outputs/chroma_db")
    b_index = BM25Index()
    b_index.build(chunks)
    
    retriever = HybridRetriever(v_index, b_index, k=60)
    print("✅ Indexes loaded.")

    print("\n--- 2. Initializing LLM Generator ---")
    # 如果显存紧张，可以换成 "Qwen/Qwen2.5-1.5B-Instruct" 先做测试
    generator = ManualGenerator(model_name="Qwen/Qwen2.5-7B-Instruct")
    generator.load_model()

    query = "What should I do if internal leakage is out of tolerance?"
    print(f"\n🔍 User Query: {query}\n")

    print("--- 3. Retrieving Context ---")
    retrieval_results = retriever.search(query, top_k=3) # 取 top 3 给 LLM，避免超出 context window
    print(f"Retrieved {len(retrieval_results)} chunks.\n")

    print("--- 4. Generating Response ---")
    response = generator.generate(query, retrieval_results)
    
    print("="*60)
    print("🤖 LLM RESPONSE:")
    print("="*60)
    print(response)
    print("="*60)

if __name__ == "__main__":
    main()