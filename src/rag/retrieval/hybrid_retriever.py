# src/rag/retrieval/hybrid_retriever.py
import os
from src.rag.indexing.vector_index import VectorIndex
from src.rag.indexing.bm25_index import BM25Index

class HybridRetriever:
    """结合 Vector 和 BM25 的混合检索器，使用 RRF 算法融合结果。"""

    def __init__(self, vector_index, bm25_index, k=60):
        """
        :param vector_index: 已构建的 VectorIndex 实例
        :param bm25_index: 已构建的 BM25Index 实例
        :param k: RRF 算法的常数，通常设为 60
        """
        self.vector_index = vector_index
        self.bm25_index = bm25_index
        self.k = k

    def search(self, query, top_k=3):
        """执行混合检索"""
        # 1. 分别获取两个索引的结果 (多取一些用于融合)
        # 注意：这里 top_k 可以设大一点，比如 10，融合后再取最终的 top_k
        fetch_k = max(top_k * 2, 10) 
        
        v_results = self.vector_index.search(query, top_k=fetch_k)
        b_results = self.bm25_index.search(query, top_k=fetch_k)

        # 2. 提取 ID 和 排名
        v_ranks = {doc_id: rank + 1 for rank, doc_id in enumerate(v_results['ids'][0])}
        b_ranks = {res['chunk_id']: rank + 1 for rank, res in enumerate(b_results)}

        # 3. 收集所有出现过的 chunk_id
        all_ids = set(list(v_ranks.keys()) + list(b_ranks.keys()))

        # 4. 计算 RRF 分数
        rrf_scores = {}
        for doc_id in all_ids:
            score = 0.0
            if doc_id in v_ranks:
                score += 1.0 / (self.k + v_ranks[doc_id])
            if doc_id in b_ranks:
                score += 1.0 / (self.k + b_ranks[doc_id])
            rrf_scores[doc_id] = score

        # 5. 按 RRF 分数降序排序
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # 6. 组装最终结果 (取 top_k)
        final_results = []
        # 我们需要一个快速查找 chunk 内容的方法，这里简单遍历 bm25 的映射
        # 实际项目中可以优化，但 MVP 阶段这样写最清晰
        id_to_chunk = self.bm25_index.id_to_chunk 
        
        for doc_id in sorted_ids[:top_k]:
            if doc_id in id_to_chunk:
                chunk = id_to_chunk[doc_id]
                final_results.append({
                    "chunk_id": doc_id,
                    "rrf_score": rrf_scores[doc_id],
                    "section": chunk.get("section", "UNKNOWN"),
                    "manual_page_start": chunk.get("manual_page_start", 0),
                    "manual_page_end": chunk.get("manual_page_end", 0),
                    "text": chunk["text"]
                })

        return final_results