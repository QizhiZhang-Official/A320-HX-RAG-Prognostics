from rank_bm25 import BM25Okapi
import re

class BM25Index:
    def __init__(self):
        self.bm25 = None
        self.corpus = []       # 分词后的文本列表
        self.chunk_ids = []    # 对应的 chunk_id
        self.id_to_chunk = {}  # 快速查找原始 chunk 信息

    def _tokenize(self, text):
        """简单的英文分词：转小写，去标点，按空格分割"""
        text = text.lower()
        # 去除非字母数字字符（保留空格）
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

    def build(self, chunks):
        """构建 BM25 索引"""
        self.corpus = []
        self.chunk_ids = []
        
        for c in chunks:
            self.chunk_ids.append(c["chunk_id"])
            # BM25 使用原始 text 字段，因为 embedding_text 里的前缀可能会干扰关键词匹配
            self.corpus.append(self._tokenize(c["text"]))
            self.id_to_chunk[c["chunk_id"]] = c
            
        self.bm25 = BM25Okapi(self.corpus)
        print(f"✅ BM25 Index built: {len(chunks)} chunks indexed.")

    def search(self, query, top_k=3):
        """关键词检索"""
        tokenized_query = self._tokenize(query)
        # 获取所有 chunk 的 BM25 分数
        scores = self.bm25.get_scores(tokenized_query)
        
        # 按分数降序排序，取 top_k
        # 使用 enumerate 获取原始索引
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            chunk_id = self.chunk_ids[idx]
            chunk_info = self.id_to_chunk[chunk_id]
            results.append({
                "chunk_id": chunk_id,
                "score": float(scores[idx]), # 转为 float 方便序列化
                "text": chunk_info["text"],
                "metadata": {
                    "section": chunk_info["section"],
                    "manual_id": chunk_info["manual_id"]
                }
            })
        return results