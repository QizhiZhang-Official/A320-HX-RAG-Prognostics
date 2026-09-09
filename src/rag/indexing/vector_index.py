import os
import chromadb
from chromadb.config import Settings
from src.rag.indexing.embedder import ManualEmbedder

class VectorIndex:
    def __init__(self, persist_directory="outputs/chroma_db"):
        self.persist_directory = persist_directory
        # 初始化 ChromaDB 客户端（持久化存储）
        self.client = chromadb.PersistentClient(path=persist_directory)
        # 创建或获取集合，使用余弦相似度
        self.collection = self.client.get_or_create_collection(
            name="manual_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        # 懒加载 embedder
        self.embedder = None

    def _get_embedder(self):
        if self.embedder is None:
            self.embedder = ManualEmbedder()
            self.embedder.load_model()
        return self.embedder

    def build(self, chunks):
        """将 chunks 列表构建为向量索引"""
        embedder = self._get_embedder()
        
        ids = []
        documents = []  # 存储原始 text，用于展示
        embeddings = [] # 存储向量
        metadatas = []  # 存储过滤条件
        
        # 分批处理，防止显存溢出
        batch_size = 32
        # 我们使用 embedding_text 来生成向量，因为它包含章节前缀，区分度更高
        texts_to_embed = [c.get("embedding_text", c["text"]) for c in chunks]
        
        print(f"Encoding {len(chunks)} chunks...")
        for i in range(0, len(texts_to_embed), batch_size):
            batch_text = texts_to_embed[i:i+batch_size]
            # encode_texts 返回 numpy array，需要转为 list
            batch_embeddings = embedder.encode_texts(batch_text).tolist()
            embeddings.extend(batch_embeddings)
            
        for c in chunks:
            ids.append(c["chunk_id"])
            documents.append(c["text"]) # 存原始 text
            
            # 构造 metadata (ChromaDB 要求值必须是 str, int, float, bool)
            meta = {
                "manual_id": c["manual_id"],
                "ata": c["ata"],
                "component": c["component"],
                "section": c["section"],
                # 确保 page 是整数，如果是 None 则给默认值
                "page_start": int(c.get("manual_page_start", 0) or 0),
                "page_end": int(c.get("manual_page_end", 0) or 0)
            }
            metadatas.append(meta)
            
        # 写入 ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"✅ Vector Index built: {len(chunks)} chunks saved to {self.persist_directory}")

    def search(self, query, top_k=3, filters=None):
        """检索向量"""
        embedder = self._get_embedder()
        # 将 query 转为向量
        query_embedding = embedder.encode_single_text(query).tolist()
        
        # 执行检索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters  # 例如: {"section": "TESTING AND FAULT ISOLATION"}
        )
        return results