import os
import sys
import json
import numpy as np

# 确保能导入 src 包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.indexing.embedder import ManualEmbedder


def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度（越接近 1 越相似）。"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def main():
    # 1. 初始化并加载模型 (使用你本地下载好的路径)
    embedder = ManualEmbedder(model_name="bge-m3")
    embedder.load_model()

    # 2. 读取真实的 chunks
    chunks_path = os.path.join(
        os.path.dirname(__file__), "..", "outputs", "chunks.jsonl"
    )
    if not os.path.exists(chunks_path):
        print(
            f"Error: {chunks_path} not found. Please run 'python main.py chunk' first."
        )
        return

    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    print(f"Loaded {len(chunks)} chunks.")

    # 3. 挑选有代表性的 chunk 文本 (使用 .lower() 进行大小写不敏感匹配)
    text_leakage_1 = None
    text_repair_tool = None

    for c in chunks:
        # 将换行符替换为空格，并转为小写，避免 PDF 提取时的断行和大小写导致匹配失败
        clean_text = c.get("text", "").replace("\n", " ").lower()

        if "internal leakage" in clean_text and text_leakage_1 is None:
            text_leakage_1 = c["text"]  # 保存原始文本用于 embedding
            print(f"Found leakage chunk: {c['chunk_id']} ({c['section']})")

        if "special tool" in clean_text and text_repair_tool is None:
            text_repair_tool = c["text"]
            print(f"Found special tools chunk: {c['chunk_id']} ({c['section']})")

        # 只要找到这两个代表性的就足够了
        if text_leakage_1 and text_repair_tool:
            break

    # 如果还是没找到，打印前两个 chunk 的前 100 个字符帮助调试
    if not text_leakage_1 or not text_repair_tool:
        print(
            "\n[DEBUG] Could not find representative chunks. Here are the first 2 chunks:"
        )
        for i, c in enumerate(chunks[:2]):
            print(
                f"Chunk {i + 1} ({c['chunk_id']}): {c.get('text', '')[:150].replace(chr(10), ' ')}..."
            )
        return

    # 4. 编码并计算相似度
    print("\n--- Testing Semantic Similarity ---")
    print("Text 1 (Leakage Fault):", text_leakage_1[:80].replace("\n", " "), "...")
    print("Text 2 (Special Tools):", text_repair_tool[:80].replace("\n", " "), "...")

    vec1 = embedder.encode_single_text(text_leakage_1)
    vec2 = embedder.encode_single_text(text_repair_tool)

    sim = cosine_similarity(vec1, vec2)
    print(
        f"\nSimilarity between Leakage and Special Tools (Should be LOW, e.g., < 0.6): {sim:.4f}"
    )

    # 额外测试：找另一个关于 leakage 的 chunk 看看相似度是否高
    text_leakage_2 = None
    for c in chunks:
        clean_text = c.get("text", "").replace("\n", " ").lower()
        if "internal leakage" in clean_text and c["text"] != text_leakage_1:
            text_leakage_2 = c["text"]
            break

    if text_leakage_2:
        vec3 = embedder.encode_single_text(text_leakage_2)
        sim_high = cosine_similarity(vec1, vec3)
        print(
            f"Similarity between Leakage 1 and Leakage 2 (Should be HIGH, e.g., > 0.75): {sim_high:.4f}"
        )

    print("\n✅ Embedder test passed! Your GPU is working correctly with bge-m3.")


if __name__ == "__main__":
    main()
