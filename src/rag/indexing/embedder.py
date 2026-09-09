import os
import json
import yaml
from FlagEmbedding import BGEM3FlagModel


class ManualEmbedder:
    """负责加载 bge-m3 模型，并将文本转换为向量。"""

    def __init__(self, model_name: str, use_fp16: bool = True):
        self.model_path = self._get_model_path(model_name)
        self.use_fp16 = use_fp16
        self.model = None

    def _get_model_path(self, model_name: str):
        with open("configs/path.yaml", "r") as f:
            path = yaml.safe_load(f)

        return path["models"][model_name]

    def load_model(self):
        """加载模型到 GPU。"""
        if self.model is not None:
            return

        print(f"Loading model: {self.model_path} ...")
        # 自动检测并使用 GPU
        self.model = BGEM3FlagModel(
            self.model_path, use_fp16=self.use_fp16, device="cuda"
        )
        print("Model loaded successfully.")

    def encode_texts(self, texts, batch_size=256):
        """
        将文本列表转换为稠密向量 (Dense Vectors)。
        :param texts: 字符串列表
        :param batch_size: 批处理大小，4070 可以设大一点（如 256 或 512）
        :return: 二维 numpy 数组，形状为 (len(texts), 1024)
        """
        if self.model is None:
            self.load_model()

        # bge-m3 的 encode 返回一个字典，包含 'dense_vecs', 'lexical_weights', 'colbert_vecs'
        # 我们目前只需要 dense_vecs（稠密向量）
        output = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=8192,  # m3 支持的最大长度
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return output["dense_vecs"]

    def encode_single_text(self, text):
        """编码单条文本，返回一维向量。"""
        vecs = self.encode_texts([text])
        return vecs[0]


if __name__ == "__main__":
    # 简单的本地测试
    embedder = ManualEmbedder()
    embedder.load_model()

    test_texts = [
        "Internal leakage is out of tolerance",
        "The hot air circuit is clogged",
        "How to bake a cake",  # 完全无关的文本
    ]

    vecs = embedder.encode_texts(test_texts)
    print("Vector shape:", vecs.shape)  # 应该输出 (3, 1024)
    print("First 5 dimensions of first vector:", vecs[0][:5])
