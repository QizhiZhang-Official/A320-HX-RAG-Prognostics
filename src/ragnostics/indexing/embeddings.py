import inspect
from langchain_huggingface import HuggingFaceEmbeddings
print(inspect.signature(HuggingFaceEmbeddings.__init__))

from sentence_transformers import SentenceTransformer
m = SentenceTransformer("BAAI/bge-m3")
print("max_seq_length = ", m.max_seq_length)
print("dimension = ", m.get_embedding_dimension)

