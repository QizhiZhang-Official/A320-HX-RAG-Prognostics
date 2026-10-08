import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from langchain_docling.loader import DoclingLoader

FILE_PATH = ["data/manuals/21-52-11-R.pdf", "data/manuals/21-52-12-R.pdf"]

loader = DoclingLoader(file_path=FILE_PATH)
docs = loader.load()

with open('content.txt', 'w', encoding='utf-8') as f:
    for doc in docs:
        f.write(doc.page_content)
        f.write('\n')
