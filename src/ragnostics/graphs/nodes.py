from typing import Any

from langchain_openai import ChatOpenAI

from ragnostics.retrieval.hybrid import HybridRetriever
from ragnostics.graphs.state import RAGState
from ragnostics.generation.prompts import load_prompt


class RAGNodes:
    def __init__(self, retriever: HybridRetriever, model: ChatOpenAI) -> None:
        self.retriever = retriever
        self.model = model

    def retrieve(self, state: RAGState) -> dict[str, Any]:
        documents = self.retriever.invoke(state["question"])
        chunk_ids = []
        contexts = ""
        for doc in documents:
            chunk_ids.append(doc.metadata['chunk_id'])
            contexts += f"chunk_id: {doc.metadata['chunk_id']}\n"
            contexts += f"section: {doc.metadata['section']}\n"
            contexts += f"title: {doc.metadata['title']}\n"
            contexts += f"page_content: {doc.page_content}\n"
        return {
            "chunk_ids": chunk_ids,
            "contexts": "\n\n" + contexts,
        }

    def generate(self, state: RAGState) -> dict[str, Any]:
        message = [
            ("system", load_prompt(file_name="rag_answer.md")),
            ("human", f"问题：{state['question']}\n\n上下文：\n{state['contexts']}"),
        ]
        response = self.model.invoke(input=message)
        return {'answer': response.content}
