from ragnostics.retrieval.hybrid import HybridRetriever
from ragnostics.graphs.nodes import RAGNodes
from ragnostics.graphs.state import RAGState
from ragnostics.generation.models import build_chat_model

from langgraph.graph import StateGraph, START, END


def build_rag_graph(retriever: HybridRetriever):
    nodes = RAGNodes(retriever=retriever, model=build_chat_model())
    builder = StateGraph(state_schema=RAGState)
    builder.add_node(node="retrieve", action=nodes.retrieve)
    builder.add_node(node="generate", action=nodes.generate)
    builder.add_edge(start_key=START, end_key="retrieve")
    builder.add_edge(start_key="retrieve", end_key="generate")
    builder.add_edge(start_key="generate", end_key=END)
    return builder.compile()
