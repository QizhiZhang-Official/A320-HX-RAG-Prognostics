from langchain_core.retrievers import BaseRetriever
from ragnostics.retrieval.fusion import FusionModule


class HybridRetriever(BaseRetriever):
    dense: BaseRetriever
    lexical: BaseRetriever
    k: int = 5

    def _get_relevant_documents(self, query: str):
        dense_documents = self.dense.invoke(query)
        lexical_documents = self.lexical.invoke(query)
        fusion = FusionModule()
        return fusion.run(
            dense_documents=dense_documents,
            lexical_documents=lexical_documents,
            top_k=self.k,
        )
