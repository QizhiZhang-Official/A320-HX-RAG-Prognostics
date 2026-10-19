from langchain_core.documents import Document


class FusionModule:
    def __init__(self):
        self._dense_documents: list[Document] = []
        self._lexical_documents: list[Document] = []
        self._unique_chunk_id: list[str] = []
        self._fusion_documents: list[Document] = []

    def _add_documents(self, documents: list[Document], type: str) -> None:
        for doc in documents:
            if doc.metadata["score"] <= 0:
                continue
            if type == "dense":
                self._dense_documents += [doc]
            elif type == "lexical":
                self._lexical_documents += [doc]
            else:
                raise KeyError("Unknown type. Make sure type=dense or type=lexical.")
            if doc.metadata["chunk_id"] not in self._unique_chunk_id:
                self._unique_chunk_id.append(doc.metadata["chunk_id"])

    def _get_rank(self, chunk_id: str, documents: list[Document]) -> int:
        for i, doc in enumerate(documents):
            if doc.metadata["chunk_id"] == chunk_id:
                return i + 1
        return None

    def _calculate_RRF(self, k: int = 60) -> None:
        for chunk_id in self._unique_chunk_id:
            rank_dense = self._get_rank(
                chunk_id=chunk_id, documents=self._dense_documents
            )
            rank_lexical = self._get_rank(
                chunk_id=chunk_id, documents=self._lexical_documents
            )
            if rank_dense != None and rank_lexical != None:
                doc = self._dense_documents[rank_dense - 1]
                dense_contribution = 1 / (k + rank_dense)
                lexical_contribution = 1 / (k + rank_lexical)
            elif rank_dense == None and rank_lexical != None:
                doc = self._lexical_documents[rank_lexical - 1]
                dense_contribution = 0
                lexical_contribution = 1 / (k + rank_lexical)
            elif rank_dense != None and rank_lexical == None:
                doc = self._dense_documents[rank_dense - 1]
                dense_contribution = 1 / (k + rank_dense)
                lexical_contribution = 0
            else:
                raise KeyError(f"Unknown chunk_id: {chunk_id}.")
            assert dense_contribution != None and lexical_contribution != None
            doc.metadata["rrf"] = dense_contribution + lexical_contribution
            self._fusion_documents.append(doc)

    def _sort(self) -> None:
        self._fusion_documents.sort(
            key=lambda doc: (doc.metadata["rrf"], doc.metadata["chunk_id"]),
            reverse=True,
        )

    def _clean(self) -> None:
        self._dense_documents = []
        self._lexical_documents = []
        self._unique_chunk_id = []
        self._fusion_documents = []

    def run(
        self,
        dense_documents: list[Document],
        lexical_documents: list[Document],
        top_k: int = 5,
    ) -> list[Document]:
        self._clean()
        self._add_documents(documents=dense_documents, type="dense")
        self._add_documents(documents=lexical_documents, type="lexical")
        assert self._dense_documents != [] and self._lexical_documents != None
        self._calculate_RRF()
        self._sort()
        if self._fusion_documents == []:
            raise RuntimeError("self.fusion_documents is empty.")
        if top_k > len(self._fusion_documents):
            raise IndexError(
                f"top_k: {top_k} is larger than len(self.fusion_documents): {self._fusion_documents}."
            )
        top_fusion_documents = self._fusion_documents[:top_k]
        self._clean()
        return top_fusion_documents
