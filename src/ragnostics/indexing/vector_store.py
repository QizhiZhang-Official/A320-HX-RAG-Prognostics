from psycopg import sql
from ragnostics.indexing.embeddings import EmbeddingSpec


def vector_table_name(slug: str) -> str:
    return 'chunk_vectors_' + slug.replace('-', '_')

def distance_operator_class(spec: EmbeddingSpec) -> str:
    return 'vector_cosine_ops' if spec.normalize else 'vector_l2_ops'

class PostgresVectorStore:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
    
    def apply_schema(self) -> None:
        