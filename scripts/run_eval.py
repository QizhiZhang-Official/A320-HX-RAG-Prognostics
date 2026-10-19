from collections import defaultdict
from pathlib import Path

import yaml

from ragnostics.retrieval.dense import DenseRetriever
from ragnostics.retrieval.hybrid import HybridRetriever
from ragnostics.retrieval.lexical import LexicalRetriever

EVAL_PATH = Path(__file__).resolve().parents[1] / "configs" / "eval.yaml"
RECALL_K = 20
TOP_K = 5
METHODS = ("dense", "lexical", "hybrid")


def is_hit(documents, expected_ids: list[str]) -> bool:
    return any(doc.metadata["chunk_id"] in expected_ids for doc in documents[:TOP_K])


def mark(value: bool) -> str:
    return "✓" if value else "✗"


def main() -> None:
    queries = yaml.safe_load(EVAL_PATH.read_text(encoding="utf-8"))["queries"]

    dense = DenseRetriever(k=RECALL_K, device="cpu")
    dense.connect_database()
    lexical = LexicalRetriever(k=RECALL_K)
    lexical.connect_database()
    hybrid = HybridRetriever(dense=dense, lexical=lexical, k=TOP_K)
    retrievers = {"dense": dense, "lexical": lexical, "hybrid": hybrid}

    rows = []
    for query in queries:
        row = {"id": query["id"], "type": query["type"]}
        for name in METHODS:
            documents = retrievers[name].invoke(query["query"])
            row[name] = is_hit(documents, query["expected_chunk_ids"])
        rows.append(row)

    print(f'{"id":<4} {"type":<12} ' + " ".join(f"{name:>7}" for name in METHODS))
    print("-" * 48)
    for row in rows:
        marks = " ".join(f'{mark(row[name]):>7}' for name in METHODS)
        print(f'{row["id"]:<4} {row["type"]:<12} {marks}')

    print()
    print(f"总体命中率 (top-{TOP_K})")
    for name in METHODS:
        hits = sum(row[name] for row in rows)
        print(f"  {name:<8} {hits}/{len(rows)}")

    print()
    print("按查询类型")
    by_type = defaultdict(list)
    for row in rows:
        by_type[row["type"]].append(row)
    for query_type, group in by_type.items():
        parts = [f"{name} {sum(row[name] for row in group)}/{len(group)}" for name in METHODS]
        print(f"  {query_type:<12} " + "   ".join(parts))

    dense.close_connection()
    lexical.close_connection()


if __name__ == "__main__":
    main()
