import argparse
from pathlib import Path

from ragnostics.ingestion.chunking import ChunkStats, TaskChunker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split ingested manual tasks into retrieval chunks."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=TaskChunker.DEFAULT_INPUT_PATH,
        help="JSONL file with ingested tasks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TaskChunker.DEFAULT_OUTPUT_PATH,
        help="JSONL file to write chunks to.",
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=TaskChunker.DEFAULT_MAX_CHUNK_CHARS,
        help="Maximum characters before a section falls back to fixed-size splitting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunker = TaskChunker(
        input_path=args.input,
        output_path=args.output,
        max_chunk_chars=args.max_chunk_chars,
    )
    chunks = chunker.run()
    print(f"{args.output} {ChunkStats.from_chunks(chunks)}")


if __name__ == "__main__":
    main()
