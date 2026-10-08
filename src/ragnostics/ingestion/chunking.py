import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    metadata: dict[str, Any]
    text: str
    embedding_text: str

    @property
    def char_count(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "metadata": self.metadata,
            "text": self.text,
            "embedding_text": self.embedding_text,
            "char_count": self.char_count,
        }


@dataclass(frozen=True, slots=True)
class ChunkStats:
    count: int
    minimum: int
    median: int
    maximum: int

    @classmethod
    def from_chunks(cls, chunks: Sequence[Chunk]) -> ChunkStats:
        lengths = sorted(chunk.char_count for chunk in chunks)
        if not lengths:
            raise ValueError("cannot summarize an empty chunk sequence")
        return cls(
            count=len(lengths),
            minimum=lengths[0],
            median=lengths[len(lengths) // 2],
            maximum=lengths[-1],
        )

    def __str__(self) -> str:
        return f"chunks={self.count} min={self.minimum} median={self.median} max={self.maximum}"


class TaskChunker:
    SUBTASK_PATTERN = re.compile(r"^(SUBTASK \S+)\s*$", re.MULTILINE)
    METADATA_FIELDS = ("manual_id", "task_id", "section", "title")
    OVERVIEW_LABEL = "OVERVIEW"
    DEFAULT_INPUT_PATH = Path("artifacts/pages.jsonl")
    DEFAULT_OUTPUT_PATH = Path("artifacts/chunks.jsonl")
    DEFAULT_MAX_CHUNK_CHARS = 2000

    def __init__(
        self,
        input_path: str | Path = DEFAULT_INPUT_PATH,
        output_path: str | Path = DEFAULT_OUTPUT_PATH,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
        fallback_chunk_size: int = 1500,
        fallback_chunk_overlap: int = 150,
        fallback_separators: Sequence[str] = ("\n\n", "\n", " ", ""),
    ) -> None:
        self._input_path = Path(input_path)
        self._output_path = Path(output_path)
        self._max_chunk_chars = max_chunk_chars
        self._fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=fallback_chunk_size,
            chunk_overlap=fallback_chunk_overlap,
            separators=list(fallback_separators),
        )
        self._next_chunk_number = 1

    def run(self) -> list[Chunk]:
        chunks = self.chunk_tasks(self.load_tasks())
        self.write_chunks(chunks)
        return chunks

    def load_tasks(self) -> list[dict[str, Any]]:
        lines = self._input_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def write_chunks(self, chunks: Iterable[Chunk]) -> None:
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._output_path.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                line = json.dumps(chunk.to_dict(), ensure_ascii=False)
                handle.write(line + "\n")

    def chunk_tasks(self, tasks: Iterable[Mapping[str, Any]]) -> list[Chunk]:
        self._next_chunk_number = 1
        chunks: list[Chunk] = []
        for task in tasks:
            chunks.extend(self.chunk_task(task))
        return chunks

    def chunk_task(self, task: Mapping[str, Any]) -> list[Chunk]:
        text = task["text"]
        metadata = self._task_metadata(task)
        matches = list(self.SUBTASK_PATTERN.finditer(text))

        if not matches:
            return self._emit(metadata, None, text)

        chunks: list[Chunk] = []
        header = text[: matches[0].start()].strip()
        if header:
            chunks.extend(self._emit(metadata, None, header))

        for index, match in enumerate(matches):
            if index + 1 < len(matches):
                segment_end = matches[index + 1].start()
            else:
                segment_end = len(text)
            segment = text[match.start() : segment_end].strip()
            chunks.extend(self._emit(metadata, match.group(1), segment))

        return chunks

    def _task_metadata(self, task: Mapping[str, Any]) -> dict[str, Any]:
        return {field: task[field] for field in self.METADATA_FIELDS}

    def _emit(self, metadata: Mapping[str, Any], subtask_id: str | None, text: str) -> list[Chunk]:
        if len(text) <= self._max_chunk_chars:
            return [self._build_chunk(metadata, subtask_id, text)]

        parts = self._fallback_splitter.split_text(text)
        return [
            self._build_chunk(metadata, subtask_id, part, part_number=number)
            for number, part in enumerate(parts, start=1)
        ]

    def _build_chunk(
        self,
        metadata: Mapping[str, Any],
        subtask_id: str | None,
        text: str,
        part_number: int | None = None,
    ) -> Chunk:
        chunk_metadata = dict(metadata)
        chunk_metadata["subtask_id"] = subtask_id
        if part_number is not None:
            chunk_metadata["part"] = part_number

        label = subtask_id or self.OVERVIEW_LABEL
        prefix = f"[{metadata['manual_id']} | {metadata['section']} | {metadata['title']} | {label}]"
        return Chunk(
            chunk_id=self._take_chunk_id(),
            metadata=chunk_metadata,
            text=text,
            embedding_text=f"{prefix}\n{text}",
        )

    def _take_chunk_id(self) -> str:
        chunk_id = f"c_{self._next_chunk_number:04d}"
        self._next_chunk_number += 1
        return chunk_id
