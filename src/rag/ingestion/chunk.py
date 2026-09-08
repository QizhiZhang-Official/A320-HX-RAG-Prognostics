import os
import re

from src.rag.common.io import read_jsonl, write_jsonl

SECTION_KEYWORDS = [
    "DESCRIPTION AND OPERATION",
    "TESTING AND FAULT ISOLATION",
    "DISASSEMBLY",
    "CLEANING",
    "INSPECTION/CHECK",
    "INSPECTION",
    "CHECK",
    "REPAIR",
    "ASSEMBLY",
    "FITS AND CLEARANCES",
    "SPECIAL TOOLS, FIXTURES, EQUIPMENT, AND CONSUMABLES",
    "SPECIAL TOOLS, FIXTURES, EQUIPMENT, AND CONSUMABLE",
    "ILLUSTRATED PARTS LIST",
]

SECTION_RE = re.compile(
    r"^(" + "|".join(re.escape(k) for k in SECTION_KEYWORDS) + r")\s*[A-Z0-9 \-/&,\.]*$"
)

VALUE_SECTIONS = {
    "TESTING AND FAULT ISOLATION",
    "REPAIR",
    "SPECIAL TOOLS, FIXTURES, EQUIPMENT, AND CONSUMABLES",
    "SPECIAL TOOLS, FIXTURES, EQUIPMENT, AND CONSUMABLE",
}

SECTION_BLOCK_MAP = {
    "TESTING AND FAULT ISOLATION": 1000,
    "REPAIR": 6000,
    "SPECIAL TOOLS, FIXTURES, EQUIPMENT, AND CONSUMABLES": 9000,
    "SPECIAL TOOLS, FIXTURES, EQUIPMENT, AND CONSUMABLE": 9000,
    "ILLUSTRATED PARTS LIST": 10000,
}

BODY_MIN_MANUAL_PAGE = 1000


def _get_block(page):
    if page is None:
        return None
    for block_start in [1000, 6000, 9000, 10000]:
        if block_start <= page < block_start + 1000:
            return block_start
    return None


class ChunkBuilder:
    def __init__(
        self,
        pages_jsonl=os.path.join("outputs", "pages.jsonl"),
        output_jsonl=os.path.join("outputs", "chunks.jsonl"),
        max_chars=1200,
        min_chars=80,
    ):
        self.pages_jsonl = pages_jsonl
        self.output_jsonl = output_jsonl
        self.max_chars = max_chars
        self.min_chars = min_chars

    def build(self):
        pages = read_jsonl(self.pages_jsonl)
        body_pages = []
        for p in pages:
            mp = p.get("manual_page")
            if mp is not None and mp >= BODY_MIN_MANUAL_PAGE:
                body_pages.append(p)

        by_manual = {}
        for p in body_pages:
            by_manual.setdefault(p["manual_id"], []).append(p)

        chunks = []
        for manual_id in sorted(by_manual.keys()):
            first = by_manual[manual_id][0]
            meta = {
                "manual_id": manual_id,
                "ata": first["ata"],
                "component": first["component"],
                "issue": first["issue"],
            }
            chunks.extend(self._split_manual(by_manual[manual_id], meta))

        write_jsonl(chunks, self.output_jsonl)
        print("body pages: %d, total chunks: %d" % (len(body_pages), len(chunks)))
        return chunks

    def _split_manual(self, pages, meta):
        self._chunks = []
        self._section = "FRONT"
        self._buf = []
        self._start_page = None
        self._cid = 0

        for p in pages:
            mp = p.get("manual_page")
            current_block = _get_block(mp)

            for line in p["text"].splitlines():
                s = line.strip()
                m = SECTION_RE.match(s)
                if m:
                    matched_section = m.group(1)
                    matched_block = SECTION_BLOCK_MAP.get(matched_section)

                    if (
                        matched_block is not None
                        and current_block is not None
                        and matched_block != current_block
                    ):
                        self._buf.append(s)
                        continue

                    self._flush(mp, meta)
                    self._section = matched_section
                    self._start_page = mp
                    continue

                self._buf.append(s)
                if sum(len(x) for x in self._buf) > self.max_chars:
                    self._flush(mp, meta)
                    self._start_page = mp
            if self._start_page is None:
                self._start_page = mp

        last = pages[-1].get("manual_page") if pages else None
        self._flush(last, meta)
        return self._chunks

    def _flush(self, end_page, meta):
        text = "\n".join(self._buf).strip()
        self._buf = []
        if self._section not in VALUE_SECTIONS:
            self._start_page = None
            return
        if len(text) < self.min_chars:
            self._start_page = None
            return
        self._cid += 1
        chunk = {
            "chunk_id": "%s-c%04d" % (meta["manual_id"], self._cid),
            "manual_id": meta["manual_id"],
            "ata": meta["ata"],
            "component": meta["component"],
            "issue": meta["issue"],
            "section": self._section,
            "manual_page_start": self._start_page,
            "manual_page_end": end_page,
            "text": text,
        }
        chunk["embedding_text"] = "%s | ATA %s | %s | Page %s-%s | %s" % (
            meta["component"],
            meta["ata"],
            self._section,
            self._start_page,
            end_page,
            text,
        )
        self._chunks.append(chunk)
        self._start_page = None


if __name__ == "__main__":
    ChunkBuilder().build()
