import os
import pymupdf

from src.rag.common.io import write_jsonl
from src.rag.common.registry import ManualRegistry
from src.rag.common.text_clean import clean_page_text, extract_manual_page


class PageExtractor:
    def __init__(self, registry: ManualRegistry):
        self.registry = registry
        self.results = None

    def extract_one(self, manual):
        pages = []
        doc = pymupdf.open(manual["file_path"])
        for i, page in enumerate(doc):
            raw = page.get_text()
            manual_page = extract_manual_page(raw=raw)
            text = clean_page_text(raw)
            if not text:
                continue
            pages.append(
                {
                    "manual_id": manual["manual_id"],
                    "ata": manual["ata"],
                    "component": manual["component"],
                    "issue": manual["issue"],
                    "page": i + 1,
                    "manual_page": manual_page,
                    "text": text,
                }
            )
        doc.close()
        return pages

    def run(self):
        all_pages = []
        for m in self.registry:
            pages = self.extract_one(m)
            all_pages.extend(pages)
            print("%s: kept %d pages" % (m["manual_id"], len(pages)))
        self.results = all_pages

    def save(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        write_jsonl(self.results, save_path)


if __name__ == "__main__":
    PageExtractor(ManualRegistry()).run()
