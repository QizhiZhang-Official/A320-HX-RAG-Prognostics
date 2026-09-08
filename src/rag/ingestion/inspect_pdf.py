import os
import csv
import pymupdf

from src.rag.common.registry import ManualRegistry


class PDFInspector:
    def __init__(self, registry: ManualRegistry):
        self.registry = registry
        self.results = None

    def inspect_one(self, file_path: str):
        doc = pymupdf.open(file_path)
        counts = [len(page.get_text().strip()) for page in doc]
        n_pages = doc.page_count
        doc.close()
        avg = sum(counts) / max(n_pages, 1)
        return {
            "pages": n_pages,
            "avg_chars_per_page": round(avg, 1),
            "is_scanned": avg < 50,
        }

    def run(self):
        rows = []
        for manual in self.registry:
            statics = self.inspect_one(file_path=manual["file_path"])
            row = {"manual_id": manual["manual_id"]}
            row.update(statics)
            rows.append(row)
        self.results = rows

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)


if __name__ == "__main__":
    registry = ManualRegistry()
    pdf_inspector = PDFInspector(registry=registry)

    pdf_inspector.run()
    pdf_inspector.save(save_path="outputs/pdf_inspection.csv")
