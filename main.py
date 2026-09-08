import argparse
import os

from src.rag.common.registry import ManualRegistry
from src.rag.ingestion.inspect_pdf import PDFInspector
from src.rag.ingestion.extract import PageExtractor


def cmd_inspect(args):
    registry = ManualRegistry()
    pdf_inspector = PDFInspector(registry=registry)

    pdf_inspector.run()
    pdf_inspector.save(save_path="outputs/pdf_inspection.csv")


def cmd_extract(args):
    registry = ManualRegistry()
    page_extractor = PageExtractor(registry=registry)

    page_extractor.run()
    page_extractor.save(save_path="outputs/pages.jsonl")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="main.py", description="A320 HeatExchanger RAG pipeline CLI"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_inspect = subparsers.add_parser("inspect")
    p_inspect.set_defaults(func=cmd_inspect)

    p_extract = subparsers.add_parser("extract")
    p_extract.set_defaults(func=cmd_extract)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
