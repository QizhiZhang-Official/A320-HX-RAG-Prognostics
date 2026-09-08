# /src/rag/common/text_clean.py
import re


WATERMARK_RE = re.compile(
    r"(Downloaded the \d{4}-\d{2}-\d{2}"
    r"|by Air china under contract"
    r"|For the airline.{0,3}s own fleet only"
    r"|valid until)",
    re.I,
)

HEADER_LINE_RE = re.compile(
    r"(^COMPONENT REPAIR MANUAL$"
    r"|^\d{3}ABC SERIES$)",
    re.I,
)

FOOTER_LINE_RE = re.compile(
    r"(^\d{2}-\d{2}-\d{2}R?M?$"
    r"|^_[A-Z0-9 ]*$"
    r"|^Page \d+$"
    r"|^[a-zA-Z]?[A-Z][a-z]{2} \d{1,2}/\d{2}$)",
    re.I,
)

BLANK_LINE_RE = re.compile(r"^This page intentionally left blank\.?$", re.I)

MANUAL_PAGE_RE = re.compile(r"^Page\s+(\d+)$", re.I)

def clean_page_text(raw):
    lines = []
    for line in raw.splitlines():
        s = line.strip()
        if WATERMARK_RE.search(s):
            continue
        if HEADER_LINE_RE.match(s):
            continue
        if FOOTER_LINE_RE.match(s):
            continue
        if BLANK_LINE_RE.match(s):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def extract_manual_page(raw):
    for line in raw.splitlines():
        manual = MANUAL_PAGE_RE.match(line.strip())
        if manual:
            return int(manual.group(1))
    return None