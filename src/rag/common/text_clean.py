import re


WATERMARK_RE = re.compile(
    r"(Downloaded the \d{4}-\d{2}-\d{2}"
    r"|by Air china under contract"
    r"|For the airline.{0,3}s own fleet only"
    r"|valid until)",
    re.I,
)

HEADER_RE = re.compile(r"^COMPONENT REPAIR MANUAL\s*753ABC SERIES\s*$", re.I)

FOOTER_RE = re.compile(r"^\d{2}-\d{2}-\d{2}\S*\s+_\S*\s+Page\s+\d+", re.I)

BLANK_RE = re.compile(r"^This page intentionally left blank\.?$", re.I)


def clean_page_text(raw):
    lines = []
    for ln in raw.splitlines():
        s = ln.strip()
        if WATERMARK_RE.search(s):
            continue
        if HEADER_RE.match(s):
            continue
        if FOOTER_RE.match(s):
            continue
        if BLANK_RE.match(s):
            continue
        lines.append(ln)
    return "\n".join(lines).strip()
