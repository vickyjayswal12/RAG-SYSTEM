"""
Turns an uploaded file's raw bytes into plain text, based on extension.

Kept separate from ingest.py so adding a new file type (e.g. .docx)
later is a one-function change here, not a rewrite of the ingest route.
"""

import io
from pypdf import PdfReader


def load_text_from_bytes(filename: str, raw_bytes: bytes) -> str:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext == "pdf":
        reader = PdfReader(io.BytesIO(raw_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)

    if ext in ("txt", "md"):
        return raw_bytes.decode("utf-8", errors="ignore")

    raise ValueError(f"Unsupported file type: .{ext}. Use .txt, .md, or .pdf")
