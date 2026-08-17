"""
The "build the knowledge base" side of RAG:
load file -> chunk -> embed each chunk -> store (text + vector) in Postgres.
"""

from app.chunking import chunk_text
from app.gemini_client import embed_text
from app.db import insert_chunks


def ingest_document(filename: str, text: str) -> int:
    chunks = chunk_text(text)

    rows = []
    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk, task_type="retrieval_document")
        rows.append((filename, i, chunk, embedding))

    insert_chunks(rows)
    return len(chunks)
