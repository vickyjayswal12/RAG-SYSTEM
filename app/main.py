"""
FastAPI entrypoint.

Endpoints:
- GET  /health          -> liveness check (useful for Docker/AWS health checks)
- POST /ingest          -> upload a .txt/.md/.pdf file to add to the knowledge base
- POST /query           -> ask a question, get an answer grounded in ingested docs
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.db import init_db
from app.loaders import load_text_from_bytes
from app.ingest import ingest_document
from app.query import answer_question

app = FastAPI(
    title="RAG Demo API",
    description="A minimal Retrieval-Augmented Generation system using Gemini + pgvector.",
    version="1.0.0",
)


@app.on_event("startup")
def on_startup():
    # Creates the pgvector extension + documents table if they don't exist yet.
    # Safe to run every time the container starts (CREATE ... IF NOT EXISTS).
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    Accepts a .txt, .md, or .pdf file, splits it into chunks, embeds
    each chunk with Gemini, and stores them in Postgres/pgvector.
    """
    raw_bytes = await file.read()

    try:
        text = load_text_from_bytes(file.filename, raw_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not text.strip():
        raise HTTPException(status_code=400, detail="No extractable text found in file.")

    chunk_count = ingest_document(file.filename, text)
    return {"filename": file.filename, "chunks_stored": chunk_count}


class QueryRequest(BaseModel):
    question: str
    top_k: int = 4


@app.post("/query")
def query(request: QueryRequest):
    """
    Runs the full RAG pipeline for a single question and returns the
    generated answer along with which chunks/sources it was grounded in.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty.")

    result = answer_question(request.question, top_k=request.top_k)
    return result
