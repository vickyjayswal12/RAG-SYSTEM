# RAG Demo — FastAPI + Gemini + pgvector

A minimal, production-shaped Retrieval-Augmented Generation (RAG) system:
upload documents, ask questions, get answers grounded in those documents
with cited sources.

## Architecture

```
Upload file -> extract text -> chunk -> embed (Gemini) -> store in Postgres/pgvector
Ask question -> embed question (Gemini) -> similarity search (pgvector)
             -> stuff top chunks into a prompt -> Gemini generates answer
```

## Stack

- **FastAPI** — REST API framework
- **Google Gemini API** — embeddings (`text-embedding-004`) + generation (`gemini-1.5-flash`)
- **PostgreSQL + pgvector** — stores chunk text and embeddings, does similarity search in SQL
- **Docker Compose** — runs the API and database together with one command
- **LangChain's text splitter** — chunks documents intelligently

See `note.txt` for a full explanation of every library and why it's there —
useful as interview prep / talking points.

## Setup

### 1. Get a Gemini API key

Go to https://aistudio.google.com/app/apikey and create a free key.

### 2. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and paste your key into `GEMINI_API_KEY=`.

### 3. Run with Docker (recommended)

```bash
docker compose up --build
```

This starts two containers:
- `db` — Postgres 16 with the pgvector extension
- `api` — the FastAPI app on http://localhost:8000

Wait for logs to show `Application startup complete.` before calling the API.

### 4. Try it out

Interactive API docs (Swagger UI): **http://localhost:8000/docs**

Or with curl:

```bash
# 1. Ingest the sample document
curl -X POST http://localhost:8000/ingest \
  -F "file=@data/sample_docs/sample.txt"

# 2. Ask a question about it
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the return policy?"}'
```

Expected response shape:

```json
{
  "answer": "Unworn jewellery can be returned within 15 days of purchase...",
  "sources": [
    {"source": "sample.txt", "chunk_index": 1, "similarity_distance": 0.0821}
  ]
}
```

You can also upload your own `.txt`, `.md`, or `.pdf` files through the
`/ingest` endpoint (or the Swagger UI's "Try it out" button).

### 5. Stop everything

```bash
docker compose down          # stops containers, keeps data
docker compose down -v       # stops containers AND deletes stored vectors
```

## Running without Docker (optional)

You'll need a local Postgres with the pgvector extension installed.

```bash
pip install -r requirements.txt
# set POSTGRES_HOST=localhost in .env
uvicorn app.main:app --reload
```

## Project structure

```
rag-system/
├── app/
│   ├── main.py            # FastAPI routes
│   ├── config.py          # env var loading
│   ├── db.py               # Postgres/pgvector connection + queries
│   ├── gemini_client.py    # embedding + generation calls to Gemini
│   ├── chunking.py         # text splitting logic
│   ├── loaders.py          # .txt/.pdf -> plain text
│   ├── ingest.py            # ingest pipeline (chunk -> embed -> store)
│   └── query.py             # query pipeline (embed -> retrieve -> generate)
├── data/sample_docs/       # a sample .txt to test with
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── note.txt                # library-by-library explanation for interview prep
```
