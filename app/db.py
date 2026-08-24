"""
Postgres + pgvector access layer.

Why pgvector instead of Pinecone/Chroma/FAISS?
The job description explicitly lists "vector DBs like Pinecone/pgvector",
and your resume already shows strong PostgreSQL experience. Using pgvector
lets you say in the interview: "I extended a database you already know how
to operate into a vector store, instead of adding a brand-new managed
service" - which is a more senior answer than "I called Pinecone's API".

pgvector is a Postgres extension that adds a `vector` column type and
distance operators (<-> for L2, <=> for cosine, <#> for inner product),
so similarity search is just a SQL ORDER BY.
"""

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
from pgvector.psycopg2.vector import Vector

from app.config import (
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    EMBEDDING_DIM,
)


def get_connection(register_pgvector=True):
    """
    Opens a new connection and registers the pgvector type adapter so
    Python lists/np.arrays can be inserted directly into `vector` columns.
    """
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )
    if register_pgvector:
        register_vector(conn)
    return conn


def init_db():
    """
    Creates the pgvector extension (if missing) and the documents table.
    Called once on FastAPI startup - see main.py's startup event.

    Table design:
    - id: primary key
    - source: original filename, so answers can cite where they came from
    - chunk_index: position of this chunk within its source document
    - content: the raw chunk text (needed to build the LLM prompt later)
    - embedding: the vector representation used for similarity search
    """
    conn = get_connection(register_pgvector=False)
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    register_vector(conn)
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            source TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding vector({EMBEDDING_DIM})
        );
        """
    )
    # ivfflat is an approximate-nearest-neighbor index type pgvector ships.
    # It speeds up similarity search on larger tables. It needs at least a
    # few rows to build well, so we guard it - harmless if it's skipped on
    # a fresh/empty database, it'll just do a full scan until you re-run it.
    cur.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_indexes WHERE indexname = 'documents_embedding_idx'
            ) THEN
                CREATE INDEX documents_embedding_idx
                ON documents USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            END IF;
        EXCEPTION WHEN OTHERS THEN
            -- Not enough rows yet to build the index; safe to ignore on first boot.
            NULL;
        END $$;
        """
    )
    conn.commit()
    cur.close()
    conn.close()


def insert_chunks(rows):
    """
    Bulk-inserts (source, chunk_index, content, embedding) tuples.
    Using execute_values instead of one INSERT per row is a lot faster
    for batches - worth mentioning if asked about performance.
    """
    conn = get_connection()
    cur = conn.cursor()
    psycopg2.extras.execute_values(
        cur,
        """
        INSERT INTO documents (source, chunk_index, content, embedding)
        VALUES %s
        """,
        rows,
    )
    conn.commit()
    cur.close()
    conn.close()


def similarity_search(query_embedding, top_k=4):
    """
    Returns the top_k most similar chunks to the query embedding using
    cosine distance (<=>). Lower distance = more similar, so we ORDER BY
    ascending and take the first top_k rows - this is the "retrieval"
    step of Retrieval-Augmented Generation.
    """
    conn = get_connection()
    cur = conn.cursor()
    query_vector = Vector(query_embedding).to_text()
    cur.execute(
        """
        SELECT source, chunk_index, content, embedding <=> %s::vector AS distance
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """,
        (query_vector, query_vector, top_k),
    )
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results
