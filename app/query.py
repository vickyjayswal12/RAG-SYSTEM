"""
The "answer a question" side of RAG:
embed the question -> retrieve the most similar chunks from Postgres ->
pass them to Gemini as context -> return a grounded answer + sources.
"""

from app.gemini_client import embed_text, generate_answer
from app.db import similarity_search


def answer_question(question: str, top_k: int = 4) -> dict:
    query_embedding = embed_text(question, task_type="retrieval_query")

    results = similarity_search(query_embedding, top_k=top_k)
    # each row: (source, chunk_index, content, distance)

    context_chunks = [row[2] for row in results]
    answer = generate_answer(question, context_chunks)

    sources = [
        {"source": row[0], "chunk_index": row[1], "similarity_distance": round(float(row[3]), 4)}
        for row in results
    ]

    return {"answer": answer, "sources": sources}
