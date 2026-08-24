"""
Thin wrapper around Google's `google-generativeai` SDK.

Why a wrapper instead of calling the SDK everywhere?
Keeps the rest of the app provider-agnostic. If you later swap Gemini
for OpenAI/Claude (the JD mentions all three), you only change this
one file - ingest.py and query.py don't need to know which provider
is behind embed_text() / generate_answer().
"""

import google.generativeai as genai

from app.config import GEMINI_API_KEY, EMBEDDING_MODEL, GENERATION_MODEL

genai.configure(api_key=GEMINI_API_KEY)


def embed_text(text: str, task_type: str = "retrieval_document"):
    """
    Turns text into a 768-dim vector using Gemini's embedding model.

    task_type matters for Gemini specifically: it nudges the embedding
    model to produce vectors optimized for how they'll be used.
    - "retrieval_document": use when embedding chunks you're storing.
    - "retrieval_query": use when embedding the user's question.
    Using the right task_type for each side measurably improves
    retrieval quality - a good detail to mention in an interview.
    """
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type=task_type,
        output_dimensionality=768,
    )
    return result["embedding"]


def generate_answer(question: str, context_chunks: list[str]) -> str:
    """
    The "generation" half of RAG: stuff retrieved chunks into a prompt
    and ask Gemini to answer using only that context.
    """
    context_text = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are a helpful assistant answering questions using ONLY the context provided below.
If the answer isn't in the context, say you don't have enough information - do not make things up.

Context:
{context_text}

Question: {question}

Answer:"""

    model = genai.GenerativeModel(GENERATION_MODEL)
    response = model.generate_content(prompt)
    return response.text
