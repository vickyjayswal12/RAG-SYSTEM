"""
Splits long text into overlapping chunks before embedding.

Why chunk at all?
1. Embedding models have an input size limit.
2. Smaller chunks = more precise retrieval (you get back the exact
   paragraph that answers the question, not a whole 50-page document).
3. Overlap (CHUNK_OVERLAP) prevents a sentence that answers the question
   from being awkwardly cut in half at a chunk boundary.

We use LangChain's RecursiveCharacterTextSplitter rather than writing a
naive `text[i:i+800]` splitter, because it tries to split on paragraph
breaks, then sentences, then words - in that order - so chunks stay
semantically coherent instead of stopping mid-word.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)
