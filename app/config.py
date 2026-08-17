"""
Central place to read environment variables.

Why a separate config.py?
Interviewers like seeing that you don't scatter os.getenv() calls all
over the codebase. One module reads the environment once, and every
other module imports typed, validated settings from here.
"""

import os
from dotenv import load_dotenv

# Loads variables from a .env file into the process environment.
# In Docker we also pass these via docker-compose's `environment:` key,
# so this is mainly useful for local (non-docker) runs.
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "ragdb")
POSTGRES_USER = os.getenv("POSTGRES_USER", "raguser")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "ragpassword")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "models/gemini-1.5-flash")

# text-embedding-004 outputs 768-dimension vectors.
# This has to match the column size of the `embedding` column in Postgres
# (see db.py) or inserts will fail.
EMBEDDING_DIM = 768
