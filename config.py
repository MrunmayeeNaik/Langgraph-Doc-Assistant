"""Central configuration for the RAG technical documentation assistant."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()

# Storage
CHROMA_DIR = str(PROJECT_ROOT / "chroma_db")
COLLECTION_NAME = "tech_docs"
DOCS_DIR = str(PROJECT_ROOT / "docs")

# Document sources crawled at ingest time. WebBaseLoader loads exactly these
# URLs (no link following) — list each page you want indexed.
URL_SOURCES: list[str] = [
    "https://fastapi.tiangolo.com/tutorial/",
    "https://fastapi.tiangolo.com/tutorial/query-params/",
]

# Models
EMBED_MODEL = "embed-english-v3.0"
LLM_MODEL = "llama-3.3-70b-versatile"

# Chunking & retrieval
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVER_K = 4

# Tavily fallback
TAVILY_MAX_RESULTS = 3
