from pathlib import Path

import chromadb
from langchain_chroma import Chroma

import config

from .llm import get_embeddings


def get_retriever():
    """Build a retriever against the persisted Chroma collection. Args: none (requires COHERE_API_KEY for query embedding)."""
    vs = Chroma(
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.CHROMA_DIR,
        embedding_function=get_embeddings(),
    )
    return vs.as_retriever(search_kwargs={"k": config.RETRIEVER_K})


def get_collection_count() -> int:
    """Return the number of chunks in the persisted Chroma collection (0 if none). Args: none (no API keys required)."""
    if not Path(config.CHROMA_DIR).exists():
        return 0
    try:
        client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        col = client.get_collection(config.COLLECTION_NAME)
        return col.count()
    except Exception:
        return 0
