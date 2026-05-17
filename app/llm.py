"""Factory functions for the LLM, embeddings, and web-search clients.

All clients read their API keys from environment variables, so a missing
key fails at first use rather than at import time. This lets the FastAPI
app boot even before keys are configured.
"""

import os

from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from tavily import TavilyClient

import config


def get_chat_model() -> ChatGroq:
    """Construct the Groq chat model used for grading + generation. Args: none."""
    return ChatGroq(model=config.LLM_MODEL, temperature=0)


def get_embeddings() -> CohereEmbeddings:
    """Construct the Cohere embeddings client used for indexing + query embedding. Args: none."""
    return CohereEmbeddings(model=config.EMBED_MODEL)


def get_tavily_client() -> TavilyClient:
    """Construct the Tavily web-search client. Args: none (reads TAVILY_API_KEY from env)."""
    return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
