"""Build the Chroma vector index from local docs and configured URLs.

Usage:
    python ingest.py             # add to existing index
    python ingest.py --reset     # wipe existing index first
"""

import argparse
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def load_local_docs(docs_dir: str):
    """Load all .md, .txt, and .pdf files from a directory tree. Args: docs_dir - filesystem path to scan recursively."""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        print(f"  docs/ directory not found at {docs_dir}, skipping local docs")
        return []

    docs = []
    for glob, loader_cls in [
        ("**/*.md", UnstructuredMarkdownLoader),
        ("**/*.txt", TextLoader),
        ("**/*.pdf", PyPDFLoader),
    ]:
        loader = DirectoryLoader(
            docs_dir, glob=glob, loader_cls=loader_cls, show_progress=True
        )
        docs.extend(loader.load())
    return docs


def load_url_docs(urls: list[str]):
    """Fetch and parse each URL into a LangChain Document via WebBaseLoader. Args: urls - list of fully-qualified page URLs to load."""
    if not urls:
        return []
    print(f"  loading {len(urls)} URLs...")
    return WebBaseLoader(urls).load()


def main():
    """Entrypoint: load docs+URLs, split, embed with Cohere, and persist into Chroma. Args: none (parses CLI flags from sys.argv)."""
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the existing Chroma index before ingesting",
    )
    args = ap.parse_args()

    if args.reset and Path(config.CHROMA_DIR).exists():
        print(f"Removing existing index at {config.CHROMA_DIR}")
        shutil.rmtree(config.CHROMA_DIR)

    print("Loading local docs...")
    local_docs = load_local_docs(config.DOCS_DIR)
    print(f"  loaded {len(local_docs)} local documents")

    print("Loading URL docs...")
    url_docs = load_url_docs(config.URL_SOURCES)
    print(f"  loaded {len(url_docs)} URL documents")

    docs = local_docs + url_docs
    if not docs:
        print(
            "\nNo documents found. Add files to ./docs or URLs to config.URL_SOURCES."
        )
        return

    print("Splitting...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"  {len(chunks)} chunks")

    print("Embedding + persisting to Chroma...")
    Chroma.from_documents(
        documents=chunks,
        embedding=CohereEmbeddings(model=config.EMBED_MODEL),
        persist_directory=config.CHROMA_DIR,
        collection_name=config.COLLECTION_NAME,
    )
    print(f"\nDone. {len(chunks)} chunks persisted to {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
