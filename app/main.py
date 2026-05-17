import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from .graph import build_graph
from .retriever import get_collection_count, get_retriever

load_dotenv()


class AskRequest(BaseModel):
    question: str
    session_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    session_id: str
    hallucination_check: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: load collection count and compile the LangGraph once at startup. Args: app - the FastAPI instance."""
    app.state.collection_count = get_collection_count()
    app.state.graph = None
    app.state.graph_error = None
    app.state.checkpointer = MemorySaver()

    if app.state.collection_count > 0:
        try:
            app.state.graph = build_graph(
                get_retriever(), checkpointer=app.state.checkpointer
            )
        except Exception as e:
            app.state.graph_error = f"{type(e).__name__}: {e}"
    yield


app = FastAPI(title="RAG Technical Documentation Assistant", lifespan=lifespan)


@app.get("/")
def root():
    """Return a small landing payload listing available endpoints. Args: none."""
    return {
        "name": "RAG Technical Documentation Assistant",
        "endpoints": ["GET /health", "POST /ask"],
    }


@app.get("/health")
def health():
    """Report index + graph readiness so callers know whether /ask will work. Args: none."""
    if app.state.collection_count == 0:
        return {
            "status": "no_index",
            "collection_count": 0,
            "hint": "Run `python ingest.py` after setting COHERE_API_KEY.",
        }
    if app.state.graph is None:
        return {
            "status": "graph_not_ready",
            "collection_count": app.state.collection_count,
            "error": app.state.graph_error,
        }
    return {"status": "ok", "collection_count": app.state.collection_count}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Run the LangGraph workflow on the user's question and return answer + sources + session id. Args: req - AskRequest containing the question and optional session_id."""
    if app.state.graph is None:
        raise HTTPException(
            status_code=503,
            detail="Graph not ready. Ensure the Chroma index is built and API keys are set.",
        )

    session_id = req.session_id or str(uuid.uuid4())
    config_in = {"configurable": {"thread_id": session_id}}
    try:
        result = app.state.graph.invoke(
            {"question": req.question, "regeneration_count": 0}, config=config_in
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    sources: list[str] = []
    for d in result.get("documents", []):
        src = d.metadata.get("source") or d.metadata.get("url")
        if src and src not in sources:
            sources.append(src)

    return AskResponse(
        answer=result["generation"],
        sources=sources,
        session_id=session_id,
        hallucination_check=result.get("hallucination_check"),
    )
