from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

import config

from .llm import get_chat_model, get_tavily_client
from .prompts import (
    GENERATOR_PROMPT,
    GRADER_PROMPT,
    HALLUCINATION_PROMPT,
    STRICT_GENERATOR_PROMPT,
)
from .state import GraphState

MAX_REGENERATIONS = 1


class GradeBinary(BaseModel):
    """Binary relevance score for a retrieved document."""

    binary_score: str = Field(description="Relevance: 'yes' or 'no'")


def retrieve(state: GraphState, retriever) -> dict:
    """Fetch top-k relevant chunks from Chroma for the question. Args: state - GraphState; retriever - LangChain retriever instance."""
    docs = retriever.invoke(state["question"])
    return {"documents": docs, "question": state["question"]}


def grade_documents(state: GraphState) -> dict:
    """Grade each retrieved doc yes/no for relevance; mark web_search_needed if any failed. Args: state - GraphState with question + documents."""
    llm = get_chat_model()
    grader = GRADER_PROMPT | llm.with_structured_output(GradeBinary)

    relevant: list[Document] = []
    web_search_needed = "No"
    for d in state["documents"]:
        result: GradeBinary = grader.invoke(
            {"question": state["question"], "document": d.page_content}
        )
        if result.binary_score.strip().lower() == "yes":
            relevant.append(d)
        else:
            web_search_needed = "Yes"

    return {
        "documents": relevant,
        "question": state["question"],
        "web_search_needed": web_search_needed,
    }


def web_search(state: GraphState) -> dict:
    """Query Tavily and append web results as Documents; degrade gracefully if Tavily is unreachable. Args: state - GraphState with question + existing documents."""
    try:
        client = get_tavily_client()
        results = client.search(
            query=state["question"], max_results=config.TAVILY_MAX_RESULTS
        )
        web_docs = [
            Document(page_content=r["content"], metadata={"source": r["url"]})
            for r in results.get("results", [])
        ]
    except KeyError:
        placeholder = Document(
            page_content="(Web search skipped: TAVILY_API_KEY not configured.)",
            metadata={"source": "tavily://disabled"},
        )
        web_docs = [placeholder]
    except Exception as e:
        placeholder = Document(
            page_content=f"(Web search unavailable: {type(e).__name__}. Answer from local docs only.)",
            metadata={"source": "tavily://error"},
        )
        web_docs = [placeholder]

    return {
        "documents": state["documents"] + web_docs,
        "question": state["question"],
    }


def generate(state: GraphState) -> dict:
    """Run the generator LLM over the question + accumulated documents and return the answer. Args: state - GraphState with question, documents, and optional chat_history."""
    llm = get_chat_model()
    is_retry = state.get("regeneration_count", 0) > 0
    prompt = STRICT_GENERATOR_PROMPT if is_retry else GENERATOR_PROMPT
    chain = prompt | llm
    context = "\n\n".join(d.page_content for d in state["documents"])
    history = state.get("chat_history", []) or []
    output = chain.invoke(
        {
            "context": context,
            "question": state["question"],
            "chat_history": history,
        }
    )
    return {
        "generation": output.content,
        "documents": state["documents"],
        "question": state["question"],
        "regeneration_count": state.get("regeneration_count", 0),
    }


def grade_hallucination(state: GraphState) -> dict:
    """Check whether the generated answer is grounded in the provided documents (Self-RAG style). Args: state - GraphState with generation + documents."""
    llm = get_chat_model()
    grader = HALLUCINATION_PROMPT | llm.with_structured_output(GradeBinary)
    context = "\n\n".join(d.page_content for d in state["documents"])
    result: GradeBinary = grader.invoke(
        {"context": context, "generation": state["generation"]}
    )
    score = result.binary_score.strip().lower()
    return {
        "hallucination_check": "grounded" if score == "yes" else "hallucinated",
        "question": state["question"],
        "documents": state["documents"],
        "generation": state["generation"],
        "regeneration_count": state.get("regeneration_count", 0),
    }


def mark_low_confidence(state: GraphState) -> dict:
    """Append a low-confidence warning to the generation when regeneration still fails groundedness. Args: state - GraphState with the latest generation."""
    warning = (
        "\n\n_Note: this answer could not be fully verified against the retrieved context "
        "and may contain inaccuracies. Treat with caution._"
    )
    return {
        "generation": state["generation"] + warning,
        "question": state["question"],
        "documents": state["documents"],
    }


def record_exchange(state: GraphState) -> dict:
    """Append the user question and assistant answer to chat_history (merged via add_messages reducer). Args: state - GraphState with the final question + generation."""
    return {
        "chat_history": [
            HumanMessage(content=state["question"]),
            AIMessage(content=state["generation"]),
        ]
    }


def increment_regeneration(state: GraphState) -> dict:
    """Bump the regeneration counter before re-running generate with the strict prompt. Args: state - GraphState with the previous attempt."""
    return {
        "regeneration_count": state.get("regeneration_count", 0) + 1,
        "question": state["question"],
        "documents": state["documents"],
        "generation": state["generation"],
    }


def decide_to_generate(state: GraphState) -> str:
    """Conditional edge: route to web_search if grading rejected any doc, else generate. Args: state - GraphState with web_search_needed set."""
    return "web_search" if state["web_search_needed"] == "Yes" else "generate"


def decide_after_hallucination(state: GraphState) -> str:
    """Conditional edge after hallucination grading: end, retry once, or flag low confidence. Args: state - GraphState with hallucination_check + regeneration_count."""
    if state["hallucination_check"] == "grounded":
        return "end"
    if state.get("regeneration_count", 0) < MAX_REGENERATIONS:
        return "retry"
    return "low_confidence"
