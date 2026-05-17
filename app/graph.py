from functools import partial

from langgraph.graph import END, START, StateGraph

from .nodes import (
    decide_after_hallucination,
    decide_to_generate,
    generate,
    grade_documents,
    grade_hallucination,
    increment_regeneration,
    mark_low_confidence,
    record_exchange,
    retrieve,
    web_search,
)
from .state import GraphState


def build_graph(retriever, checkpointer=None):
    """Compile the LangGraph workflow: retrieve -> grade -> (web_search ->) generate -> hallucination check. Args: retriever - LangChain retriever bound to Chroma; checkpointer - optional LangGraph checkpointer for conversation memory."""
    g = StateGraph(GraphState)

    g.add_node("retrieve", partial(retrieve, retriever=retriever))
    g.add_node("grade_documents", grade_documents)
    g.add_node("web_search", web_search)
    g.add_node("generate", generate)
    g.add_node("grade_hallucination", grade_hallucination)
    g.add_node("increment_regeneration", increment_regeneration)
    g.add_node("mark_low_confidence", mark_low_confidence)
    g.add_node("record_exchange", record_exchange)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"web_search": "web_search", "generate": "generate"},
    )
    g.add_edge("web_search", "generate")
    g.add_edge("generate", "grade_hallucination")
    g.add_conditional_edges(
        "grade_hallucination",
        decide_after_hallucination,
        {
            "end": "record_exchange",
            "retry": "increment_regeneration",
            "low_confidence": "mark_low_confidence",
        },
    )
    g.add_edge("increment_regeneration", "generate")
    g.add_edge("mark_low_confidence", "record_exchange")
    g.add_edge("record_exchange", END)

    if checkpointer is not None:
        return g.compile(checkpointer=checkpointer)
    return g.compile()
