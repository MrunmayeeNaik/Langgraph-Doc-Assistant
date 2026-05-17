from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    question: str
    documents: list[Document]
    web_search_needed: str
    generation: str
    hallucination_check: str
    regeneration_count: int
    chat_history: Annotated[list[BaseMessage], add_messages]
