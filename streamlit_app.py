"""Minimal Streamlit chat UI for the RAG Technical Documentation Assistant.

Run with:
    streamlit run streamlit_app.py

Requires the FastAPI backend to be running:
    uvicorn app.main:app --reload
"""

import os
import uuid

import requests
import streamlit as st

API_BASE = os.getenv("RAG_API_BASE", "http://localhost:8000")

SUGGESTIONS = [
    "What are query parameters in FastAPI?",
    "How do I define a path parameter?",
    "How does FastAPI handle data validation?",
    "Show me a minimal FastAPI app example.",
]

HALLUCINATION_BADGE = {
    "grounded": ("🟢", "Grounded"),
    "hallucinated": ("🟠", "Low confidence"),
}


def fetch_health() -> dict:
    """Call the backend /health endpoint and return its JSON payload. Args: none."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}


def ask_question(question: str, session_id: str) -> dict:
    """Send a question to the backend /ask endpoint and return its JSON payload. Args: question - user query; session_id - per-browser session id for memory."""
    try:
        r = requests.post(
            f"{API_BASE}/ask",
            json={"question": question, "session_id": session_id},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        return {"error": f"{e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"error": str(e)}


def render_message(msg: dict) -> None:
    """Render a single chat message (user or assistant) with optional badge and sources. Args: msg - dict with role, content, and optional sources/hallucination_check."""
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            check = msg.get("hallucination_check")
            if check in HALLUCINATION_BADGE:
                emoji, label = HALLUCINATION_BADGE[check]
                st.caption(f"{emoji} {label}")
            sources = msg.get("sources") or []
            if sources:
                with st.expander(f"Sources · {len(sources)}"):
                    for src in sources:
                        st.markdown(f"- {src}")


def handle_question(question: str) -> None:
    """Run a question through the backend and append both sides of the exchange to session state. Args: question - the user's prompt string."""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask_question(question, st.session_state.session_id)

        if "error" in result:
            st.error(result["error"])
            st.session_state.messages.append(
                {"role": "assistant", "content": f"⚠️ {result['error']}"}
            )
            return

        answer = result.get("answer", "(no answer returned)")
        sources = result.get("sources", [])
        check = result.get("hallucination_check")

        st.markdown(answer)
        if check in HALLUCINATION_BADGE:
            emoji, label = HALLUCINATION_BADGE[check]
            st.caption(f"{emoji} {label}")
        if sources:
            with st.expander(f"Sources · {len(sources)}"):
                for src in sources:
                    st.markdown(f"- {src}")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "hallucination_check": check,
            }
        )


st.set_page_config(
    page_title="Doc Assistant",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

FEATURES = [
    ("🔎", "Vector retrieval"),
    ("✅", "Document grading"),
    ("🌐", "Web search fallback"),
    ("🛡️", "Hallucination check"),
    ("💬", "Conversation memory"),
    ("⚡", "LangGraph workflow"),
]

with st.sidebar:
    st.markdown("### 📚 Doc Assistant")
    st.caption("Technical documentation Q&A")
    st.markdown("")
    st.markdown("**Features**")
    for emoji, name in FEATURES:
        st.markdown(f"{emoji} &nbsp; {name}", unsafe_allow_html=True)

st.markdown(
    """
    <style>
      #MainMenu, header, footer {visibility: hidden;}
      .block-container {padding-top: 2.5rem; padding-bottom: 6rem; max-width: 760px;}
      .stChatMessage {background: transparent;}
      div[data-testid="stExpander"] {border: 1px solid #334155; background: #1E293B; border-radius: 8px;}
      div[data-testid="stExpander"] summary {font-size: 0.85rem; color: #94A3B8;}
      div[data-testid="stCaptionContainer"] {color: #94A3B8; font-size: 0.8rem;}
      hr {border-color: #334155;}
    </style>
    """,
    unsafe_allow_html=True,
)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

health = fetch_health()
status = health.get("status", "unknown")
status_dot = {
    "ok": "🟢",
    "no_index": "🟡",
    "graph_not_ready": "🟠",
    "unreachable": "🔴",
}.get(status, "⚪")

col_title, col_status, col_new = st.columns([7, 1, 2], vertical_alignment="center")
with col_title:
    st.markdown("### 📚 Doc Assistant")
with col_status:
    st.markdown(
        f"<div style='text-align:right; font-size:1.2rem;' title='Backend: {status}'>{status_dot}</div>",
        unsafe_allow_html=True,
    )
with col_new:
    if st.button("New chat", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

if status != "ok":
    if status == "no_index":
        st.warning("No index found. Run `python ingest.py` first.")
    elif status == "graph_not_ready":
        st.warning("Backend index exists but the graph failed to load. Check API keys.")
    elif status == "unreachable":
        st.error("Backend is unreachable. Start it with `uvicorn app.main:app --reload`.")

st.divider()

if not st.session_state.messages and st.session_state.pending_question is None:
    st.markdown(
        "<p style='color:#94A3B8; margin-bottom:1.5rem;'>Ask a question about the indexed documentation. Try one of these:</p>",
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, q in enumerate(SUGGESTIONS):
        with cols[i % 2]:
            if st.button(q, key=f"suggestion_{i}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()
else:
    for msg in st.session_state.messages:
        render_message(msg)

if st.session_state.pending_question:
    q = st.session_state.pending_question
    st.session_state.pending_question = None
    handle_question(q)

if prompt := st.chat_input("Ask a question..."):
    handle_question(prompt)
