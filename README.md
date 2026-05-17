# RAG Technical Documentation Assistant

A Retrieval-Augmented Generation (RAG) assistant that answers questions about technical documentation. The retrieval, grading, web-search fallback, generation, and hallucination check are orchestrated as a LangGraph workflow. The system is served via a FastAPI backend and ships with a Streamlit chat UI.

The default corpus is the FastAPI tutorial (URLs configured in `config.py`); the ingester also accepts any local markdown / PDF / text files dropped into `./docs`.

## Features

- **LangGraph workflow** with `retrieve → grade_documents → (web_search) → generate → grade_hallucination → record_exchange` nodes
- **Document relevance grading** before generation, filtering irrelevant retrievals out
- **Web search fallback (Tavily)** when retrieved documents are graded irrelevant
- **Self-RAG style hallucination check** that validates the generated answer against the retrieved context, with one strict-retry attempt and a low-confidence fallback
- **Conversation memory** via LangGraph's `MemorySaver` checkpointer, keyed by `session_id`
- **Streamlit chat UI** with status indicator, suggestion chips, hallucination badges, and source citations
- **Graceful degradation** when external services (e.g. Tavily) are unreachable — the graph appends a placeholder and continues rather than crashing

## Architecture

### LangGraph workflow

```
                      START
                        │
                        ▼
                   retrieve            (Chroma similarity search, k=4)
                        │
                        ▼
                grade_documents        (Groq LLM grades each chunk yes/no)
                        │
              ┌─────────┴─────────┐
              │ any doc rejected? │
              ▼                   ▼
            Yes                  No
              │                   │
              ▼                   │
         web_search                │   (Tavily, append web docs)
              │                   │
              └─────────┬─────────┘
                        ▼
                    generate           (Groq llama-3.3-70b synthesises answer)
                        │
                        ▼
              grade_hallucination      (Groq verifies answer is grounded)
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   grounded          retry          low_confidence
        │               │               │
        │               ▼               ▼
        │      increment_regeneration   mark_low_confidence
        │               │               │
        │               └──► generate   │
        │                   (strict)    │
        │                               │
        └────────┬──────────────────────┘
                 ▼
           record_exchange              (append question + answer to chat_history)
                 │
                 ▼
                END
```

### Components

| Layer | Choice | Why |
|---|---|---|
| LLM | Groq `llama-3.3-70b-versatile` | Very fast inference, free tier sufficient for grading + generation |
| Embeddings | Cohere `embed-english-v3.0` | Strong retrieval quality, dedicated retrieval-optimised model |
| Vector store | Chroma (persisted to `./chroma_db`) | Local, zero-ops, embeddable in the same Python process |
| Orchestration | LangGraph `StateGraph` + `MemorySaver` | Native support for conditional edges and per-thread state |
| Web fallback | Tavily | Purpose-built search API for LLM agents |
| API | FastAPI + Uvicorn | Lightweight, automatic Swagger docs at `/docs` |
| UI | Streamlit | Fast to build, supports `st.chat_message` natively |

## Project structure

```
.
├── README.md
├── requirements.txt
├── .env.example                  Template for the three required API keys
├── .gitignore
├── config.py                     Paths, model names, URL list, chunk + retrieval settings
├── ingest.py                     One-shot script: load docs + URLs → split → embed → persist Chroma
├── docs/                         Drop local .md / .pdf / .txt files here for ingestion
├── chroma_db/                    Persisted vector store (created by ingest.py, gitignored)
├── .streamlit/
│   └── config.toml               Dark theme for the Streamlit UI
├── streamlit_app.py              Streamlit chat client (talks to FastAPI over HTTP)
└── app/
    ├── __init__.py
    ├── main.py                   FastAPI app: POST /ask, GET /health, GET /
    ├── state.py                  GraphState TypedDict (question, documents, history, etc.)
    ├── llm.py                    Factories for ChatGroq, CohereEmbeddings, TavilyClient
    ├── prompts.py                Grader, generator, strict-generator, hallucination prompts
    ├── retriever.py              Chroma loader + count helper
    ├── nodes.py                  retrieve / grade_documents / web_search / generate / grade_hallucination / ...
    └── graph.py                  LangGraph wiring + conditional edges
```

## Setup

### 1. Prerequisites

- Python 3.12 (the venv shipped with this repo targets 3.12; any 3.10+ should work)
- Git
- API keys for Groq, Cohere, and Tavily (all have free tiers)

### 2. Clone and set up the virtual environment

```powershell
git clone <repo-url>
cd "RAG technical Doc Assistant"

python -m venv venv
.\venv\Scripts\Activate.ps1     # PowerShell
# or: source venv/bin/activate   # bash / macOS / Linux

python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run once in an elevated shell:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 3. API keys

Get free-tier keys from:

| Provider | Where | Env var |
|---|---|---|
| Groq | https://console.groq.com | `GROQ_API_KEY` |
| Cohere | https://dashboard.cohere.com/api-keys | `COHERE_API_KEY` |
| Tavily | https://app.tavily.com | `TAVILY_API_KEY` |

Copy the template and fill in the values:

```powershell
copy .env.example .env
```

Then open `.env` and paste the three keys.

### 4. Document corpus

Two sources, both configurable:

- **URLs**: edit `URL_SOURCES` in `config.py`. The default value crawls two FastAPI tutorial pages.
- **Local files**: drop `.md`, `.pdf`, or `.txt` files into `./docs`.

`ingest.py` loads both, so you can use either or both.

## Running the application

### 1. Build the index

```powershell
python ingest.py
```

Expected output:
```
Loading local docs...
  loaded 0 local documents
Loading URL docs...
  loading 2 URLs...
  loaded 2 URL documents
Splitting...
  ~80 chunks
Embedding + persisting to Chroma...
Done. N chunks persisted to ./chroma_db
```

Pass `--reset` to wipe the existing index before rebuilding.

### 2. Start the FastAPI backend

```powershell
uvicorn app.main:app --reload
```

The server listens on `http://localhost:8000`. Open `http://localhost:8000/docs` for the Swagger UI.

### 3. (Optional) Start the Streamlit UI

In a second terminal:

```powershell
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`.

## API examples

### Using the Swagger UI (no curl required)

FastAPI auto-generates an interactive API explorer. With the backend running, open:

```
http://localhost:8000/docs
```

You'll see all endpoints listed. The flow for trying any of them is the same:

1. Click the endpoint row to expand it (e.g. `POST /ask`).
2. Click **Try it out** in the top-right of the panel.
3. The request body becomes an editable form. Replace the placeholder JSON with your input.
4. Click the blue **Execute** button.
5. Scroll down to see the response — status code, headers, and the JSON body.

#### Walkthrough: ask an in-doc question

1. Open `http://localhost:8000/docs`.
2. Expand **`POST /ask`** → click **Try it out**.
3. Replace the request body with:
   ```json
   {
     "question": "What are query parameters in FastAPI?"
   }
   ```
4. Click **Execute**.
5. The response body will look like:
   ```json
   {
     "answer": "In FastAPI, query parameters are function parameters that are not part of the path...",
     "sources": ["https://fastapi.tiangolo.com/tutorial/query-params/"],
     "session_id": "a3f1c2e9-...",
     "hallucination_check": "grounded"
   }
   ```
6. **Copy the `session_id`** value — you'll need it for follow-up questions.

#### Walkthrough: ask a follow-up question (conversation memory)

1. Still on **`POST /ask`** → **Try it out** again.
2. Paste the previous `session_id` into the request body:
   ```json
   {
     "question": "How do I validate one?",
     "session_id": "a3f1c2e9-..."
   }
   ```
3. Click **Execute**. The assistant resolves "one" using the prior turn (because LangGraph's `MemorySaver` keeps history per `thread_id`).

#### Walkthrough: trigger the Tavily web-search fallback

Ask a question that the indexed docs cannot answer:

```json
{
  "question": "What is the latest version of PostgreSQL?"
}
```

The grader rejects the FastAPI chunks, the `web_search` node fires, and the `sources` array in the response will contain external URLs (e.g. `postgresql.org`, `wikipedia.org`) rather than `fastapi.tiangolo.com`.

#### Walkthrough: check the health endpoint

1. Expand **`GET /health`** → **Try it out** → **Execute**.
2. Expected response when the index is built:
   ```json
   { "status": "ok", "collection_count": 78 }
   ```
3. If you see `"status": "no_index"`, you forgot to run `python ingest.py`.

#### Reading the hallucination badge

Every `/ask` response includes a `hallucination_check` field:
- `"grounded"` — the Self-RAG grader confirms the answer is supported by the retrieved context
- `"hallucinated"` — the grader flagged unsupported claims; the answer will also contain a low-confidence notice appended at the end

### `GET /health`

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "collection_count": 78
}
```

When no index exists:
```json
{
  "status": "no_index",
  "collection_count": 0,
  "hint": "Run `python ingest.py` after setting COHERE_API_KEY."
}
```

### `POST /ask` — in-doc question (no web fallback)

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are query parameters in FastAPI?"}'
```

Response:
```json
{
  "answer": "In FastAPI, query parameters are function parameters that are not part of the path. When you declare them with default values, they become optional query string arguments...",
  "sources": [
    "https://fastapi.tiangolo.com/tutorial/query-params/"
  ],
  "session_id": "a3f1c2e9-...-...",
  "hallucination_check": "grounded"
}
```

### `POST /ask` — out-of-doc question (triggers Tavily fallback)

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the latest version of PostgreSQL?"}'
```

Response (truncated):
```json
{
  "answer": "Based on the web results, the latest stable version of PostgreSQL is ...",
  "sources": [
    "https://www.postgresql.org/about/news/...",
    "https://en.wikipedia.org/wiki/PostgreSQL"
  ],
  "session_id": "b7d8e6f2-...-...",
  "hallucination_check": "grounded"
}
```

### `POST /ask` — follow-up question with session memory

Re-use the `session_id` returned by an earlier call:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I validate one?", "session_id": "a3f1c2e9-...-..."}'
```

The generator receives the prior turn via `chat_history` (LangGraph reducer-merged) and can resolve the pronoun ("one" → "a query parameter").



## Design Decisions & Tradeoffs

**Why Groq for the LLM**
Groq runs Llama 3.3 70B with very low latency — important here because each request makes multiple LLM calls (grading, generation, hallucination check). Fast per-call speed keeps total response time tolerable.
Tradeoff: no embedding endpoint, so a second provider (Cohere) is needed.

**Why Cohere for embeddings**
`embed-english-v3.0` is purpose-built for retrieval and works well on technical Q&A. Free tier is sufficient for this project.
Tradeoff: adds a second provider to the stack. An all-local setup with `sentence-transformers` would be simpler but slightly lower quality.

**Why ChromaDB**
Runs locally, no separate service needed, integrates cleanly with LangChain. Perfect for a single-user assistant.
Tradeoff: not suitable for multi-user or scaled deployments — that would need Pinecone, Qdrant, or pgvector.

**Why this graph structure**
Follows the CRAG (Corrective RAG) pattern extended with a Self-RAG hallucination check:
- Grading before generation filters out bad retrievals before they reach the LLM
- Web search only fires when grading fails — clean queries skip it entirely
- Hallucination check is bounded to one retry to keep latency predictable

**Hallucination check — binary instead of scored**
The grader returns `yes` / `no` rather than a confidence score. Simpler prompt, more reliable structured output.
Tradeoff: less granular. A numeric score would allow smarter routing but adds calibration complexity.

If Tavily (web search) fails due to a network error or missing key, the node catches the exception and passes a placeholder document instead of crashing. The generator still produces an answer from whatever local docs it has.

## Write-up

## Thought Process

The assignment called for retrieval → grading → generation as separate LangGraph nodes — that's the CRAG pattern. Once the skeleton was clear, extras were added where they added real value:

- **Web fallback** — if retrieval fails, searching the web is better than giving up
- **Hallucination check** — catches confident-sounding answers built on weak context
- **Conversation memory** — turns a one-shot Q&A into something useful for real research
- **Streamlit UI** — makes the whole system accessible to non-technical reviewers

---

## What I'd Improve With More Time

- **Reranker** — add Cohere Rerank between retrieval and grading to sharpen results before the expensive LLM grading step
- **Batched grading** — grade all retrieved chunks in one prompt instead of one call per chunk

---

## Assumptions

- Corpus is small enough for a local Chroma index — no external vector DB needed
- Single user, no auth or rate limiting
- English only — for multilingual use `embed-multilingual-v3.0`
- API keys loaded from `.env` at startup; rotating keys requires a restart
- Reviewer runs locally with their own free-tier API keys
- Tavily fallback is best-effort — if it fails, a graceful placeholder is returned

## Chunking & Embedding Strategy

- **Chunk size 1000 / overlap 150**
Large enough to capture a complete concept, small enough to stay precise. The 150-character overlap ensures sentences at chunk boundaries aren't lost.

- **Smart splitting over fixed cutting**
Uses `RecursiveCharacterTextSplitter` which breaks at paragraph → line → word boundaries. Chunks follow the document's natural structure instead of cutting mid-sentence.

- **Cohere `embed-english-v3.0`**
Trained specifically for retrieval tasks. Uses separate encoding modes for documents (`search_document`) and queries (`search_query`) which improves matching accuracy.


## License

MIT (or whatever the assignment context requires).
