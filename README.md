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



## Design decisions and tradeoffs

### Why Groq for the LLM
Groq's hosted inference is exceptionally fast on Llama 3.3 70B, which matters because the graph makes multiple LLM calls per request (one per retrieved doc for grading, one for generation, one for hallucination check, potentially one for strict regeneration). Lower latency per node call keeps total request time tolerable. Tradeoff: no embedding endpoint, so a second provider is required.

### Why Cohere for embeddings
Cohere's `embed-english-v3.0` is purpose-built for retrieval and consistently performs well on technical Q&A. Free trial keys are sufficient. Tradeoff: an extra provider in the stack; an all-OpenAI or all-local-sentence-transformers setup would be simpler operationally but trade quality (sentence-transformers) or cost (OpenAI).

### Why Chroma for the vector store
Chroma persists to local disk, runs in-process (no separate service), and integrates cleanly with LangChain. For a single-tenant assistant this is the lowest-friction option. Tradeoff: not suitable for multi-tenant or horizontally-scaled deployments — that would call for Qdrant, Pinecone, or pgvector.

### Why this graph topology
The pattern is the adaptive-RAG / corrective-RAG (CRAG) pattern from the LangGraph examples, extended with a Self-RAG-inspired hallucination check. Concretely:

1. **Retrieve → grade → fallback** filters out poor retrievals before they reach the generator, reducing hallucinations from irrelevant context.
2. **Web search fallback** only fires when grading rejected at least one doc, so happy-path queries skip the network call.
3. **Hallucination check → strict regenerate → low-confidence label** is a bounded loop: at most one regeneration attempt to keep latency predictable, and a clear "we tried but the answer is shaky" signal as a last resort.

Tradeoff: the per-doc grading is a multiplier on LLM calls. For `k=4`, every query makes 4 grading calls + 1 generation + 1 hallucination check (+ 1 strict regen if needed). With Groq this is acceptable, but on a slower provider it would dominate latency. A batched grading prompt would cut this to one call.

### Why MemorySaver for conversation memory
`MemorySaver` is in-process and per-thread (keyed by `session_id`). It supports follow-up questions within a session without any external dependency. Tradeoff: state evaporates when the FastAPI process restarts. For a production system, swap in `SqliteSaver` (single instance) or `PostgresSaver` (multi-instance) — same API, persistent storage.

### Chunking strategy
- **Splitter**: `RecursiveCharacterTextSplitter` with `chunk_size=1000`, `chunk_overlap=150`.
- **Rationale**: technical documentation tends to have clear semantic boundaries (paragraphs, code blocks, headings). 1000 characters is roughly a paragraph or two, large enough to capture a complete concept but small enough to surface focused snippets. The 15% overlap protects against splitting a fact across two chunks while keeping index size manageable.
- **Retriever k**: 4. Empirically this gives the grader enough breadth to filter from without overwhelming the generator's context window. Configurable in `config.py`.

### Hallucination check threshold
The grader produces a binary `yes` / `no` rather than a numeric score. Tradeoff: simpler prompt and more deterministic structured output, at the cost of granularity. A confidence score would allow nuanced routing (e.g. partial fallback) but adds calibration overhead.

### Graceful degradation
When Tavily fails (network, key missing, rate limit), the `web_search` node catches the exception and appends a placeholder document explaining the failure, rather than letting the whole graph crash with a 500. This keeps the assistant useful even under partial outages — the generator simply produces an answer from whatever local docs it still has.

## Write-up

### Thought process
The assignment asks for retrieval, grading, and generation as discrete LangGraph nodes — that's exactly the CRAG pattern. Once that skeleton was clear, the design choices were about *where to add value*:

1. **Bonus: web fallback** is the natural extension of grading — once you know retrieval failed, doing nothing is leaving signal on the table.
2. **Bonus: hallucination check** is Self-RAG's contribution and addresses the largest remaining failure mode: the model generates fluently from weak context.
3. **Bonus: conversation memory** turns a single-shot Q&A into something usable for real research workflows.
4. **Bonus: Streamlit UI** makes all of the above legible to a non-technical reviewer in one place.

FastAPI is the API contract; Streamlit is purely a thin client over it. Keeping the two layered means the same backend can later power a CLI, a Slack bot, or a different UI.

### What I would improve with more time
- **Re-ranking** — insert a Cohere `rerank` node between `retrieve` and `grade_documents` to sharpen the candidate set before the (expensive) LLM grading
- **Batched grading** — send all `k` retrieved documents to the grader in a single prompt, dropping `k` calls to one
- **Streaming responses** — `StreamingResponse` from FastAPI and `st.write_stream` in Streamlit so the answer appears token-by-token
- **Better citation rendering** — show inline `[1]` markers tied to the sources expander, instead of a flat list
- **Persistent memory** — swap `MemorySaver` for `SqliteSaver` so chat history survives restarts
- **Tests** — pytest with a stubbed retriever and mocked LLM so the graph wiring is regression-protected
- **Observability** — wire LangSmith tracing in to surface per-node latency and token usage
- **More aggressive corpus** — sitemap-driven crawl for whole documentation sites instead of an enumerated URL list
- **Multi-tenant collections** — namespacing per user/project, with metadata filters at retrieve time

### Assumptions
- The corpus is small enough that an in-process Chroma index is sufficient (no need for a separate vector DB service)
- A single user / single tenant — no auth, no rate limiting, no per-user collections
- English-only — Cohere's `embed-english-v3.0` is monolingual; for multilingual use `embed-multilingual-v3.0`
- API keys are loaded from `.env` at process start; rotating keys requires a restart
- The reviewer will run locally and provide their own free-tier keys; the assignment does not require deployment
- The Tavily fallback is best-effort — if it fails, the system surfaces a placeholder rather than failing the request

### Chunking/embedding strategy choices
- **Chunk size 1000 / overlap 150**: tuned for prose-heavy technical docs; struck a balance between semantic completeness (large enough to contain a full concept) and retrieval precision (small enough that irrelevant context doesn't crowd a chunk)
- **`RecursiveCharacterTextSplitter`** over token-aware splitters: it respects paragraph and sentence boundaries via its default separator hierarchy (`["\n\n", "\n", " ", ""]`), which matches markdown / HTML doc structure better than a fixed-token splitter
- **Cohere `embed-english-v3.0`**: 1024-dim embeddings, trained for retrieval (distinct from their generation embeddings). At indexing time the model encodes with `input_type="search_document"` and at query time with `"search_query"` — handled automatically by `langchain-cohere`
- **No metadata filtering at retrieval time**: with a small corpus, plain similarity is enough. For a larger corpus I'd add `source`, `section`, and `last_updated` metadata and use `Chroma.as_retriever(search_kwargs={"filter": ...})`
- **No re-ranking** in this version: a planned next step. The current grading node is a coarse form of re-ranking, but a dedicated cross-encoder (e.g. Cohere `rerank-english-v3.0`) would produce sharper ordering.

## License

MIT (or whatever the assignment context requires).
