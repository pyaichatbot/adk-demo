# ADK × LiteLLM × AG‑UI (Static) — Demo

This demo shows:
- **Two ADK LLM agents** + an **Orchestrator** (sequential pipeline + LLM collaboration).
- **In-house LLM via LiteLLM Proxy** (OpenAI-compatible), configurable to Azure OpenAI/OpenRouter/Gemini via LiteLLM.
- **AG‑UI compliant streaming** over **SSE** with a **static HTML UI**.

> Built to follow the official references for ADK and AG‑UI. See docs cited inline.

## What’s inside

```
adk_agui_litellm_demo/
├─ docker-compose.yml
├─ .env.example
├─ README.md
├─ backend/                # FastAPI app: AG‑UI server + Orchestrator endpoints
│  ├─ pyproject.toml
│  ├─ backend/
│  │  ├─ __init__.py
│  │  ├─ agui_protocol.py  # minimal AG‑UI event types + SSE encoder (Python port)
│  │  ├─ agents.py         # ADK agents (2 workers + orchestrator)
│  │  ├─ orchestrator.py   # sequential & collaboration flows
│  │  └─ main.py           # FastAPI routes: /api/agui/run (SSE), /api/run/sequential, /api/run/collab
│  ├─ Dockerfile
├─ litellm/
│  ├─ docker-compose.override.sample.yml
│  ├─ config.yaml          # LiteLLM routing (edit for your providers)
│  └─ Dockerfile
└─ frontend/
   ├─ index.html           # static AG‑UI chat (vanilla)
   └─ Dockerfile
```

## Quick start

### 0) Prereqs
- Docker & Docker Compose
- Python 3.10+ (optional, only if you want to run backend outside Docker)

### 1) Configure LiteLLM
Edit `litellm/config.yaml` with your provider(s). Example uses **OpenRouter** by default.
Create `.env` from example:
```bash
cp .env.example .env
# Fill values: OPENROUTER_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, AZURE_... as needed
```

### 2) Run everything
```bash
docker compose up --build
```

- Frontend: http://localhost:5173
- AG‑UI SSE Endpoint: http://localhost:8080/api/agui/run
- Orchestrator APIs (JSON, non-AG‑UI): 
  - `POST http://localhost:8080/api/run/sequential`
  - `POST http://localhost:8080/api/run/collab`

### 3) Try the demo
Open the UI, ask: “Summarize this URL and then draft 3 insights.”  
Behind the scenes:
- **Sequential flow**: WebResearcher → Writer
- **Collaboration flow**: Orchestrator LLM delegates to both workers and merges

---

## Notes on conformance

- **ADK**: Uses `google-adk` Python package and the **LiteLlm** wrapper per “Models & Authentication → LiteLLM” (ADK docs).  
- **LiteLLM**: Runs official proxy in Docker with OpenAI‑compatible `/chat/completions` endpoint (docs.litellm.ai).  
- **AG‑UI**: Implements server per **AG‑UI “Server Quickstart”** with SSE events (`RUN_STARTED`, `TEXT_MESSAGE_*`, `RUN_FINISHED`).

### References
- ADK repo and docs: google/adk-python; “Models & Authentication → LiteLLM” section.  
- AG‑UI: ag-ui-protocol/ag-ui & “Server Quickstart” (Python example).  
- LiteLLM proxy: official Docker quickstart + compose docs.

---

## Environment

- `LITELLM_BASE_URL` (default `http://litellm:4000` in Docker; `http://localhost:4000` locally)
- `LITELLM_API_KEY`  (LiteLLM **virtual key** or pass‑through key)
- `LITELLM_MODEL`    (e.g., `openrouter/auto`, `openai/gpt-4o-mini`, `anthropic/claude-3-5-sonnet`)

---

## Security

This is a demo. For production:
- Put LiteLLM behind auth (JWT/API gateway), set RPM/TPS quotas.
- Rotate keys + disable direct provider keys exposure.
- Add request logging, trace IDs, and guardrails/tool confirmation (ADK has built‑ins).

Enjoy!
