# Personal Medical Assistant (RAG + Groq) — Research Edition

A **safety-first, evidence-grounded medical information assistant** built on a
lightweight BM25 retrieval pipeline and Groq LLM generation.  Designed to run
comfortably within Render's 512 MB free tier while delivering research-grade
retrieval quality and comprehensive safety coverage.

---

## Architecture

```
User query / symptoms
        │
        ▼
┌─────────────────────┐
│  Safety Layer        │  ← Emergency keyword detection (50+ phrases)
│  (rules.py)          │  ← Red-flag symptom-combination detection (9 combos)
└────────┬────────────┘
         │ safe
         ▼
┌─────────────────────┐
│  BM25 Retriever      │  ← Robertson et al. (1995), k1=1.5, b=0.75
│  (retriever.py)      │  ← Index built once at startup; O(1) per query
│                      │  ← 33-entry verified knowledge base
└────────┬────────────┘
         │ top-k scored docs
         ▼
┌─────────────────────┐
│  RAG Pipeline        │  ← BM25-score-calibrated confidence
│  (pipeline.py)       │  ← Context-grounded LLM prompt
│                      │  ← Reliable source citation (from retrieval, not LLM)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Groq LLM Client     │  ← async httpx, retry + exponential backoff
│  (groq_client.py)    │  ← Structured chain-of-thought system prompt
│                      │  ← Robust JSON extraction (3 fallback strategies)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Output Validator    │  ← Enum coercion, unsafe-term filtering
│  (validator.py)      │  ← Always-present disclaimer enforcement
└─────────────────────┘
```

---

## What's research-grade about it

| Dimension | Before | After |
|---|---|---|
| **Retrieval** | Raw token-overlap count | BM25 (Robertson et al., 1995) with IDF weighting and double-weighted symptoms |
| **Knowledge base** | 5 entries | 33 verified conditions (WHO, CDC, NIH, AHA/ACC, ATA, AAAAI, ACG, NEJM …) |
| **Confidence** | Naive doc count | BM25 score ratio — calibrated to retrieval quality |
| **Pipeline lifecycle** | Rebuilt on every request | Singleton via FastAPI `lifespan` — dataset loaded + index built once |
| **LLM I/O** | Sync, single-shot, no retry | Async (`httpx.AsyncClient`), retry + exponential backoff, robust JSON extraction |
| **LLM prompting** | Minimal instruction | Structured evidence-grounding prompt with chain-of-thought guidelines |
| **Source citations** | None | Attached directly from retrieved documents (not LLM hallucination) |
| **Emergency coverage** | 6 keyword phrases | 50+ keyword phrases across cardiac, respiratory, neurological, mental health, trauma |
| **Red-flag combos** | 3 | 9 (meningitis, MI, stroke, PE/DVT, sepsis, hypertensive crisis, GI emergency, …) |
| **Output validation** | Basic unsafe-term filter | Enum coercion, empty-field defaults, expanded unsafe-term list |
| **Observability** | Basic logging | Per-request timing logs, X-Request-ID middleware, structured fields |
| **Tests** | 5 | 62 across 4 test modules (retriever, pipeline, knowledge-base schema, safety, API) |
| **Frontend** | Static history (lost on refresh) | localStorage-persisted history, click-to-restore, confidence bar, source citations, Enter-key submit, spinner |

---

## Safety-first behaviour

- Does **not** diagnose or act as a licensed doctor
- Immediately escalates any emergency keyword to a high-severity response
- Detects 9 red-flag symptom combinations and forces urgent-care guidance
- Every response carries a mandatory disclaimer
- Possible conditions are grounded exclusively to the retrieved knowledge base
- Output is validated for enum correctness and unsafe-term content before returning

---

## Folder structure

```text
.
├── backend
│   ├── app
│   │   ├── api/routes.py            # async routes, singleton pipeline, timing logs
│   │   ├── models/schemas.py        # Pydantic models incl. sources field
│   │   ├── rag/
│   │   │   ├── dataset.py           # JSON loader (verified-only filter)
│   │   │   ├── embedder.py          # unused; retained for future embedding work
│   │   │   ├── pipeline.py          # async RAGPipeline with scored confidence
│   │   │   └── retriever.py         # pure-Python BM25 implementation
│   │   ├── safety/
│   │   │   ├── rules.py             # 50+ emergency keywords, 9 red-flag combos
│   │   │   └── validator.py         # output sanitisation and enum coercion
│   │   ├── services/groq_client.py  # async, retry, JSON extraction
│   │   ├── config.py
│   │   └── main.py                  # lifespan context, RequestID middleware
│   ├── data/sample_medical_knowledge.json   # 33-entry knowledge base
│   ├── tests/
│   │   ├── test_api.py              # 13 integration tests
│   │   ├── test_knowledge_base.py   # 8 schema validation tests
│   │   ├── test_pipeline.py         # 14 unit tests (mocked Groq)
│   │   ├── test_retriever.py        # 12 BM25 retrieval tests
│   │   └── test_safety.py           # 15 safety-rule tests
│   ├── Dockerfile
│   └── requirements.txt             # 6 lightweight deps — no ML frameworks
├── frontend/src/
│   ├── App.jsx                      # Enter-key submit, localStorage history, confidence bar, sources
│   └── App.css
├── render.yaml                      # Render Blueprint (free tier)
├── docker-compose.yml
└── .env.example
```

---

## API contract

### `POST /ask`

```json
{
  "query": "What could cause a persistent cough and fever?"
}
```

### `POST /symptom-check`

```json
{
  "symptoms": ["fever", "dry cough", "fatigue"]
}
```

### Response (both endpoints)

```json
{
  "possible_conditions": ["Influenza", "COVID-19"],
  "explanation": "2-3 sentences grounded in retrieved evidence...",
  "severity": "low | medium | high",
  "recommended_action": "conservative, evidence-based recommendation",
  "when_to_see_doctor": "specific actionable guidance",
  "confidence": "low | medium | high",
  "disclaimer": "This is not medical advice. Always consult a licensed healthcare provider.",
  "sources": ["CDC - Influenza (Flu)", "WHO/CDC - COVID-19 Clinical Guidance"]
}
```

### `GET /health`

```json
{
  "status": "ok",
  "rag_ready": true,
  "provider": "groq | fallback",
  "knowledge_entries": 33,
  "retriever_type": "BM25"
}
```

---

## Local setup

### 1) Backend

```bash
cd /path/to/Medical-Chatbot-
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env          # add your GROQ_API_KEY
PYTHONPATH=backend uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2) Frontend

```bash
cd frontend
npm install && npm run dev
```

Open `http://localhost:5173`.

### 3) Run tests

```bash
PYTHONPATH=backend python -m unittest discover -s backend/tests -v
```

---

## Docker deployment

```bash
cp .env.example .env    # add GROQ_API_KEY
docker compose up --build
```

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`

---

## Deploy to Render (free tier — ≤ 512 MB RAM)

The app fits comfortably in Render's 512 MB free tier — no ML frameworks,
no embedding models.  The BM25 index and 33-entry dataset occupy < 1 MB RAM.

**One-click deploy via `render.yaml`:**

1. Fork / push this repo to GitHub.
2. Log in to [render.com](https://render.com) → *New* → *Blueprint*.
3. Connect your repo — Render reads `render.yaml` and creates two services:
   - **medical-assistant-api** – Python web service (≈ 80 MB RAM)
   - **medical-assistant-ui** – Static site (0 RAM)
4. Set the secret env vars in the Render dashboard:

   | Service | Variable | Value |
   |---------|----------|-------|
   | `medical-assistant-api` | `GROQ_API_KEY` | your Groq API key |
   | `medical-assistant-ui`  | `VITE_API_BASE_URL` | backend service URL |

5. Click **Deploy**.

> The Render free tier spins down after 15 min of inactivity. First request may take ~30 s.

---

## Extension points

- **Embedding-based retrieval** — drop in `sentence-transformers` + FAISS when a larger tier is available; `Retriever` can be swapped without changing the pipeline interface
- **Vector database** — replace in-memory index with Pinecone / Qdrant for O(log N) retrieval over large corpora
- **Streaming responses** — FastAPI `StreamingResponse` + Groq streaming API
- **Patient history** — DB-backed session persistence for longitudinal context
- **Voice input** — browser `SpeechRecognition` API → query field
- **Multi-language** — system prompt localisation + locale selector

