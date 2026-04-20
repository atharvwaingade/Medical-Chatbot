# Personal Medical Assistant (RAG + Groq)

Production-oriented MVP for a **safe, retrieval-grounded medical information assistant**.

## Safety-first behavior

- Does **not** diagnose or act as a licensed doctor
- Uses verified retrieval context (sample WHO/CDC/PubMed/textbook-based dataset)
- Detects emergency and red-flag symptoms
- Returns strict structured JSON with disclaimer
- Uses conservative fallback responses when confidence is low

---

## Folder structure

```text
.
├── backend
│   ├── app
│   │   ├── api
│   │   │   └── routes.py
│   │   ├── models
│   │   │   └── schemas.py
│   │   ├── rag
│   │   │   ├── dataset.py
│   │   │   ├── embedder.py
│   │   │   ├── pipeline.py
│   │   │   └── retriever.py
│   │   ├── safety
│   │   │   ├── rules.py
│   │   │   └── validator.py
│   │   ├── services
│   │   │   └── groq_client.py
│   │   ├── config.py
│   │   └── main.py
│   ├── data
│   │   └── sample_medical_knowledge.json
│   ├── tests
│   │   ├── test_api.py
│   │   └── test_safety.py
│   ├── Dockerfile
│   └── requirements.txt
├── frontend
│   ├── src
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   ├── Dockerfile
│   └── package.json
├── .env.example
├── docker-compose.yml
└── README.md
```

---

## API response contract

```json
{
  "possible_conditions": [],
  "explanation": "",
  "severity": "low | medium | high",
  "recommended_action": "",
  "when_to_see_doctor": "",
  "confidence": "low | medium | high",
  "disclaimer": "This is not medical advice"
}
```

---

## Local setup

### 1) Backend

```bash
cd /home/runner/work/Medical-Chatbot-/Medical-Chatbot-
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env
PYTHONPATH=backend uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2) Frontend

```bash
cd /home/runner/work/Medical-Chatbot-/Medical-Chatbot-/frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

---

## Docker deployment

```bash
cd /home/runner/work/Medical-Chatbot-/Medical-Chatbot-
cp .env.example .env
docker compose up --build
```

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`

---

## Deploy to Render (free tier — ≤ 512 MB RAM)

The app is designed to fit comfortably within Render's 512 MB free tier.
Heavy ML packages (sentence-transformers, faiss-cpu, numpy) are not required —
the retrieval pipeline uses lightweight token-overlap scoring, and generation
is handled entirely by the external Groq API.

**One-click deploy via `render.yaml`:**

1. Fork / push this repo to GitHub.
2. Log in to [render.com](https://render.com) → *New* → *Blueprint*.
3. Connect your repo — Render reads `render.yaml` and creates two services:
   - **medical-assistant-api** – Python web service (backend, ≈ 80 MB RAM)
   - **medical-assistant-ui** – Static site (frontend, 0 RAM)
4. In the Render dashboard set the secret env vars:

   | Service | Variable | Value |
   |---------|----------|-------|
   | `medical-assistant-api` | `GROQ_API_KEY` | your Groq API key |
   | `medical-assistant-ui`  | `VITE_API_BASE_URL` | URL of the backend service (e.g. `https://medical-assistant-api.onrender.com`) |

5. Click **Deploy** — both services will be live within a few minutes.

> **Note:** The Render free tier spins down after 15 minutes of inactivity.
> The first request after a cold start may take ~30 s.

## Other deploy options

- Railway: deploy backend and frontend as separate services, set `.env` vars
- AWS: ECS/Fargate (two containers), ALB routing

---

## Bonus extension points

- Voice input (browser speech APIs)
- Multi-language output (prompt + locale selector)
- Patient history persistence (DB-backed)
- PDF export of structured response
