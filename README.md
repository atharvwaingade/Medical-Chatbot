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

## Deploy options

- Railway: deploy backend and frontend as separate services, set `.env` vars
- Render: web service for backend + static site for frontend
- AWS: ECS/Fargate (two containers), ALB routing

---

## Bonus extension points

- Voice input (browser speech APIs)
- Multi-language output (prompt + locale selector)
- Patient history persistence (DB-backed)
- PDF export of structured response
