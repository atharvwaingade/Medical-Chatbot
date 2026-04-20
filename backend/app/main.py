from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.routes import router
from app.config import get_settings
from app.rag.pipeline import RAGPipeline
from app.services.groq_client import GroqClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a short request-ID to every request/response for traceability."""

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:8]
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Build the RAGPipeline exactly once at startup and store it in
    ``app.state.pipeline``.  This avoids reloading the knowledge base
    and rebuilding the BM25 index on every request.
    """
    settings = get_settings()
    logger.info(
        "Initialising RAGPipeline — dataset=%s top_k=%d model=%s",
        settings.medical_dataset_path,
        settings.top_k,
        settings.groq_model,
    )
    pipeline = RAGPipeline(
        dataset_path=settings.medical_dataset_path,
        top_k=settings.top_k,
        groq_client=GroqClient(api_key=settings.groq_api_key, model=settings.groq_model),
    )
    logger.info(
        "RAGPipeline ready — %d knowledge entries loaded",
        len(pipeline.dataset.entries),
    )
    app.state.pipeline = pipeline
    yield
    logger.info("RAGPipeline shutdown")


app = FastAPI(
    title="Personal Medical Assistant API",
    version="2.0.0",
    description=(
        "Evidence-grounded medical information assistant using BM25 retrieval "
        "and Groq LLM generation.  Not a substitute for professional medical advice."
    ),
    lifespan=lifespan,
)

app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
