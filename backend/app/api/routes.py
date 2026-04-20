from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Request

from app.config import get_settings
from app.models.schemas import AskRequest, AssistantResponse, HealthResponse, SymptomCheckRequest

router = APIRouter()
logger = logging.getLogger(__name__)


def _pipeline(request: Request):
    return request.app.state.pipeline


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    pipeline = _pipeline(request)
    settings = get_settings()
    return {
        "status": "ok",
        "rag_ready": pipeline.ready,
        "provider": "groq" if settings.groq_api_key else "fallback",
        "knowledge_entries": len(pipeline.dataset.entries),
        "retriever_type": "BM25",
    }


@router.post("/ask", response_model=AssistantResponse)
async def ask(payload: AskRequest, request: Request):
    t0 = time.perf_counter()
    pipeline = _pipeline(request)
    result = await pipeline.answer(payload.query)
    rid = getattr(request.state, "request_id", "-")
    logger.info(
        "ask rid=%s q=%.60r elapsed_ms=%.1f conditions=%s",
        rid,
        payload.query,
        (time.perf_counter() - t0) * 1000,
        result.get("possible_conditions", []),
    )
    return result


@router.post("/symptom-check", response_model=AssistantResponse)
async def symptom_check(payload: SymptomCheckRequest, request: Request):
    t0 = time.perf_counter()
    pipeline = _pipeline(request)
    result = await pipeline.answer(query="Symptom check", symptoms=payload.symptoms)
    rid = getattr(request.state, "request_id", "-")
    logger.info(
        "symptom-check rid=%s symptoms=%s elapsed_ms=%.1f conditions=%s",
        rid,
        payload.symptoms,
        (time.perf_counter() - t0) * 1000,
        result.get("possible_conditions", []),
    )
    return result
