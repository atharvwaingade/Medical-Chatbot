from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Request

from app.config import get_settings
from app.models.schemas import (
    AskRequest,
    AssistantResponse,
    DifferentialRequest,
    DifferentialResponse,
    HealthResponse,
    RetrievalMetadata,
    SymptomCheckRequest,
)
from app.rag.query_processor import QueryProcessor
from app.rag.retriever import Retriever

router = APIRouter()
logger = logging.getLogger(__name__)


def _pipeline(request: Request):
    return request.app.state.pipeline


def _store(request: Request):
    return request.app.state.session_store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _retrieval_metadata(query: str, symptoms: list[str], retriever: Retriever) -> RetrievalMetadata:
    """Build RetrievalMetadata for a given query without triggering a full pipeline call."""
    pq = QueryProcessor.process(query, symptoms)
    results = retriever.retrieve_results(pq.expanded_query, pq.medical_tokens, top_k=3)
    scores = [r.hybrid_score for r in results]
    entropy = retriever.score_entropy(scores) if scores else 1.0
    return RetrievalMetadata(
        negated_terms=pq.negated_terms,
        expanded_terms=pq.normalized_terms,
        retrieval_entropy=round(entropy, 4),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    pipeline = _pipeline(request)
    store = _store(request)
    settings = get_settings()
    return {
        "status": "ok",
        "rag_ready": pipeline.ready,
        "provider": "groq" if settings.groq_api_key else "fallback",
        "knowledge_entries": len(pipeline.dataset.entries),
        "retriever_type": "MedHybrid-BM25+SCS+PRF",
        "active_sessions": len(store),
    }


@router.post("/ask", response_model=AssistantResponse)
async def ask(payload: AskRequest, request: Request):
    t0 = time.perf_counter()
    pipeline = _pipeline(request)
    store = _store(request)
    rid = getattr(request.state, "request_id", "-")

    # Session context injection
    sid, session = store.get_or_create(payload.session_id)
    session_ctx = session.context_summary()

    result = await pipeline.answer(payload.query, session_context=session_ctx)

    # Record turn for future context
    store.record_turn(sid, payload.query, [], result)

    # Attach retrieval metadata
    pq = QueryProcessor.process(payload.query)
    results = pipeline.retriever.retrieve_results(pq.expanded_query, pq.medical_tokens, 3)
    scores = [r.hybrid_score for r in results]
    entropy = pipeline.retriever.score_entropy(scores) if scores else 1.0
    result["retrieval_metadata"] = {
        "negated_terms": pq.negated_terms,
        "expanded_terms": pq.normalized_terms,
        "retrieval_entropy": round(entropy, 4),
        "retriever_type": "MedHybrid-BM25+SCS+PRF",
    }
    result["session_id"] = sid

    logger.info(
        "ask rid=%s sid=%s q=%.60r elapsed_ms=%.1f conditions=%s entropy=%.3f",
        rid,
        sid,
        payload.query,
        (time.perf_counter() - t0) * 1000,
        result.get("possible_conditions", []),
        entropy,
    )
    return result


@router.post("/symptom-check", response_model=AssistantResponse)
async def symptom_check(payload: SymptomCheckRequest, request: Request):
    t0 = time.perf_counter()
    pipeline = _pipeline(request)
    store = _store(request)
    rid = getattr(request.state, "request_id", "-")

    sid, session = store.get_or_create(payload.session_id)
    session_ctx = session.context_summary()

    result = await pipeline.answer(
        query="Symptom check",
        symptoms=payload.symptoms,
        session_context=session_ctx,
    )

    store.record_turn(sid, "Symptom check", payload.symptoms, result)

    pq = QueryProcessor.process("", payload.symptoms)
    results = pipeline.retriever.retrieve_results(pq.expanded_query, pq.medical_tokens, 3)
    scores = [r.hybrid_score for r in results]
    entropy = pipeline.retriever.score_entropy(scores) if scores else 1.0
    result["retrieval_metadata"] = {
        "negated_terms": pq.negated_terms,
        "expanded_terms": pq.normalized_terms,
        "retrieval_entropy": round(entropy, 4),
        "retriever_type": "MedHybrid-BM25+SCS+PRF",
    }
    result["session_id"] = sid

    logger.info(
        "symptom-check rid=%s sid=%s symptoms=%s elapsed_ms=%.1f conditions=%s",
        rid,
        sid,
        payload.symptoms,
        (time.perf_counter() - t0) * 1000,
        result.get("possible_conditions", []),
    )
    return result


@router.post("/differential", response_model=DifferentialResponse)
async def differential(payload: DifferentialRequest, request: Request):
    t0 = time.perf_counter()
    pipeline = _pipeline(request)
    store = _store(request)
    rid = getattr(request.state, "request_id", "-")

    sid, session = store.get_or_create(payload.session_id)
    session_ctx = session.context_summary()

    result = await pipeline.differential(
        symptoms=payload.symptoms,
        query=payload.query,
        top_k=pipeline.top_k,
        session_context=session_ctx,
    )
    result["session_id"] = sid

    logger.info(
        "differential rid=%s sid=%s symptoms=%s elapsed_ms=%.1f n_differentials=%d",
        rid,
        sid,
        payload.symptoms,
        (time.perf_counter() - t0) * 1000,
        len(result.get("differentials", [])),
    )
    return result
