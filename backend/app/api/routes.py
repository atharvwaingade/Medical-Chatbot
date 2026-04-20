from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.models.schemas import AskRequest, AssistantResponse, HealthResponse, SymptomCheckRequest
from app.rag.pipeline import RAGPipeline
from app.services.groq_client import GroqClient

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)):
    pipeline = RAGPipeline(
        dataset_path=settings.medical_dataset_path,
        top_k=settings.top_k,
        groq_client=GroqClient(api_key=settings.groq_api_key, model=settings.groq_model),
    )
    return {
        "status": "ok",
        "rag_ready": pipeline.ready,
        "model_provider": "groq" if settings.groq_api_key else "fallback",
    }


@router.post("/ask", response_model=AssistantResponse)
def ask(payload: AskRequest, settings: Settings = Depends(get_settings)):
    pipeline = RAGPipeline(
        dataset_path=settings.medical_dataset_path,
        top_k=settings.top_k,
        groq_client=GroqClient(api_key=settings.groq_api_key, model=settings.groq_model),
    )
    return pipeline.answer(payload.query)


@router.post("/symptom-check", response_model=AssistantResponse)
def symptom_check(payload: SymptomCheckRequest, settings: Settings = Depends(get_settings)):
    pipeline = RAGPipeline(
        dataset_path=settings.medical_dataset_path,
        top_k=settings.top_k,
        groq_client=GroqClient(api_key=settings.groq_api_key, model=settings.groq_model),
    )
    return pipeline.answer(query="Symptom check", symptoms=payload.symptoms)
