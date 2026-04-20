from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(min_length=2)


class SymptomCheckRequest(BaseModel):
    symptoms: list[str] = Field(min_length=1)
    age: int | None = Field(default=None, ge=0, le=120)


class AssistantResponse(BaseModel):
    possible_conditions: list[str]
    explanation: str
    severity: Literal["low", "medium", "high"]
    recommended_action: str
    when_to_see_doctor: str
    confidence: Literal["low", "medium", "high"]
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    rag_ready: bool
    provider: str
