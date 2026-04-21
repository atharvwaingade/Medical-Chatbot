from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    session_id: str | None = None


class SymptomCheckRequest(BaseModel):
    symptoms: list[str] = Field(min_length=1)
    age: int | None = Field(default=None, ge=0, le=120)
    session_id: str | None = None


class DifferentialRequest(BaseModel):
    symptoms: list[str] = Field(min_length=1)
    query: str = Field(default="", max_length=500)
    age: int | None = Field(default=None, ge=0, le=120)
    session_id: str | None = None


class RetrievalMetadata(BaseModel):
    negated_terms: list[str]
    expanded_terms: list[str]
    retrieval_entropy: float
    retriever_type: str = "MedHybrid-BM25+SCS+PRF"


class ConformalPrediction(BaseModel):
    """Coverage-guaranteed prediction set (Vovk et al., 2005)."""

    prediction_set: list[str]
    coverage_level: float
    threshold: float
    set_size: int
    calibration_n: int


class AssistantResponse(BaseModel):
    possible_conditions: list[str]
    explanation: str
    severity: Literal["low", "medium", "high"]
    recommended_action: str
    when_to_see_doctor: str
    confidence: Literal["low", "medium", "high"]
    disclaimer: str
    sources: list[str] | None = None
    session_id: str | None = None
    retrieval_metadata: RetrievalMetadata | None = None
    # ── Novel research contributions ──────────────────────────────────────
    symptom_attributions: dict[str, float] | None = None
    """LOO Shapley attribution of each symptom token to the top diagnosis."""
    next_question: str | None = None
    """Information-theoretically optimal follow-up question to ask the patient."""
    discriminating_symptom: str | None = None
    """The canonical symptom behind next_question (machine-readable)."""
    expected_information_gain: float | None = None
    """Expected entropy reduction from asking next_question (0–1)."""
    conformal_prediction: ConformalPrediction | None = None
    """Prediction set with mathematical 90% coverage guarantee."""


class DifferentialEntry(BaseModel):
    rank: int
    condition: str
    icd10: str | None = None
    hybrid_score: float
    symptom_match_ratio: float
    prevalence_score: float = 0.0
    causal_score: float = 0.0
    evidence_tier: int
    evidence_tier_description: str
    source: str
    prevalence: str  # "common" | "uncommon" | "rare"
    ruling_in_symptoms: list[str]
    ruling_out_symptoms: list[str]
    severity: str
    recommended_action: str
    syndrome_cluster: str = ""
    organ_system: str = ""
    aetiology_class: str = ""


class ContrastEntry(BaseModel):
    """Pairwise symptom contrast between two adjacent conditions."""

    condition_a: str
    condition_b: str
    for_a: list[str]
    for_b: list[str]
    shared: list[str]
    patient_discriminating_a: list[str]
    patient_discriminating_b: list[str]
    ambiguous: bool


class DifferentialResponse(BaseModel):
    differentials: list[DifferentialEntry]
    query_metadata: RetrievalMetadata
    disclaimer: str
    session_id: str | None = None
    contrasts: list[ContrastEntry] | None = None
    """Pairwise contrastive DDx analysis for top adjacent condition pairs."""


class HealthResponse(BaseModel):
    status: str
    rag_ready: bool
    provider: str
    knowledge_entries: int
    retriever_type: str
    active_sessions: int
