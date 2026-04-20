from __future__ import annotations

import json
import logging

from app.rag.dataset import MedicalDataset
from app.rag.retriever import Retriever
from app.safety.rules import detect_emergency, detect_red_flags
from app.safety.validator import OutputValidator, SAFE_DISCLAIMER
from app.services.groq_client import GroqClient

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.

    Lifecycle
    ---------
    Build the pipeline once at application startup (via FastAPI lifespan) and
    reuse it across all requests.  The BM25 index and loaded dataset are
    retained in memory, avoiding repeated disk I/O and index rebuilds.
    """

    def __init__(self, dataset_path: str, top_k: int, groq_client: GroqClient) -> None:
        self.dataset = MedicalDataset(dataset_path)
        self.retriever = Retriever(self.dataset.entries)
        self.top_k = top_k
        self.groq_client = groq_client

    @property
    def ready(self) -> bool:
        return bool(self.dataset.entries)

    # ------------------------------------------------------------------
    # Context / confidence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context(docs: list[dict]) -> str:
        """
        Serialise retrieved documents into a compact JSON string that is
        injected verbatim into the LLM prompt.  All fields relevant to
        safe triage are included so the model can produce grounded responses.
        """
        return json.dumps(
            [
                {
                    "source": d.get("source"),
                    "condition": d.get("condition"),
                    "symptoms": d.get("symptoms", []),
                    "severity": d.get("severity", "medium"),
                    "explanation": d.get("explanation", ""),
                    "recommended_action": d.get("recommended_action"),
                    "when_to_see_doctor": d.get("when_to_see_doctor"),
                    "warnings": d.get("warnings", []),
                }
                for d in docs
            ],
            ensure_ascii=False,
        )

    @staticmethod
    def _confidence_from_scores(scores: list[float]) -> str:
        """
        Derive confidence from BM25 scores.

        High confidence  → top score is clearly dominant (ratio ≥ 2.5) AND
                           the absolute score indicates strong term overlap.
        Medium confidence → some term overlap but not clearly dominant.
        Low confidence   → very low absolute score (near-zero overlap).
        """
        if not scores or scores[0] <= 0.5:
            return "low"
        top = scores[0]
        if len(scores) >= 2 and scores[1] > 0:
            ratio = top / scores[1]
            if ratio >= 2.5 and top >= 5.0:
                return "high"
            if ratio >= 1.5 or top >= 4.0:
                return "medium"
            return "low"
        # Only one result (or all others scored 0)
        return "high" if top >= 5.0 else "medium"

    # ------------------------------------------------------------------
    # Fallback response (used when Groq is unavailable or returns None)
    # ------------------------------------------------------------------

    def _fallback_response(
        self,
        docs: list[dict],
        scores: list[float] | None = None,
        uncertain: bool = False,
    ) -> dict:
        conditions = [doc.get("condition", "Unknown") for doc in docs][:3]
        if not conditions:
            conditions = ["Insufficient information to identify a condition"]

        high_severity = any(d.get("severity") == "high" for d in docs)
        severity = "high" if high_severity else (docs[0].get("severity", "low") if docs else "low")

        if uncertain or not scores:
            confidence = "low"
        else:
            confidence = self._confidence_from_scores(scores)

        explanation = (
            "There is insufficient evidence in the knowledge base to identify a specific condition. "
            "A professional evaluation is recommended."
            if uncertain
            else (
                "Possible causes are inferred from verified medical references based on symptom overlap. "
                "This is not a diagnosis."
            )
        )

        sources = list({d.get("source", "") for d in docs if d.get("source")})

        return {
            "possible_conditions": conditions,
            "explanation": explanation,
            "severity": severity,
            "recommended_action": docs[0].get(
                "recommended_action",
                "Track symptoms and consult a licensed clinician if they persist.",
            )
            if docs
            else "Consult a licensed clinician.",
            "when_to_see_doctor": docs[0].get(
                "when_to_see_doctor",
                "If symptoms worsen or persist beyond 24–48 hours.",
            )
            if docs
            else "If symptoms persist or worsen.",
            "confidence": confidence,
            "disclaimer": SAFE_DISCLAIMER,
            "sources": sources,
        }

    # ------------------------------------------------------------------
    # Main answer method
    # ------------------------------------------------------------------

    async def answer(self, query: str, symptoms: list[str] | None = None) -> dict:
        symptoms = symptoms or []
        merged_text = " ".join([query] + symptoms)

        # ── 1. Hard emergency detection ────────────────────────────────
        if detect_emergency(merged_text):
            logger.info("Emergency keywords detected in query")
            return OutputValidator.sanitize(
                {
                    "possible_conditions": ["Potential medical emergency"],
                    "explanation": "Emergency symptoms were detected in your input.",
                    "severity": "high",
                    "recommended_action": "Seek emergency medical help immediately.",
                    "when_to_see_doctor": "Call emergency services (e.g. 911) now.",
                    "confidence": "high",
                    "disclaimer": SAFE_DISCLAIMER,
                    "sources": [],
                }
            )

        # ── 2. BM25 retrieval ──────────────────────────────────────────
        scored_docs = self.retriever.retrieve_scored(merged_text, self.top_k)
        scores = [s for s, _ in scored_docs]
        docs = [d for _, d in scored_docs]

        logger.debug(
            "Retrieved %d docs; top BM25 score=%.3f",
            len(docs),
            scores[0] if scores else 0,
        )

        # ── 3. Red-flag symptom combination check ─────────────────────
        if detect_red_flags(symptoms):
            logger.info("Red-flag symptom combination detected")
            flagged = self._fallback_response(docs or [], scores or [], uncertain=not bool(docs))
            flagged["severity"] = "high"
            flagged["recommended_action"] = (
                "Red-flag symptom combination detected. Seek urgent medical evaluation today."
            )
            flagged["when_to_see_doctor"] = "Go to urgent care or emergency services today."
            return OutputValidator.sanitize(flagged)

        # ── 4. No relevant docs → uncertain fallback ───────────────────
        if not docs or (scores and scores[0] <= 0):
            return OutputValidator.sanitize(self._fallback_response([], uncertain=True))

        # ── 5. LLM generation ──────────────────────────────────────────
        context = self._build_context(docs)
        try:
            model_response = await self.groq_client.generate(
                query=query,
                symptoms=symptoms,
                context=context,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("GroqClient raised unexpectedly: %s — using fallback", exc)
            model_response = None

        actual_sources = list({d.get("source", "") for d in docs if d.get("source")})

        if model_response is None:
            fallback = self._fallback_response(docs, scores)
            fallback["sources"] = actual_sources
            return OutputValidator.sanitize(fallback)

        # ── 6. Ground possible_conditions to retrieved set ────────────
        allowed_conditions = {d.get("condition", "") for d in docs}
        model_conds = model_response.get("possible_conditions", [])
        filtered = [c for c in model_conds if c in allowed_conditions]
        if not filtered:
            model_response["possible_conditions"] = [
                docs[0].get("condition", "Condition not identified")
            ]
        else:
            model_response["possible_conditions"] = filtered

        # ── 7. Prefix uncertainty marker when confidence is low ───────
        if model_response.get("confidence") == "low":
            expl = model_response.get("explanation", "")
            if not expl.startswith("I am not certain"):
                model_response["explanation"] = f"I am not certain. {expl}".strip()

        # Always attach reliable sources from the retrieved documents
        model_response["sources"] = actual_sources

        return OutputValidator.sanitize(model_response)
