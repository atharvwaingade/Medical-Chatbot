from __future__ import annotations

import json

from app.rag.dataset import MedicalDataset
from app.rag.retriever import Retriever
from app.safety.rules import detect_emergency, detect_red_flags
from app.safety.validator import OutputValidator, SAFE_DISCLAIMER
from app.services.groq_client import GroqClient


class RAGPipeline:
    def __init__(self, dataset_path: str, top_k: int, groq_client: GroqClient):
        self.dataset = MedicalDataset(dataset_path)
        self.retriever = Retriever(self.dataset.entries)
        self.top_k = top_k
        self.groq_client = groq_client

    @property
    def ready(self) -> bool:
        return bool(self.dataset.entries)

    def _build_context(self, docs: list[dict]) -> str:
        return json.dumps(
            [
                {
                    "source": d.get("source"),
                    "condition": d.get("condition"),
                    "symptoms": d.get("symptoms", []),
                    "severity": d.get("severity", "medium"),
                    "recommended_action": d.get("recommended_action"),
                    "when_to_see_doctor": d.get("when_to_see_doctor"),
                }
                for d in docs
            ]
        )

    @staticmethod
    def _confidence_from_docs(docs: list[dict]) -> str:
        if len(docs) >= 3:
            return "high"
        if len(docs) == 2:
            return "medium"
        return "low"

    def _fallback_response(self, docs: list[dict], uncertain: bool = False) -> dict:
        conditions = [doc.get("condition", "Unknown") for doc in docs][:3]
        if not conditions:
            conditions = ["No strong match found"]

        high_severity = any(d.get("severity") == "high" for d in docs)
        severity = "high" if high_severity else (docs[0].get("severity", "low") if docs else "low")
        confidence = "low" if uncertain else self._confidence_from_docs(docs)

        explanation = (
            "I am not certain. Based on limited verified context, consider a professional evaluation."
            if uncertain
            else "Possible causes are inferred from verified medical references and symptom overlap."
        )

        return {
            "possible_conditions": conditions,
            "explanation": explanation,
            "severity": severity,
            "recommended_action": docs[0].get("recommended_action", "Track symptoms and consult a doctor if they persist."),
            "when_to_see_doctor": docs[0].get("when_to_see_doctor", "If symptoms worsen or persist beyond 24-48 hours."),
            "confidence": confidence,
            "disclaimer": SAFE_DISCLAIMER,
        }

    def answer(self, query: str, symptoms: list[str] | None = None) -> dict:
        symptoms = symptoms or []
        merged_text = " ".join([query] + symptoms)

        if detect_emergency(merged_text):
            return {
                "possible_conditions": ["Potential medical emergency"],
                "explanation": "Emergency symptoms were detected from your input.",
                "severity": "high",
                "recommended_action": "Seek emergency medical help immediately.",
                "when_to_see_doctor": "Call emergency services now.",
                "confidence": "high",
                "disclaimer": SAFE_DISCLAIMER,
            }

        docs = self.retriever.retrieve(merged_text, self.top_k)

        if detect_red_flags(symptoms):
            flagged = self._fallback_response(docs or [], uncertain=not bool(docs))
            flagged["severity"] = "high"
            flagged["recommended_action"] = "Red-flag symptom combination detected. Seek urgent medical evaluation today."
            flagged["when_to_see_doctor"] = "Go to urgent care or emergency services today."
            return OutputValidator.sanitize(flagged)

        if not docs:
            return OutputValidator.sanitize(self._fallback_response([], uncertain=True))

        prompt = (
            "User query: "
            + query
            + "\nSymptoms: "
            + ", ".join(symptoms)
            + "\nUse only this retrieved context: "
            + self._build_context(docs)
            + "\nRules: no diagnosis, cite uncertainty when needed, safe recommendation only."
        )

        model_response = self.groq_client.generate(prompt)

        if model_response is None:
            return OutputValidator.sanitize(self._fallback_response(docs))

        allowed_conditions = {d.get("condition", "") for d in docs}
        model_conds = model_response.get("possible_conditions", [])
        filtered = [c for c in model_conds if c in allowed_conditions]
        if not filtered:
            model_response["possible_conditions"] = [docs[0].get("condition", "No strong match found")]
        else:
            model_response["possible_conditions"] = filtered

        if model_response.get("confidence") == "low":
            model_response["explanation"] = f"I am not certain. {model_response.get('explanation', '')}".strip()

        return OutputValidator.sanitize(model_response)
