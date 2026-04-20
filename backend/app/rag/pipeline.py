from __future__ import annotations

import json
import logging

from app.rag.dataset import MedicalDataset
from app.rag.evidence_grader import confidence_bonus, grade_source, tier_description
from app.rag.query_processor import QueryProcessor
from app.rag.retriever import RetrievalResult, Retriever
from app.safety.rules import detect_emergency, detect_red_flags
from app.safety.validator import OutputValidator, SAFE_DISCLAIMER
from app.services.groq_client import GroqClient

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    MedRAG-Hybrid Pipeline
    ======================
    Five-stage pipeline for evidence-grounded medical information retrieval:

    1. **Query preprocessing** (QueryProcessor)
       Synonym normalization → NegEx negation detection → medical token extraction.

    2. **Safety gating**
       Emergency keyword and red-flag symptom-combination detection.

    3. **Multi-stage retrieval** (Retriever)
       BM25 + Symptom Coverage Score + RM3-PRF with entropy-based confidence.

    4. **Evidence-weighted context construction**
       Retrieved documents sorted by GRADE evidence tier before LLM injection.
       Higher-tier sources appear earlier in the context window (Liu et al.,
       2023 — "Lost in the Middle" attention bias).

    5. **LLM generation** (GroqClient)
       Async Groq call with session context injection for multi-turn dialogue.
       Hallucination guard: generated conditions are filtered to the retrieved set.
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
    # Context construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context(results: list[RetrievalResult]) -> str:
        """
        Serialize retrieved documents into a compact JSON context string.

        Documents are sorted by GRADE evidence tier (strongest first) so the
        LLM attends to the highest-quality evidence at the top of its context
        window (Liu et al., 2023).
        """
        # Sort by evidence tier ascending (tier 1 = strongest evidence)
        sorted_results = sorted(
            results,
            key=lambda r: grade_source(r.entry.get("source", "")),
        )
        return json.dumps(
            [
                {
                    "source": r.entry.get("source"),
                    "evidence_tier": grade_source(r.entry.get("source", "")),
                    "condition": r.entry.get("condition"),
                    "icd10": r.entry.get("icd10"),
                    "symptoms": r.entry.get("symptoms", []),
                    "severity": r.entry.get("severity", "medium"),
                    "explanation": r.entry.get("explanation", ""),
                    "recommended_action": r.entry.get("recommended_action"),
                    "when_to_see_doctor": r.entry.get("when_to_see_doctor"),
                    "warnings": r.entry.get("warnings", []),
                    "ruling_in": r.ruling_in,
                    "ruling_out": r.ruling_out,
                    "symptom_match_ratio": round(r.scs_score, 3),
                }
                for r in sorted_results
            ],
            ensure_ascii=False,
        )

    def _compute_confidence(
        self, results: list[RetrievalResult], scores: list[float]
    ) -> str:
        """
        Calibrate confidence using score entropy + best evidence tier.

        Entropy is computed over the top-k hybrid scores.  The best available
        GRADE tier bonus is added to the raw 1 - H_norm signal.
        """
        entropy = self.retriever.score_entropy(scores)
        best_tier = min(
            (grade_source(r.entry.get("source", "")) for r in results),
            default=5,
        )
        bonus = confidence_bonus(best_tier)
        return self.retriever.confidence_from_entropy(entropy, bonus)

    # ------------------------------------------------------------------
    # Fallback response (Groq unavailable or returned None)
    # ------------------------------------------------------------------

    def _fallback_response(
        self,
        results: list[RetrievalResult],
        scores: list[float] | None = None,
        uncertain: bool = False,
        pq=None,
    ) -> dict:
        docs = [r.entry for r in results]
        conditions = [doc.get("condition", "Unknown") for doc in docs][:3]
        if not conditions:
            conditions = ["Insufficient information to identify a condition"]

        high_severity = any(d.get("severity") == "high" for d in docs)
        severity = (
            "high"
            if high_severity
            else (docs[0].get("severity", "low") if docs else "low")
        )

        if uncertain or not scores:
            confidence = "low"
        else:
            confidence = self._compute_confidence(results, scores)

        explanation = (
            "There is insufficient evidence in the knowledge base to identify a "
            "specific condition. A professional evaluation is recommended."
            if uncertain
            else (
                "Possible causes inferred from verified medical references based on "
                "symptom overlap. This is not a diagnosis."
            )
        )

        sources = list({d.get("source", "") for d in docs if d.get("source")})

        ret: dict = {
            "possible_conditions": conditions,
            "explanation": explanation,
            "severity": severity,
            "recommended_action": (
                docs[0].get(
                    "recommended_action",
                    "Track symptoms and consult a licensed clinician.",
                )
                if docs
                else "Consult a licensed clinician."
            ),
            "when_to_see_doctor": (
                docs[0].get(
                    "when_to_see_doctor",
                    "If symptoms worsen or persist beyond 24–48 hours.",
                )
                if docs
                else "If symptoms persist or worsen."
            ),
            "confidence": confidence,
            "disclaimer": SAFE_DISCLAIMER,
            "sources": sources,
        }
        if pq is not None:
            ret["_negated"] = pq.negated_terms
            ret["_expanded"] = pq.normalized_terms
        return ret

    # ------------------------------------------------------------------
    # Main answer method
    # ------------------------------------------------------------------

    async def answer(
        self,
        query: str,
        symptoms: list[str] | None = None,
        session_context: str = "",
    ) -> dict:
        symptoms = symptoms or []
        merged_text = " ".join([query] + symptoms)

        # ── 1. Hard emergency detection ───────────────────────────────
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

        # ── 2. Query preprocessing ────────────────────────────────────
        pq = QueryProcessor.process(query, symptoms)
        logger.debug(
            "QueryProcessor: negated=%s normalized=%s",
            pq.negated_terms,
            pq.normalized_terms,
        )

        # ── 3. MedHybrid-BM25 retrieval ───────────────────────────────
        results = self.retriever.retrieve_results(
            pq.expanded_query, pq.medical_tokens, self.top_k
        )
        scores = [r.hybrid_score for r in results]
        docs = [r.entry for r in results]

        logger.debug(
            "Retrieved %d docs; top hybrid score=%.3f",
            len(docs),
            scores[0] if scores else 0,
        )

        # ── 4. Red-flag symptom combination check ─────────────────────
        if detect_red_flags(symptoms):
            logger.info("Red-flag symptom combination detected")
            flagged = self._fallback_response(results, scores, pq=pq)
            flagged["severity"] = "high"
            flagged["recommended_action"] = (
                "Red-flag symptom combination detected. Seek urgent evaluation today."
            )
            flagged["when_to_see_doctor"] = "Go to urgent care or emergency services today."
            return OutputValidator.sanitize(flagged)

        # ── 5. No relevant docs → uncertain fallback ──────────────────
        if not docs or (scores and scores[0] <= 0):
            return OutputValidator.sanitize(
                self._fallback_response([], uncertain=True, pq=pq)
            )

        # ── 6. LLM generation ─────────────────────────────────────────
        context = self._build_context(results)
        try:
            model_response = await self.groq_client.generate(
                query=query,
                symptoms=symptoms,
                context=context,
                session_context=session_context,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("GroqClient raised unexpectedly: %s — using fallback", exc)
            model_response = None

        actual_sources = list({d.get("source", "") for d in docs if d.get("source")})

        if model_response is None:
            fallback = self._fallback_response(results, scores, pq=pq)
            fallback["sources"] = actual_sources
            return OutputValidator.sanitize(fallback)

        # ── 7. Ground possible_conditions to retrieved set ────────────
        allowed_conditions = {d.get("condition", "") for d in docs}
        model_conds = model_response.get("possible_conditions", [])
        filtered = [c for c in model_conds if c in allowed_conditions]
        if not filtered:
            model_response["possible_conditions"] = [
                docs[0].get("condition", "Condition not identified")
            ]
        else:
            model_response["possible_conditions"] = filtered

        # ── 8. Prefix uncertainty marker when confidence is low ───────
        if model_response.get("confidence") == "low":
            expl = model_response.get("explanation", "")
            if not expl.startswith("I am not certain"):
                model_response["explanation"] = f"I am not certain. {expl}".strip()

        # Always attach reliable sources from retrieved documents
        model_response["sources"] = actual_sources

        return OutputValidator.sanitize(model_response)

    # ------------------------------------------------------------------
    # Differential diagnosis
    # ------------------------------------------------------------------

    async def differential(
        self,
        symptoms: list[str],
        query: str = "",
        top_k: int | None = None,
        session_context: str = "",
    ) -> dict:
        """
        Generate a ranked structured differential diagnosis.

        Returns all retrieved candidates with full scoring metadata,
        ruling-in/ruling-out symptoms, evidence tier, and ICD-10 codes.
        Suitable for display as a differential diagnosis table.
        """
        k = top_k or self.top_k

        # Preprocess
        pq = QueryProcessor.process(query, symptoms)
        merged_text = " ".join([query] + symptoms)

        # Safety check
        if detect_emergency(merged_text):
            return {
                "differentials": [],
                "query_metadata": {
                    "negated_terms": pq.negated_terms,
                    "expanded_terms": pq.normalized_terms,
                    "retrieval_entropy": 1.0,
                    "retriever_type": "MedHybrid-BM25+SCS+PRF",
                    "emergency": True,
                },
                "disclaimer": SAFE_DISCLAIMER,
            }

        results = self.retriever.retrieve_results(pq.expanded_query, pq.medical_tokens, k)
        scores = [r.hybrid_score for r in results]
        entropy = self.retriever.score_entropy(scores)

        differentials = []
        for rank, r in enumerate(results, 1):
            src = r.entry.get("source", "")
            tier = grade_source(src)
            differentials.append(
                {
                    "rank": rank,
                    "condition": r.entry.get("condition", "Unknown"),
                    "icd10": r.entry.get("icd10"),
                    "hybrid_score": round(r.hybrid_score, 4),
                    "symptom_match_ratio": round(r.scs_score, 3),
                    "evidence_tier": tier,
                    "evidence_tier_description": tier_description(tier),
                    "source": src,
                    "prevalence": r.entry.get("prevalence", "unknown"),
                    "ruling_in_symptoms": r.ruling_in,
                    "ruling_out_symptoms": r.ruling_out,
                    "severity": r.entry.get("severity", "medium"),
                    "recommended_action": r.entry.get("recommended_action", ""),
                }
            )

        return {
            "differentials": differentials,
            "query_metadata": {
                "negated_terms": pq.negated_terms,
                "expanded_terms": pq.normalized_terms,
                "retrieval_entropy": round(entropy, 4),
                "retriever_type": "MedHybrid-BM25+SCS+PRF",
            },
            "disclaimer": SAFE_DISCLAIMER,
        }
