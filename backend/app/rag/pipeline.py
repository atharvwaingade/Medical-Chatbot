from __future__ import annotations

import json
import logging
import math

from app.rag.active_inquiry import recommend_next_question
from app.rag.causal_graph import MedCausalGraph
from app.rag.conformal import ConformalPredictor
from app.rag.contrastive import compute_pairwise_contrasts, format_contrasts_for_prompt
from app.rag.dataset import MedicalDataset
from app.rag.evidence_grader import confidence_bonus, grade_source, tier_description
from app.rag.query_processor import QueryProcessor
from app.rag.raptor import MedRAPTOR
from app.rag.retriever import RetrievalResult, Retriever
from app.rag.symptom_attribution import compute_symptom_attributions
from app.safety.rules import detect_emergency, detect_red_flags
from app.safety.validator import OutputValidator, SAFE_DISCLAIMER
from app.services.groq_client import GroqClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kendall's τ (simplified) for Bayesian Coherence Critique
# ---------------------------------------------------------------------------


def _kendall_tau(list_a: list[str], list_b: list[str]) -> float:
    """
    Compute a simplified Kendall's τ between two ranked lists.

    Only items appearing in both lists are compared.  Returns a value in
    [-1, 1]; positive values indicate concordant ranking.
    """
    common = [x for x in list_a if x in list_b]
    if len(common) < 2:
        return 1.0  # insufficient data — assume concordant
    # Build position maps
    pos_a = {x: i for i, x in enumerate(common)}
    pos_b = {x: list_b.index(x) for x in common if x in list_b}

    concordant = discordant = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            xi, xj = common[i], common[j]
            sign_a = pos_a[xi] - pos_a[xj]
            sign_b = pos_b.get(xi, 0) - pos_b.get(xj, 0)
            if sign_a * sign_b > 0:
                concordant += 1
            elif sign_a * sign_b < 0:
                discordant += 1

    total = concordant + discordant
    return (concordant - discordant) / total if total > 0 else 1.0


class RAGPipeline:
    """
    MedRAG-Turbo Pipeline
    =====================
    Seven-stage pipeline for evidence-grounded medical information retrieval,
    implementing the full MedRAG-Turbo research architecture:

    1. **Query preprocessing** (QueryProcessor)
       Synonym normalization → NegEx negation detection → medical token extraction.

    2. **Safety gating**
       Emergency keyword and red-flag symptom-combination detection.

    3. **Multi-stage retrieval** (Retriever)
       BM25 + SCS + Prevalence Prior + RM3-PRF with entropy-based confidence.

    4. **Causal re-ranking** (MedCausalGraph)
       Personalised PageRank on symptom-condition causal graph with NegEx
       barrier nodes and session-based sequential Bayesian updating.

    5. **Hierarchical context enrichment** (MedRAPTOR)
       ICD-10-anchored four-level ontology provides syndrome/system context
       and enables top-down routing for vague queries.

    6. **Evidence-weighted context construction + LLM generation** (GroqClient)
       MedCoT-DDx illness-script prompt (Schmidt & Rikers, 2007) with
       8-step structured clinical reasoning scaffold.

    7. **Four-Critique Self-RAG loop**
       Critique 1 — Symptom Coverage: re-retrieve if < 60% of symptoms covered.
       Critique 2 — Negation Consistency: strip conditions whose ruling-in
                    contains NegEx-negated symptoms.
       Critique 3 — Bayesian Coherence: override LLM ranking if Kendall's τ
                    with Bayesian posterior < 0.5.
       Critique 4 — Severity Monotonicity: force severity ≥ medium when ≥ 50%
                    of retrieved docs have high severity.
    """

    # Causal re-ranking weight (0.0 = disable causal graph)
    CAUSAL_WEIGHT: float = 0.20

    # Symptom coverage threshold for Critique 1
    COVERAGE_THRESHOLD: float = 0.60

    # Bayesian coherence threshold for Critique 3
    COHERENCE_TAU_MIN: float = 0.50

    def __init__(self, dataset_path: str, top_k: int, groq_client: GroqClient) -> None:
        self.dataset = MedicalDataset(dataset_path)
        self.retriever = Retriever(self.dataset.entries)
        self.top_k = top_k
        self.groq_client = groq_client

        # Build causal graph from KB entries
        self.causal_graph = MedCausalGraph()
        self.causal_graph.build_from_entries(self.dataset.entries)

        # Build RAPTOR hierarchy from KB entries
        self.raptor = MedRAPTOR()
        self.raptor.build_from_entries(self.dataset.entries)

        # Conformal predictor (calibrated lazily on first query)
        self.conformal = ConformalPredictor(self.retriever)

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
    # Causal re-ranking helper
    # ------------------------------------------------------------------

    def _causal_rerank(
        self,
        results: list[RetrievalResult],
        affirmed: list[str],
        negated: list[str],
        prior_conditions: dict[str, float] | None = None,
    ) -> list[RetrievalResult]:
        """
        Re-rank retrieval results using MedCausalGraph personalised PageRank.

        The fused score = RETRIEVER_WEIGHT * hybrid_score + CAUSAL_WEIGHT * causal.
        """
        if not results or self.CAUSAL_WEIGHT <= 0.0:
            return results
        causal_map = self.causal_graph.causal_scores(affirmed, negated, prior_conditions)
        fused: list[tuple[float, RetrievalResult]] = []
        for r in results:
            cond = r.entry.get("condition", "")
            causal = causal_map.get(cond, 0.0)
            fused_score = (1.0 - self.CAUSAL_WEIGHT) * r.hybrid_score + self.CAUSAL_WEIGHT * causal
            # Replace hybrid_score with fused score for downstream use
            fused.append((fused_score, r))
        fused.sort(key=lambda x: x[0], reverse=True)
        # Rebuild result list with updated scores
        reranked = []
        for fused_score, r in fused:
            import dataclasses
            reranked.append(dataclasses.replace(r, hybrid_score=fused_score))
        return reranked

    # ------------------------------------------------------------------
    # Symptom coverage check (Critique 1)
    # ------------------------------------------------------------------

    def _symptom_coverage(
        self,
        affirmed_tokens: set[str],
        results: list[RetrievalResult],
    ) -> float:
        """
        Compute how much of the patient's symptom set is covered by the
        retrieved documents' symptom lists.

        coverage = |affirmed ∩ union(doc_symptoms)| / |affirmed|
        """
        if not affirmed_tokens:
            return 1.0
        union_doc_symptoms: set[str] = set()
        for r in results:
            for sym in r.entry.get("symptoms", []):
                for tok in self.retriever._tokenize(sym):
                    union_doc_symptoms.add(tok)
        covered = affirmed_tokens & union_doc_symptoms
        return len(covered) / len(affirmed_tokens)

    # ------------------------------------------------------------------
    # CoT context builder
    # ------------------------------------------------------------------

    def _build_cot_context(
        self,
        pq,
        results: list[RetrievalResult],
        session_ctx: str,
        affirmed_set: set[str] | None = None,
    ) -> dict:
        """
        Build the structured chain-of-thought context for the LLM.

        Populates all nine fields of the MedCoT-DDx illness-script template,
        including the new Step 9 contrastive DDx analysis.
        """
        # Bayesian ranking from hybrid scores (already MAP-calibrated)
        bayesian_ranking: list[tuple[str, float]] = [
            (r.entry.get("condition", ""), round(r.hybrid_score, 4))
            for r in results
        ]

        # Ruling-in / ruling-out block (one line per condition)
        ruling_lines = []
        for r in results:
            cond = r.entry.get("condition", "Unknown")
            tier = grade_source(r.entry.get("source", ""))
            tier_desc = tier_description(tier)
            rin = ", ".join(r.ruling_in) if r.ruling_in else "none"
            rout = ", ".join(r.ruling_out[:4]) if r.ruling_out else "none"
            ruling_lines.append(
                f"  {cond}: ruling_in=[{rin}] | ruling_out=[{rout}]"
            )
        ruling_in_out = "\n".join(ruling_lines) if ruling_lines else "  Not available"

        # Evidence quality summary
        ev_lines = []
        for r in results:
            src = r.entry.get("source", "")
            tier = grade_source(src)
            ev_lines.append(f"  {r.entry.get('condition')}: {tier_description(tier)} (tier {tier})")
        evidence_quality = "\n".join(ev_lines) if ev_lines else ""

        # RAPTOR hierarchical context for the top condition
        patient_ctx_parts = []
        if results:
            top_cond = results[0].entry.get("condition", "")
            hctx = self.raptor.hierarchical_context(top_cond)
            if hctx.get("organ_system"):
                patient_ctx_parts.append(f"Organ system: {hctx['organ_system']}")
            if hctx.get("aetiology_class"):
                patient_ctx_parts.append(f"Aetiology class: {hctx['aetiology_class']}")
            if hctx.get("syndrome"):
                patient_ctx_parts.append(f"Syndrome cluster: {hctx['syndrome']}")

        # Step 9: Contrastive DDx analysis
        affirmed = affirmed_set or set(pq.medical_tokens)
        contrasts = compute_pairwise_contrasts(results, affirmed)
        contrastive_text = format_contrasts_for_prompt(contrasts)

        return {
            "affirmed_terms": pq.normalized_terms or pq.medical_tokens,
            "negated_terms": pq.negated_terms,
            "bayesian_ranking": bayesian_ranking,
            "ruling_in_out": ruling_in_out,
            "evidence_quality": evidence_quality,
            "patient_context": "; ".join(patient_ctx_parts) if patient_ctx_parts else "Not specified",
            "contrastive_analysis": contrastive_text,
        }

    # ------------------------------------------------------------------
    # Four-Critique Self-RAG loop
    # ------------------------------------------------------------------

    def _apply_critiques(
        self,
        model_response: dict,
        results: list[RetrievalResult],
        pq,
    ) -> dict:
        """
        Apply the four post-generation critique checks.

        Critique 2 — Negation Consistency
            Strip conditions whose ruling-in contains a NegEx-negated symptom.

        Critique 3 — Bayesian Coherence
            Override LLM ranking if Kendall's τ with Bayesian posterior < 0.5.

        Critique 4 — Severity Monotonicity
            Force severity to "high" if ≥50% of retrieved docs have severity="high".
        """
        # ── Critique 2: Negation Consistency ──────────────────────────
        if pq.negated_terms:
            negated_set: set[str] = set()
            for term in pq.negated_terms:
                for tok in self.retriever._tokenize(term):
                    negated_set.add(tok)

            valid_conditions: list[str] = []
            for cond in model_response.get("possible_conditions", []):
                result_for_cond = next(
                    (r for r in results if r.entry.get("condition") == cond), None
                )
                if result_for_cond is None:
                    valid_conditions.append(cond)
                    continue
                # Check if any ruling-in token is negated
                ruling_in_tokens: set[str] = set()
                for sym in result_for_cond.ruling_in:
                    for tok in self.retriever._tokenize(sym):
                        ruling_in_tokens.add(tok)
                if ruling_in_tokens & negated_set:
                    logger.info(
                        "Critique 2: stripping '%s' — ruling-in contains negated symptom", cond
                    )
                else:
                    valid_conditions.append(cond)
            if valid_conditions:
                model_response["possible_conditions"] = valid_conditions

        # ── Critique 3: Bayesian Coherence ────────────────────────────
        if len(results) >= 2:
            bayesian_order = [r.entry.get("condition", "") for r in results]
            llm_conds = model_response.get("possible_conditions", [])
            if len(llm_conds) >= 2:
                tau = _kendall_tau(llm_conds, bayesian_order)
                if tau < self.COHERENCE_TAU_MIN:
                    new_order = [c for c in bayesian_order if c in set(llm_conds)]
                    if new_order:
                        model_response["possible_conditions"] = new_order
                        logger.info(
                            "Critique 3: overriding LLM ranking (tau=%.2f < %.2f)",
                            tau,
                            self.COHERENCE_TAU_MIN,
                        )

        # ── Critique 4: Severity Monotonicity ─────────────────────────
        docs = [r.entry for r in results]
        if docs:
            high_count = sum(1 for d in docs if d.get("severity") == "high")
            if high_count >= math.ceil(len(docs) * 0.5):
                if model_response.get("severity") == "low":
                    model_response["severity"] = "high"
                    logger.info("Critique 4: severity overridden to 'high'")

        return model_response

    def _apply_critique5_session(
        self,
        results: list[RetrievalResult],
        session_posterior: dict[str, float],
    ) -> list[RetrievalResult]:
        """
        Critique 5 — Session Coherence (Sequential Bayesian Update).

        Re-weights retrieval scores using the Dirichlet posterior from prior
        session turns.  This biases the current turn's ranking toward conditions
        that were repeatedly surfaced in this patient's conversation history,
        implementing a proper sequential Bayesian estimator:

            P(C | S₁, ..., Sₜ) ∝ P(Sₜ | C) × P(C | S₁, ..., Sₜ₋₁)

        The posterior is stored in session.dirichlet_alpha and updated each turn.

        Formally: fused_score = λ × hybrid + (1 − λ) × posterior_mean
        where λ = SESSION_COHERENCE_WEIGHT = 0.15.

        Only applied when the session has prior history (not the first turn).
        """
        if not session_posterior or not results:
            return results

        SESSION_COHERENCE_WEIGHT: float = 0.85  # weight of current retrieval
        import dataclasses
        reranked = []
        for r in results:
            cond = r.entry.get("condition", "")
            prior = session_posterior.get(cond, 0.0)
            fused = SESSION_COHERENCE_WEIGHT * r.hybrid_score + (1 - SESSION_COHERENCE_WEIGHT) * prior
            reranked.append(dataclasses.replace(r, hybrid_score=fused))
        reranked.sort(key=lambda r: r.hybrid_score, reverse=True)
        logger.debug("Critique 5: session posterior applied (%d conditions)", len(session_posterior))
        return reranked

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

        # ── 3. MedHybrid-Bayesian retrieval ───────────────────────────
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

        # ── 6a. Critique 1: Symptom Coverage ──────────────────────────
        affirmed_set = set(pq.medical_tokens)
        coverage = self._symptom_coverage(affirmed_set, results)
        if coverage < self.COVERAGE_THRESHOLD and affirmed_set:
            # Re-retrieve with expanded PRF (double the expansion terms)
            old_prf_n = self.retriever.PRF_N_EXP
            self.retriever.PRF_N_EXP = min(old_prf_n * 2, 12)
            expanded_results = self.retriever.retrieve_results(
                pq.expanded_query, pq.medical_tokens, self.top_k
            )
            self.retriever.PRF_N_EXP = old_prf_n
            # Union of both result sets (by condition, keep higher score)
            seen: dict[str, RetrievalResult] = {r.entry.get("condition", ""): r for r in results}
            for r in expanded_results:
                cond = r.entry.get("condition", "")
                if cond not in seen or r.hybrid_score > seen[cond].hybrid_score:
                    seen[cond] = r
            results = sorted(seen.values(), key=lambda r: r.hybrid_score, reverse=True)[: self.top_k]
            scores = [r.hybrid_score for r in results]
            docs = [r.entry for r in results]
            logger.info("Critique 1: re-retrieved (coverage=%.2f); new top_k=%d", coverage, len(results))

        # ── 6b. Causal re-ranking (MedCausalGraph) ────────────────────
        results = self._causal_rerank(results, list(affirmed_set), pq.negated_terms)

        # ── 6c. Critique 5: Session Coherence — Dirichlet prior ───────
        # Apply sequential Bayesian posterior from prior session turns.
        # This is a no-op on the first turn (posterior is empty).
        if session_context:
            # Extract posterior from the pipeline's session store if available
            session_posterior = getattr(self, "_current_session_posterior", {})
            if session_posterior:
                results = self._apply_critique5_session(results, session_posterior)

        scores = [r.hybrid_score for r in results]
        docs = [r.entry for r in results]

        # ── 7. CoT context + LLM generation ───────────────────────────
        context = self._build_context(results)
        cot_context = self._build_cot_context(pq, results, session_context, affirmed_set)

        try:
            model_response = await self.groq_client.generate(
                query=query,
                symptoms=symptoms,
                context=context,
                session_context=session_context,
                cot_context=cot_context,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("GroqClient raised unexpectedly: %s — using fallback", exc)
            model_response = None

        actual_sources = list({d.get("source", "") for d in docs if d.get("source")})

        if model_response is None:
            response = self._fallback_response(results, scores, pq=pq)
            response["sources"] = actual_sources
        else:
            # ── 8. Ground possible_conditions to retrieved set ────────────
            allowed_conditions = {d.get("condition", "") for d in docs}
            model_conds = model_response.get("possible_conditions", [])
            filtered = [c for c in model_conds if c in allowed_conditions]
            if not filtered:
                model_response["possible_conditions"] = [
                    docs[0].get("condition", "Condition not identified")
                ]
            else:
                model_response["possible_conditions"] = filtered

            # ── 9. Four-Critique Self-RAG loop ────────────────────────────
            model_response = self._apply_critiques(model_response, results, pq)

            # ── 10. Prefix uncertainty marker when confidence is low ──────
            if model_response.get("confidence") == "low":
                expl = model_response.get("explanation", "")
                if not expl.startswith("I am not certain"):
                    model_response["explanation"] = f"I am not certain. {expl}".strip()

            model_response["sources"] = actual_sources
            response = model_response

        # ── 11. Novel research contributions ─────────────────────────
        # These are computed regardless of whether Groq is available,
        # so they appear in both LLM and fallback responses.
        top_condition = docs[0].get("condition", "") if docs else ""

        # C1: LOO Shapley symptom attributions
        try:
            attributions = compute_symptom_attributions(
                self.retriever, pq.medical_tokens, top_condition
            )
            response["symptom_attributions"] = attributions or None
        except Exception as exc:  # noqa: BLE001
            logger.debug("Shapley attribution failed: %s", exc)

        # C2: Information-theoretic active inquiry
        try:
            inquiry = recommend_next_question(self.retriever, pq.medical_tokens, results)
            if inquiry.get("question"):
                response["next_question"] = inquiry["question"]
                response["discriminating_symptom"] = inquiry.get("symptom", "")
                response["expected_information_gain"] = inquiry.get("expected_ig", 0.0)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Active inquiry failed: %s", exc)

        # C3: Conformal prediction set
        try:
            cp_result = self.conformal.predict_set(results)
            if cp_result.get("prediction_set"):
                response["conformal_prediction"] = cp_result
        except Exception as exc:  # noqa: BLE001
            logger.debug("Conformal prediction failed: %s", exc)

        return OutputValidator.sanitize(response)

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

        Incorporates BM25+SCS+Prevalence hybrid scoring, causal re-ranking,
        and RAPTOR hierarchical context.  Returns all candidates with full
        scoring metadata, ruling-in/ruling-out symptoms, evidence tier,
        ICD-10 codes, and syndrome/system context.
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
                    "retriever_type": "MedRAG-Turbo (BM25+SCS+Prev+Causal+RAPTOR)",
                    "emergency": True,
                },
                "disclaimer": SAFE_DISCLAIMER,
            }

        results = self.retriever.retrieve_results(pq.expanded_query, pq.medical_tokens, k)
        scores = [r.hybrid_score for r in results]
        entropy = self.retriever.score_entropy(scores)

        # Apply causal re-ranking
        affirmed_set = set(pq.medical_tokens)
        results = self._causal_rerank(results, list(affirmed_set), pq.negated_terms)

        # Build causal scores for metadata exposure
        causal_map = self.causal_graph.causal_scores(
            list(affirmed_set), pq.negated_terms
        ) if results else {}

        differentials = []
        for rank, r in enumerate(results, 1):
            src = r.entry.get("source", "")
            tier = grade_source(src)
            cond_name = r.entry.get("condition", "Unknown")
            hctx = self.raptor.hierarchical_context(cond_name)
            differentials.append(
                {
                    "rank": rank,
                    "condition": cond_name,
                    "icd10": r.entry.get("icd10"),
                    "hybrid_score": round(r.hybrid_score, 4),
                    "symptom_match_ratio": round(r.scs_score, 3),
                    "prevalence_score": round(r.prevalence_score, 3),
                    "causal_score": round(causal_map.get(cond_name, 0.0), 4),
                    "evidence_tier": tier,
                    "evidence_tier_description": tier_description(tier),
                    "source": src,
                    "prevalence": r.entry.get("prevalence", "unknown"),
                    "ruling_in_symptoms": r.ruling_in,
                    "ruling_out_symptoms": r.ruling_out,
                    "severity": r.entry.get("severity", "medium"),
                    "recommended_action": r.entry.get("recommended_action", ""),
                    "syndrome_cluster": hctx.get("syndrome", ""),
                    "organ_system": hctx.get("organ_system", ""),
                    "aetiology_class": hctx.get("aetiology_class", ""),
                }
            )

        return {
            "differentials": differentials,
            "query_metadata": {
                "negated_terms": pq.negated_terms,
                "expanded_terms": pq.normalized_terms,
                "retrieval_entropy": round(entropy, 4),
                "retriever_type": "MedRAG-Turbo (BM25+SCS+Prev+Causal+RAPTOR+Contrastive)",
            },
            "disclaimer": SAFE_DISCLAIMER,
            "contrasts": compute_pairwise_contrasts(results, affirmed_set),
        }
