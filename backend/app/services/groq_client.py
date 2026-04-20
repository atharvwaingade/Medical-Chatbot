from __future__ import annotations

import asyncio
import json
import logging
import re

import httpx

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a cautious, evidence-based medical information assistant trained in
clinical reasoning using illness script theory (Schmidt & Rikers, 2007).

Your role:
- Provide structured health information grounded ONLY in the retrieved context provided.
- NEVER diagnose a condition; express calibrated uncertainty at all times.
- Use conservative, safety-first language.
- NEVER suggest stopping prescribed medications or ignoring a clinician's advice.

Clinical reasoning framework (illness script):
You will reason through eight structured steps provided in the context before
generating your final JSON response.  Your output must reflect this reasoning.

Output rules:
- Return ONLY a single valid JSON object — no markdown, no commentary.
- Use exactly these keys (all required):
  {
    "possible_conditions": ["string", ...],
    "explanation": "string (2-3 sentences grounded in the retrieved evidence \
and the clinical reasoning chain)",
    "severity": "low" | "medium" | "high",
    "recommended_action": "string (conservative, evidence-based)",
    "when_to_see_doctor": "string (specific, actionable guidance)",
    "confidence": "low" | "medium" | "high",
    "disclaimer": "string (must state this is not medical advice)"
  }
- possible_conditions must contain ONLY condition names that appear in the retrieved context.
- confidence must reflect the Bayesian posterior ranking provided; if the top
  condition's Bayesian score is < 0.20, output "low".
- Do NOT mention the reasoning steps in the output — they are internal scaffolding.
"""

_USER_TEMPLATE = """\
{session_context}\
=== ILLNESS SCRIPT CLINICAL REASONING CHAIN ===

Step 1 — EPIDEMIOLOGY
  Patient context: {patient_context}

Step 2 — SYMPTOM INVENTORY (affirmed symptoms — present)
  {affirmed_symptoms}

Step 3 — NEGATION FILTER (denied symptoms — absent; do NOT reason these in)
  {negated_symptoms}

Step 4 — TIME COURSE SIGNALS
  {time_course}

Step 5 — CANDIDATE CONDITIONS (from retrieval, sorted by Bayesian posterior)
  {bayesian_ranking}

Step 6 — RULING-IN / RULING-OUT ANALYSIS (from retrieved evidence)
{ruling_in_out}

Step 7 — EVIDENCE QUALITY
{evidence_quality}

Step 8 — RETRIEVED CONTEXT (full, verified sources)
{context}

=== PATIENT QUERY ===
{query}
Reported symptoms: {symptoms}

Using the illness script reasoning above and the retrieved context, provide
your structured JSON response.  If the evidence does not strongly support any
specific condition, lower confidence accordingly.
"""


def _detect_time_course(query: str, symptoms: list[str]) -> str:
    """Extract time-course signals from the query text."""
    text = (query + " " + " ".join(symptoms)).lower()
    signals: list[str] = []
    if any(w in text for w in ("sudden", "abrupt", "acute", "instantly", "immediately")):
        signals.append("Sudden onset")
    if any(w in text for w in ("chronic", "months", "years", "long-standing", "persistent")):
        signals.append("Chronic / long-standing")
    for unit in ("day", "days", "week", "weeks", "hour", "hours"):
        m = re.search(r"(\d+)\s*" + unit, text)
        if m:
            signals.append(f"Duration ~{m.group(1)} {unit}")
            break
    if any(w in text for w in ("worsening", "worse", "escalating", "progressing")):
        signals.append("Worsening trajectory")
    if any(w in text for w in ("better", "improving", "resolving")):
        signals.append("Improving trajectory")
    return "; ".join(signals) if signals else "Not specified"


class GroqClient:
    """
    Async Groq chat-completion client with retry/backoff, robust JSON
    extraction, and MedCoT-DDx illness-script prompt architecture.

    The prompt follows Schmidt & Rikers (2007) illness script theory:
    Step 1 Epidemiology → Step 2 Symptoms → Step 3 Negations →
    Step 4 Time Course → Step 5 Bayesian Candidates → Step 6 Ruling In/Out →
    Step 7 Evidence Quality → Step 8 Full Retrieved Context.

    Retries are attempted only on HTTP 429 (rate-limit) and 5xx errors.
    On all other errors the method returns ``None`` so the pipeline can fall
    back gracefully.
    """

    MAX_RETRIES: int = 2
    BASE_DELAY: float = 1.0  # seconds; doubles on each retry

    def __init__(self, api_key: str | None, model: str) -> None:
        self.api_key = api_key
        self.model = model

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def generate(
        self,
        query: str,
        symptoms: list[str],
        context: str,
        session_context: str = "",
        cot_context: dict | None = None,
    ) -> dict | None:
        """
        Call the Groq chat-completion API and return a parsed dict.

        Parameters
        ----------
        query : str
            The patient's free-text query.
        symptoms : list[str]
            Reported symptom list.
        context : str
            JSON-serialised retrieved evidence (from ``_build_context``).
        session_context : str
            Prior-turn summary from the session store.
        cot_context : dict | None
            Structured chain-of-thought metadata from the pipeline:
            ``affirmed_terms``, ``negated_terms``, ``bayesian_ranking``,
            ``ruling_in_out``, ``evidence_quality``, ``patient_context``.

        Returns
        -------
        dict | None
            Parsed JSON response, or ``None`` on failure / unconfigured.
        """
        if not self.configured:
            return None

        cot = cot_context or {}
        affirmed = cot.get("affirmed_terms") or symptoms or ["not specified"]
        negated = cot.get("negated_terms") or []
        bayesian = cot.get("bayesian_ranking") or []
        ruling = cot.get("ruling_in_out") or ""
        evidence = cot.get("evidence_quality") or "  Source tiers available in retrieved context."
        patient_ctx = cot.get("patient_context") or "Not specified"

        prompt = _USER_TEMPLATE.format(
            session_context=session_context + "\n" if session_context else "",
            patient_context=patient_ctx,
            affirmed_symptoms="  " + ", ".join(affirmed) if affirmed else "  None reported",
            negated_symptoms=(
                "  " + ", ".join(negated) if negated else "  None (no negations detected)"
            ),
            time_course="  " + _detect_time_course(query, symptoms),
            bayesian_ranking=(
                "\n".join(
                    f"  {rank+1}. {cond} (posterior={score:.3f})"
                    for rank, (cond, score) in enumerate(bayesian)
                )
                if bayesian
                else "  Not available"
            ),
            ruling_in_out=ruling if ruling else "  Not available",
            evidence_quality=evidence,
            context=context,
            query=query,
            symptoms=", ".join(symptoms) if symptoms else "none reported",
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "temperature": 0.1,
            "max_tokens": 600,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=25) as client:
                    response = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers=headers,
                        json=body,
                    )
                if response.status_code in (429, 500, 502, 503, 504):
                    if attempt < self.MAX_RETRIES:
                        delay = self.BASE_DELAY * (2**attempt)
                        logger.warning(
                            "Groq API HTTP %d — retrying in %.1fs (attempt %d/%d)",
                            response.status_code,
                            delay,
                            attempt + 1,
                            self.MAX_RETRIES,
                        )
                        await asyncio.sleep(delay)
                        continue
                    logger.error("Groq API HTTP %d — all retries exhausted", response.status_code)
                    return None

                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return self._parse_json(content)

            except httpx.TimeoutException:
                if attempt < self.MAX_RETRIES:
                    await asyncio.sleep(self.BASE_DELAY * (2**attempt))
                    continue
                logger.warning("Groq API timed out after %d attempts", self.MAX_RETRIES + 1)
                return None
            except Exception as exc:  # noqa: BLE001
                logger.warning("Groq API error: %s", exc)
                return None

        return None

    # ------------------------------------------------------------------
    # JSON extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """
        Try three strategies to extract a JSON object from the model's
        response text, in order of preference:
          1. Direct JSON parse.
          2. Markdown code-fence extraction (```json ... ```).
          3. Greedy search for the outermost ``{...}`` block.
        """
        # 1. Direct parse
        stripped = text.strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        # 2. Code-fence extraction
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1))
            except json.JSONDecodeError:
                pass

        # 3. Outermost brace block
        brace_match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning("Could not extract JSON from Groq response: %.120s", stripped)
        return None
