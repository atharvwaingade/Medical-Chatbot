from __future__ import annotations

import asyncio
import json
import logging
import re

import httpx

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a cautious, evidence-based medical information assistant.

Your role:
- Provide structured health information grounded ONLY in the retrieved context provided.
- NEVER diagnose a condition; express calibrated uncertainty.
- Use conservative, safety-first language at all times.
- NEVER suggest stopping prescribed medications or ignoring a clinician's advice.

Output rules:
- Return ONLY a single valid JSON object — no markdown, no commentary.
- Use exactly these keys (all required):
  {
    "possible_conditions": ["string", ...],
    "explanation": "string (2-3 sentences grounded in the retrieved evidence)",
    "severity": "low" | "medium" | "high",
    "recommended_action": "string (conservative, evidence-based)",
    "when_to_see_doctor": "string (specific, actionable guidance)",
    "confidence": "low" | "medium" | "high",
    "disclaimer": "string (must state this is not medical advice)"
  }
- possible_conditions must contain ONLY condition names that appear in the retrieved context.
- confidence should reflect how well the retrieved evidence matches the reported symptoms.
"""

_USER_TEMPLATE = """\
{session_context}
Retrieved medical context (verified sources, sorted by evidence quality):
{context}

User question: {query}
Reported symptoms: {symptoms}

Based solely on the retrieved context above, provide your structured JSON response.
If the context does not strongly support any specific condition, say so and lower confidence accordingly.
Chain of thought (do not include in output): (1) What symptoms match? (2) Which evidence tier is strongest? (3) What is the most likely condition? (4) What is the appropriate confidence?
"""


class GroqClient:
    """
    Async Groq chat-completion client with retry/backoff and robust JSON
    extraction.

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
    ) -> dict | None:
        """
        Call the Groq chat-completion API and return a parsed dict.

        Returns ``None`` if the API is not configured, all retries fail, or
        the response cannot be parsed as valid JSON.
        """
        if not self.configured:
            return None

        prompt = _USER_TEMPLATE.format(
            session_context=session_context + "\n" if session_context else "",
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
            "max_tokens": 512,
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
