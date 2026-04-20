from __future__ import annotations

import json

import httpx


class GroqClient:
    def __init__(self, api_key: str | None, model: str):
        self.api_key = api_key
        self.model = model

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str) -> dict | None:
        if not self.configured:
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a cautious medical information assistant. "
                        "Never diagnose. Return only JSON with keys: "
                        "possible_conditions, explanation, severity, recommended_action, "
                        "when_to_see_doctor, confidence, disclaimer."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }

        try:
            with httpx.Client(timeout=20) as client:
                response = client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=body,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return json.loads(content)
        except Exception:
            return None
