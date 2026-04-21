from __future__ import annotations

from typing import Literal

SAFE_DISCLAIMER = (
    "This is not medical advice. Always consult a licensed healthcare "
    "provider for diagnosis and treatment."
)

_VALID_SEVERITY: frozenset[str] = frozenset({"low", "medium", "high"})
_VALID_CONFIDENCE: frozenset[str] = frozenset({"low", "medium", "high"})

# Phrases that should never appear in a recommended action
_UNSAFE_TERMS: tuple[str, ...] = (
    "stop all medicines",
    "ignore your doctor",
    "self-surgery",
    "do not take prescribed",
    "stop taking prescribed",
    "diagnose yourself",
    "cure yourself",
)

_SAFE_ACTION_FALLBACK = "Consult a licensed medical professional for personalised guidance."


class OutputValidator:
    @staticmethod
    def sanitize(response: dict) -> dict:
        # Ensure disclaimer is always present and non-empty
        if not response.get("disclaimer"):
            response["disclaimer"] = SAFE_DISCLAIMER
        # Ensure severity is a valid enum value
        if response.get("severity") not in _VALID_SEVERITY:
            response["severity"] = "medium"
        # Ensure confidence is a valid enum value
        if response.get("confidence") not in _VALID_CONFIDENCE:
            response["confidence"] = "low"
        # Sanitize recommended_action
        action = response.get("recommended_action", "")
        if any(term in action.lower() for term in _UNSAFE_TERMS):
            response["recommended_action"] = _SAFE_ACTION_FALLBACK
        # Ensure required string fields exist
        for field in ("explanation", "recommended_action", "when_to_see_doctor"):
            if not response.get(field):
                response[field] = _SAFE_ACTION_FALLBACK
        # Ensure possible_conditions is a non-empty list
        if not response.get("possible_conditions"):
            response["possible_conditions"] = ["Insufficient information to identify a condition"]
        return response
