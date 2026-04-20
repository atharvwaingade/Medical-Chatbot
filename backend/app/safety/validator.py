from __future__ import annotations

SAFE_DISCLAIMER = "This is not medical advice"


class OutputValidator:
    @staticmethod
    def sanitize(response: dict) -> dict:
        if "disclaimer" not in response or not response["disclaimer"]:
            response["disclaimer"] = SAFE_DISCLAIMER

        unsafe_terms = ["stop all medicines", "ignore your doctor", "self-surgery"]
        recommendation = response.get("recommended_action", "")
        if any(term in recommendation.lower() for term in unsafe_terms):
            response["recommended_action"] = "Seek guidance from a licensed medical professional."

        return response
