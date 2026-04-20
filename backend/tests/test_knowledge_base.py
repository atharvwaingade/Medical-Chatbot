"""
Validates that every entry in the medical knowledge base satisfies the
required schema.  This runs as part of CI so data quality issues are caught
before they reach production.
"""
import json
import os
import unittest
from pathlib import Path

DATASET_PATH = os.environ.get(
    "MEDICAL_DATASET_PATH", "backend/data/sample_medical_knowledge.json"
)

REQUIRED_FIELDS = {
    "source",
    "verified",
    "condition",
    "symptoms",
    "explanation",
    "recommended_action",
    "when_to_see_doctor",
    "severity",
    "warnings",
}

VALID_SEVERITIES = {"low", "medium", "high"}


class KnowledgeBaseSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(DATASET_PATH)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        with path.open(encoding="utf-8") as f:
            cls.entries = json.load(f)

    def test_at_least_twenty_entries(self):
        self.assertGreaterEqual(len(self.entries), 20, "Knowledge base too small")

    def test_all_entries_verified(self):
        for entry in self.entries:
            self.assertTrue(
                entry.get("verified"),
                msg=f"Entry '{entry.get('condition')}' is not marked verified",
            )

    def test_required_fields_present(self):
        for entry in self.entries:
            missing = REQUIRED_FIELDS - set(entry.keys())
            self.assertFalse(
                missing,
                msg=f"Entry '{entry.get('condition')}' missing fields: {missing}",
            )

    def test_severity_valid_values(self):
        for entry in self.entries:
            self.assertIn(
                entry.get("severity"),
                VALID_SEVERITIES,
                msg=f"Entry '{entry.get('condition')}' has invalid severity",
            )

    def test_symptoms_is_nonempty_list(self):
        for entry in self.entries:
            symptoms = entry.get("symptoms", [])
            self.assertIsInstance(symptoms, list, msg=entry.get("condition"))
            self.assertGreater(
                len(symptoms),
                0,
                msg=f"Entry '{entry.get('condition')}' has no symptoms",
            )

    def test_warnings_is_list(self):
        for entry in self.entries:
            self.assertIsInstance(
                entry.get("warnings", []),
                list,
                msg=entry.get("condition"),
            )

    def test_condition_names_unique(self):
        names = [e.get("condition") for e in self.entries]
        duplicates = {n for n in names if names.count(n) > 1}
        self.assertFalse(duplicates, f"Duplicate condition names: {duplicates}")

    def test_string_fields_nonempty(self):
        str_fields = ("condition", "explanation", "recommended_action", "when_to_see_doctor")
        for entry in self.entries:
            for field in str_fields:
                self.assertTrue(
                    entry.get(field, "").strip(),
                    msg=f"Entry '{entry.get('condition')}' has empty '{field}'",
                )


if __name__ == "__main__":
    unittest.main()
