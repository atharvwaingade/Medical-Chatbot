import unittest

from app.safety.rules import detect_emergency, detect_red_flags


class SafetyTests(unittest.TestCase):
    def test_detect_emergency_keywords(self):
        self.assertTrue(detect_emergency("severe chest pain and breathing difficulty"))
        self.assertFalse(detect_emergency("mild seasonal allergy"))

    def test_detect_red_flags_combo(self):
        symptoms = ["Fever", "stiff neck", "severe headache"]
        self.assertTrue(detect_red_flags(symptoms))


if __name__ == "__main__":
    unittest.main()
