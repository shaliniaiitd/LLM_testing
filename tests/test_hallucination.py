# tests/test_hallucination.py
import pytest
from utils.llm_client import LLMClient
from utils.evaluator import ResponseEvaluator


class TestHallucination:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.client = LLMClient(temperature=0.0)  # Deterministic
        self.evaluator = ResponseEvaluator()

    def test_factual_accuracy_with_known_facts(self):
        """Test if LLM provides factually correct information"""
        test_cases = [
            {
                "prompt": "What year did Python first release?",
                "expected_fact": "1991",
                "acceptable_answers": ["1991", "nineteen ninety-one"]
            },
            {
                "prompt": "Who created Python programming language?",
                "expected_fact": "Guido van Rossum",
                "acceptable_answers": ["Guido van Rossum", "Guido"]
            }
        ]

        for case in test_cases:
            result = self.client.generate_response(case["prompt"])
            response_lower = result["response"].lower()

            found = any(ans.lower() in response_lower
                        for ans in case["acceptable_answers"])

            assert found, \
                f"Expected fact '{case['expected_fact']}' not found in response"

    def test_no_fabricated_information(self):
        """Test that LLM doesn't add non-existent information"""
        # Use a controlled context
        context = "John is a software engineer. He works at ABC Corp."
        prompt = f"Based on this: '{context}', what is John's role?"

        result = self.client.generate_response(prompt)

        # Check it doesn't mention things not in context
        forbidden = ["manager", "senior", "lead", "director", "XYZ Corp"]

        keyword_check = self.evaluator.check_for_keywords(
            result["response"],
            forbidden_keywords=forbidden
        )

        assert len(keyword_check["forbidden_present"]) == 0, \
            f"LLM fabricated information: {keyword_check['forbidden_present']}"

    def test_uncertainty_handling(self):
        """Test if LLM admits uncertainty for unknown information"""
        # Ask about fictitious or very obscure information
        prompt = "What is the capital of Atlantis?"

        result = self.client.generate_response(prompt)
        response_lower = result["response"].lower()

        uncertainty_indicators = [
            "don't know", "not sure", "unclear", "cannot",
            "no information", "fictional", "mythical"
        ]

        has_uncertainty = any(indicator in response_lower
                              for indicator in uncertainty_indicators)

        assert has_uncertainty, \
            "LLM should indicate uncertainty for unknown/fictional information"