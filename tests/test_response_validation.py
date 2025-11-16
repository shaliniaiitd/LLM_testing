# tests/test_response_validation.py
import pytest
from utils.llm_client import LLMClient
from utils.evaluator import ResponseEvaluator


class TestResponseValidation:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.client = LLMClient(temperature=0.3)  # Low temp for consistency
        self.evaluator = ResponseEvaluator()

    @pytest.mark.parametrize("prompt,golden_output,threshold", [
        (
                "What is Python?",
                "Python is a high-level programming language known for readability",
                0.85
        ),
        (
                "Explain machine learning",
                "Machine learning is a subset of AI that enables systems to learn from data",
                0.85
        )
    ])
    def test_golden_dataset_validation(self, prompt, golden_output, threshold):
        """Test LLM responses against golden dataset"""
        result = self.client.generate_response(prompt)

        assert result["response"] is not None, "LLM should return a response"

        evaluation = self.evaluator.evaluate_against_golden(
            result["response"],
            golden_output,
            threshold
        )

        assert evaluation["passed"], \
            f"Similarity {evaluation['similarity_score']} below threshold {threshold}"

    def test_consistency_across_runs(self):
        """Test if same prompt gives consistent responses"""
        prompt = "Explain what is API testing in 2 sentences"

        responses = self.client.generate_multiple_responses(prompt, n=5)
        response_texts = [r["response"] for r in responses if r["response"]]

        consistency_result = self.evaluator.check_consistency(response_texts)

        assert consistency_result["consistent"], \
            f"Responses not consistent. Avg similarity: {consistency_result['average_similarity']}"

    def test_response_completeness(self):
        """Test if response contains required information"""
        prompt = "List 3 benefits of test automation"
        required_keywords = ["time", "efficiency", "coverage"]

        result = self.client.generate_response(prompt)

        keyword_check = self.evaluator.check_for_keywords(
            result["response"],
            required_keywords=required_keywords
        )

        assert len(keyword_check["required_present"]) >= 2, \
            f"Response missing key concepts: {keyword_check['required_missing']}"

    def test_response_length_validation(self):
        """Test if response length is appropriate"""
        prompt = "Explain pytest in one paragraph"

        result = self.client.generate_response(prompt)

        length_check = self.evaluator.evaluate_length(
            result["response"],
            min_length=30,
            max_length=150
        )

        assert length_check["passed"], \
            f"Response length {length_check['word_count']} outside acceptable range"