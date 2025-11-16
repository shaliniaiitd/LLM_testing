# utils/evaluator.py
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict


class ResponseEvaluator:
    def __init__(self):
        # For semantic similarity
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts"""
        embeddings = self.model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)

    def evaluate_against_golden(self, response: str, golden_output: str,
                                threshold: float = 0.85) -> Dict:
        """Compare response against expected golden output"""
        similarity = self.calculate_semantic_similarity(response, golden_output)

        return {
            "similarity_score": similarity,
            "passed": similarity >= threshold,
            "threshold": threshold
        }

    def check_consistency(self, responses: List[str]) -> Dict:
        """Check consistency across multiple responses"""
        if len(responses) < 2:
            return {"error": "Need at least 2 responses"}

        similarities = []
        for i in range(len(responses) - 1):
            sim = self.calculate_semantic_similarity(responses[i], responses[i + 1])
            similarities.append(sim)

        avg_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)

        return {
            "average_similarity": avg_similarity,
            "std_deviation": std_similarity,
            "consistent": avg_similarity >= 0.8 and std_similarity <= 0.1,
            "all_similarities": similarities
        }

    def check_for_keywords(self, response: str,
                           required_keywords: List[str] = None,
                           forbidden_keywords: List[str] = None) -> Dict:
        """Check for presence/absence of specific keywords"""
        response_lower = response.lower()

        results = {
            "required_present": [],
            "required_missing": [],
            "forbidden_present": [],
            "passed": True
        }

        if required_keywords:
            for keyword in required_keywords:
                if keyword.lower() in response_lower:
                    results["required_present"].append(keyword)
                else:
                    results["required_missing"].append(keyword)
                    results["passed"] = False

        if forbidden_keywords:
            for keyword in forbidden_keywords:
                if keyword.lower() in response_lower:
                    results["forbidden_present"].append(keyword)
                    results["passed"] = False

        return results

    def evaluate_length(self, response: str, min_length: int = 50,
                        max_length: int = 500) -> Dict:
        """Validate response length"""
        length = len(response.split())

        return {
            "word_count": length,
            "min_length": min_length,
            "max_length": max_length,
            "passed": min_length <= length <= max_length
        }