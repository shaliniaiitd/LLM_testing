# tests/test_performance.py

import pytest
import time
from statistics import mean, stdev


class TestPerformance:

    def test_response_latency(self):
        """Test if response time is within acceptable limits"""
        prompt = "Explain what is CI/CD"
        latencies = []

        for _ in range(10):
            result = self.client.generate_response(prompt)
            latencies.append(result["latency"])

        avg_latency = mean(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]

        assert avg_latency < 3.0, f"Average latency {avg_latency}s exceeds 3s"
        assert p95_latency < 5.0, f"P95 latency {p95_latency}s exceeds 5s"

    def test_token_usage_optimization(self):
        """Test if token usage is optimal"""
        prompt = "Explain pytest in one sentence"

        result = self.client.generate_response(prompt)

        assert result["tokens_used"] < 200, \
            f"Token usage {result['tokens_used']} too high for simple query"

    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        import concurrent.futures

        prompt = "What is Python?"

        def make_request():
            return self.client.generate_response(prompt)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        successful = sum(1 for r in results if r.get("response") is not None)

        assert successful >= 8, \
            f"Only {successful}/10 concurrent requests succeeded"