# utils/metrics_calculator.py
class MetricsCalculator:

    def calculate_test_metrics(self, test_results: List[Dict]) -> Dict:
        """Calculate comprehensive test metrics"""
        total_tests = len(test_results)
        passed = sum(1 for r in test_results if r.get("passed", False))

        latencies = [r["latency"] for r in test_results if "latency" in r]
        tokens = [r["tokens_used"] for r in test_results if "tokens_used" in r]

        return {
            "total_tests": total_tests,
            "passed": passed,
            "failed": total_tests - passed,
            "pass_rate": (passed / total_tests * 100) if total_tests > 0 else 0,
            "avg_latency": mean(latencies) if latencies else 0,
            "p95_latency": sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0,
            "avg_tokens": mean(tokens) if tokens else 0,
            "total_cost": sum(tokens) * 0.000002  # Example pricing
        }