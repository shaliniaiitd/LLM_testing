# utils/llm_client.py
import openai
import os
from typing import Dict, List
import time


class LLMClient:
    def __init__(self, model="gpt-3.5-turbo", temperature=0.7):
        self.model = model
        self.temperature = temperature
        self.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key

    def generate_response(self, prompt: str, system_prompt: str = None) -> Dict:
        """Generate response with error handling and metrics"""
        start_time = time.time()

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=1000
            )

            end_time = time.time()

            return {
                "response": response.choices[0].message.content,
                "tokens_used": response.usage.total_tokens,
                "latency": end_time - start_time,
                "model": self.model,
                "finish_reason": response.choices[0].finish_reason
            }
        except Exception as e:
            return {
                "error": str(e),
                "response": None
            }

    def generate_multiple_responses(self, prompt: str, n: int = 5) -> List[Dict]:
        """Generate multiple responses for consistency testing"""
        return [self.generate_response(prompt) for _ in range(n)]