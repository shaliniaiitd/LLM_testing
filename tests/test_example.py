import pytest
from openai import OpenAI

client = OpenAI()


def test_basic_llm_response():
    """Simple LLM test example"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "What is 2+2?"}],
        temperature=0
    )

    answer = response.choices[0].message.content.lower()
    assert "4" in answer or "four" in answer, "LLM should correctly answer 2+2"

# Run: pytest -v test_file.py