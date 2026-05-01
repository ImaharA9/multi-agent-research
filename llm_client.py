"""
Generic LLM Client — OpenAI-compatible API wrapper.

Implements the simple chat() protocol used by the agent system:
    chat(system: str, user: str, max_tokens: int) -> str

This module uses the OpenAI SDK but can be pointed at ANY OpenAI-compatible
endpoint by setting LLM_API_BASE.  Replace this file entirely if you prefer
a different provider or SDK.
"""

from openai import OpenAI


class OpenAICompatibleClient:
    """
    Thin wrapper around the OpenAI SDK that exposes a minimal chat() method.

    Works with any provider that supports the /v1/chat/completions endpoint
    (OpenAI, Together, Fireworks, Groq, vLLM, Ollama, etc.).
    """

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def chat(self, system: str, user: str, max_tokens: int = 4096) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""
