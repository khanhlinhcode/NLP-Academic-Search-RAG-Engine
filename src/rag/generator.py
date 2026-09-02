"""
RAG Generator module — Ollama LLM integration.

Sends prompts to a locally-running Ollama instance and returns
LLM-generated answers for the RAG pipeline.
"""

from typing import Generator, Optional

import ollama

from src.config import settings


class RAGGenerator:
    """
    LLM-powered answer generator using Ollama.

    Connects to a local Ollama instance to generate answers
    based on retrieved paper contexts (RAG pipeline).

    Attributes:
        model_name: Name of the Ollama model to use.
        client: Ollama client instance.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the RAG generator with an Ollama model.

        Args:
            model_name: Name of the Ollama model. Defaults to config value.
        """
        self.model_name = model_name or settings.ollama.model_name
        self.client = ollama.Client(host=settings.ollama.base_url)

        print(f"🤖 RAG Generator initialized with model: {self.model_name}")

    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        """
        Generate an answer from the LLM.

        Args:
            prompt: The RAG prompt containing question + paper contexts.
            temperature: Sampling temperature (lower = more focused). Default 0.3
                        for factual, citation-based answers.

        Returns:
            Generated answer text.
        """
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": temperature,
                    "num_predict": 1024,  # Max tokens to generate
                },
            )
            return response["message"]["content"]
        except Exception as e:
            return f"Error generating answer: {str(e)}. Make sure Ollama is running with model '{self.model_name}'."

    def generate_stream(self, prompt: str, temperature: float = 0.3) -> Generator[str, None, None]:
        """
        Stream the LLM response token by token.

        Args:
            prompt: The RAG prompt containing question + paper contexts.
            temperature: Sampling temperature.

        Yields:
            Individual response tokens as they are generated.
        """
        try:
            stream = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": temperature,
                    "num_predict": 1024,
                },
                stream=True,
            )
            for chunk in stream:
                token = chunk["message"]["content"]
                if token:
                    yield token
        except Exception as e:
            yield f"Error: {str(e)}"

    def is_available(self) -> bool:
        """
        Check if the Ollama server is running and the model is available.

        Returns:
            True if the model is accessible, False otherwise.
        """
        try:
            models = self.client.list()
            model_names = [m.model for m in models.models]
            return any(self.model_name in name for name in model_names)
        except Exception:
            return False
