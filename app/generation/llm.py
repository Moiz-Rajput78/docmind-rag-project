"""
DocMind RAG - Ollama Cloud LLM Service

Connects DocMind to Ollama Cloud using an API key.

The API key is loaded from .env and is never hard-coded.
"""

import os
from typing import Optional

import requests
from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_API_URL = "https://ollama.com/api/chat"

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "gpt-oss:120b-cloud",
)


# ============================================================
# VALIDATION
# ============================================================

def validate_configuration() -> None:
    """
    Validate that the Ollama Cloud API key is configured.
    """

    if not OLLAMA_API_KEY:
        raise RuntimeError(
            "OLLAMA_API_KEY is not configured. "
            "Add it to the project's .env file."
        )


# ============================================================
# OLLAMA CLOUD CLIENT
# ============================================================

class OllamaCloud:
    """
    Simple client for the Ollama Cloud chat API.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        timeout: int = 120,
    ) -> None:

        validate_configuration()

        self.model = model or OLLAMA_MODEL
        self.timeout = timeout

        self.headers = {
            "Authorization": f"Bearer {OLLAMA_API_KEY}",
            "Content-Type": "application/json",
        }

    # --------------------------------------------------------
    # CHAT
    # --------------------------------------------------------

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Send a chat request to Ollama Cloud.

        Returns:
            Generated assistant response.
        """

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "stream": False,
        }

        try:

            response = requests.post(
                OLLAMA_API_URL,
                headers=self.headers,
                json=payload,
                timeout=self.timeout,
            )

        except requests.RequestException as exc:

            raise RuntimeError(
                f"Unable to connect to Ollama Cloud: {exc}"
            ) from exc

        # ----------------------------------------------------
        # HTTP ERROR
        # ----------------------------------------------------

        if not response.ok:

            try:
                error_data = response.json()
            except ValueError:
                error_data = response.text

            raise RuntimeError(
                "Ollama Cloud API request failed "
                f"(HTTP {response.status_code}): "
                f"{error_data}"
            )

        # ----------------------------------------------------
        # PARSE RESPONSE
        # ----------------------------------------------------

        try:
            data = response.json()
        except ValueError as exc:

            raise RuntimeError(
                "Ollama Cloud returned an invalid JSON response."
            ) from exc

        message = data.get("message", {})

        content = message.get("content")

        if not content:
            raise RuntimeError(
                "Ollama Cloud returned an empty response."
            )

        return content.strip()


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def generate(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
) -> str:
    """
    Convenience wrapper around OllamaCloud.
    """

    client = OllamaCloud(
        model=model,
    )

    return client.chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DOCMIND - OLLAMA CLOUD CONNECTION TEST")
    print("=" * 60)

    print()
    print(f"Model: {OLLAMA_MODEL}")
    print(f"Endpoint: {OLLAMA_API_URL}")

    print()
    print("Testing connection...")

    try:

        answer = generate(
            system_prompt=(
                "You are a simple connection test. "
                "Do not provide a long explanation."
            ),
            user_prompt=(
                "Reply with exactly: "
                "Ollama Cloud connection successful."
            ),
        )

        print()
        print("Response:")
        print(answer)

        print()
        print("=" * 60)
        print("OLLAMA CLOUD TEST PASSED")
        print("=" * 60)

    except Exception as exc:

        print()
        print("=" * 60)
        print("OLLAMA CLOUD TEST FAILED")
        print("=" * 60)

        print()
        print(f"Error: {exc}")

        raise