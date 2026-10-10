"""Cliente de Gemini: embeddings y generacion."""
import time

from google import genai
from google.genai import types

from . import config

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=config.get_api_key())
    return _client


def _with_retry(fn, retries: int = 5):
    for attempt in range(retries):
        try:
            return fn()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)  # backoff por limites de cuota


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT", batch_size: int = 50) -> list[list[float]]:
    """Devuelve un embedding por texto. task_type: RETRIEVAL_DOCUMENT | RETRIEVAL_QUERY."""
    client = get_client()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = _with_retry(
            lambda b=batch: client.models.embed_content(
                model=config.EMBEDDING_MODEL,
                contents=b,
                config=types.EmbedContentConfig(task_type=task_type),
            )
        )
        vectors.extend([e.values for e in resp.embeddings])
        time.sleep(0.5)
    return vectors


def generate(prompt: str, system_instruction: str | None = None, temperature: float = 0.2) -> str:
    client = get_client()
    resp = _with_retry(
        lambda: client.models.generate_content(
            model=config.GENERATION_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
            ),
        )
    )
    return (resp.text or "").strip()
