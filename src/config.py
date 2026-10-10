"""Configuracion central. Las claves se leen de variables de entorno o de st.secrets."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-3.1-flash-lite")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "4"))
HISTORY_TURNS = 6  # mensajes previos que se envian al LLM


def get_api_key() -> str:
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        try:
            import streamlit as st

            key = st.secrets.get("GOOGLE_API_KEY")
        except Exception:
            key = None
    if not key:
        raise RuntimeError(
            "Falta GOOGLE_API_KEY. Defínela en .env, como variable de entorno "
            "o en los Secrets de la plataforma de despliegue."
        )
    return key
