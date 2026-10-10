"""Ingesta, chunking, vectorizacion e indexado en ChromaDB.

Uso:  python -m src.ingest --chunk-size 800 --chunk-overlap 100
"""
import argparse
from pathlib import Path

import chromadb
from bs4 import BeautifulSoup
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from . import config
from .llm import embed_texts

SUPPORTED = {".pdf", ".docx", ".txt", ".html", ".htm"}


def load_document(path: Path) -> list[dict]:
    """Extrae texto. Devuelve [{'texto', 'documento', 'pagina'}] (pagina=1 si no aplica)."""
    ext = path.suffix.lower()
    pages: list[tuple[int, str]] = []
    if ext == ".pdf":
        reader = PdfReader(str(path))
        pages = [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    elif ext == ".docx":
        text = "\n".join(p.text for p in Document(str(path)).paragraphs)
        pages = [(1, text)]
    elif ext in (".html", ".htm"):
        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
        pages = [(1, soup.get_text(separator="\n"))]
    elif ext == ".txt":
        pages = [(1, path.read_text(encoding="utf-8", errors="ignore"))]
    return [{"texto": t.strip(), "documento": path.name, "pagina": n} for n, t in pages if t.strip()]


def chunk_pages(pages: list[dict], chunk_size: int, chunk_overlap: int) -> list[dict]:
    """Chunking recursivo (parrafo > linea > oracion > palabra), por pagina para conservar metadatos."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    counters: dict[str, int] = {}
    for p in pages:
        for piece in splitter.split_text(p["texto"]):
            n = counters.get(p["documento"], 0)
            counters[p["documento"]] = n + 1
            chunks.append(
                {
                    "id": f"{p['documento']}-p{p['pagina']}-c{n}",
                    "texto": piece,
                    "documento": p["documento"],
                    "pagina": p["pagina"],
                    "chunk_id": n,
                }
            )
    return chunks


def get_collection(name: str):
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def build_index(collection_name: str, chunk_size: int, chunk_overlap: int, reset: bool = True) -> int:
    files = sorted(f for f in config.DATA_DIR.iterdir() if f.suffix.lower() in SUPPORTED)
    if not files:
        raise FileNotFoundError(f"No hay documentos en {config.DATA_DIR}. Agrega tus PDFs/TXT/DOCX/HTML.")

    pages = [pg for f in files for pg in load_document(f)]
    chunks = chunk_pages(pages, chunk_size, chunk_overlap)
    print(f"{len(files)} documentos -> {len(pages)} paginas -> {len(chunks)} chunks")

    if reset:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    col = get_collection(collection_name)

    vectors = embed_texts([c["texto"] for c in chunks], task_type="RETRIEVAL_DOCUMENT")
    col.add(
        ids=[c["id"] for c in chunks],
        documents=[c["texto"] for c in chunks],
        embeddings=vectors,
        metadatas=[
            {"documento": c["documento"], "pagina": c["pagina"], "chunk_id": c["chunk_id"]} for c in chunks
        ],
    )
    print(f"Indice '{collection_name}' creado con {col.count()} chunks en {config.CHROMA_DIR}")
    return col.count()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    ap.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    ap.add_argument("--collection", default=None)
    args = ap.parse_args()
    from .rag import default_collection_name

    name = args.collection or default_collection_name(args.chunk_size, args.chunk_overlap)
    build_index(name, args.chunk_size, args.chunk_overlap)
