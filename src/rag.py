"""Pipeline RAG: recuperacion + generacion con historial conversacional."""
from . import config, prompts
from .ingest import build_index, get_collection
from .llm import embed_texts, generate


def default_collection_name(chunk_size: int, chunk_overlap: int) -> str:
    return f"tutor_cs{chunk_size}_co{chunk_overlap}"


class RAGPipeline:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None, top_k: int = None, collection_name: str = None):
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP
        self.top_k = top_k or config.TOP_K
        self.collection_name = collection_name or default_collection_name(self.chunk_size, self.chunk_overlap)
        self.collection = get_collection(self.collection_name)

    def ensure_index(self) -> None:
        """Construye el indice si no existe (util en el primer arranque del despliegue)."""
        if self.collection.count() == 0:
            build_index(self.collection_name, self.chunk_size, self.chunk_overlap)
            self.collection = get_collection(self.collection_name)

    def retrieve(self, query: str) -> list[dict]:
        q_vec = embed_texts([query], task_type="RETRIEVAL_QUERY")[0]
        res = self.collection.query(query_embeddings=[q_vec], n_results=self.top_k)
        out = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append(
                {
                    "texto": doc,
                    "documento": meta["documento"],
                    "pagina": meta["pagina"],
                    "chunk_id": meta["chunk_id"],
                    "similitud": round(1 - dist, 4),  # distancia coseno -> similitud
                }
            )
        return out

    def condense_question(self, question: str, history: list[dict]) -> str:
        """Reescribe preguntas de seguimiento para que la busqueda vectorial funcione."""
        if not history:
            return question
        recent = history[-config.HISTORY_TURNS :]
        prompt = prompts.CONDENSE_PROMPT.format(history=prompts.format_history(recent), question=question)
        return generate(prompt, temperature=0.0) or question

    def answer(self, question: str, history: list[dict] | None = None) -> dict:
        history = history or []
        standalone = self.condense_question(question, history)
        chunks = self.retrieve(standalone)
        user_prompt = prompts.build_user_prompt(question, chunks, history[-config.HISTORY_TURNS :])
        text = generate(user_prompt, system_instruction=prompts.SYSTEM_PROMPT)
        return {"respuesta": text, "fuentes": chunks, "pregunta_autonoma": standalone}


if __name__ == "__main__":
    pipe = RAGPipeline()
    pipe.ensure_index()
    hist: list[dict] = []
    print("Tutor RAG (escribe 'salir' para terminar)")
    while True:
        q = input("\nTú: ").strip()
        if q.lower() in {"salir", "exit"}:
            break
        r = pipe.answer(q, hist)
        print(f"\nTutor: {r['respuesta']}")
        for i, f in enumerate(r["fuentes"], 1):
            print(f"  [Fuente {i}] {f['documento']} p.{f['pagina']} (sim={f['similitud']})")
        hist += [{"role": "user", "content": q}, {"role": "assistant", "content": r["respuesta"]}]
