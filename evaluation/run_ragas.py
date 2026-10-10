"""Evaluacion del RAG con Ragas (faithfulness, answer_relevancy, context_precision, context_recall).

Ejemplos:
  python -m evaluation.run_ragas --tag baseline
  python -m evaluation.run_ragas --tag mejora --chunk-size 500 --chunk-overlap 80 --top-k 6
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

from src import config
from src.rag import RAGPipeline

HERE = Path(__file__).parent
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline", help="Nombre de la corrida (baseline, mejora, ...)")
    ap.add_argument("--questions", default=str(HERE / "eval_questions.json"))
    ap.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    ap.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    ap.add_argument("--top-k", type=int, default=config.TOP_K)
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))

    # 1) Ejecutar el pipeline sobre cada pregunta (cada config usa su propia coleccion Chroma)
    pipe = RAGPipeline(args.chunk_size, args.chunk_overlap, args.top_k)
    pipe.ensure_index()

    rows = []
    for q in questions:
        print(f"[{q['id']}/{len(questions)}] {q['question']}")
        r = pipe.answer(q["question"])
        rows.append(
            {
                "user_input": q["question"],
                "retrieved_contexts": [f["texto"] for f in r["fuentes"]],
                "response": r["respuesta"],
                "reference": q["ground_truth"],
            }
        )

    # 2) Evaluar con Ragas usando Gemini como juez
    api_key = config.get_api_key()
    llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model=config.GENERATION_MODEL, google_api_key=api_key, temperature=0)
    )
    emb = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model=f"models/{config.EMBEDDING_MODEL}", google_api_key=api_key)
    )
    answer_relevancy.strictness = 1  # Gemini no soporta n>1 generaciones por llamada

    result = evaluate(
        dataset=EvaluationDataset.from_list(rows),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=emb,
        run_config=RunConfig(max_workers=2, timeout=240),  # pocos hilos para evitar limites de cuota
    )

    df = result.to_pandas()
    df.insert(0, "id", [q["id"] for q in questions])
    df["in_corpus"] = [q["in_corpus"] for q in questions]

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / f"{args.tag}_detalle.csv", index=False)

    present = [m for m in METRICS if m in df.columns]
    summary = {
        "tag": args.tag,
        "config": {
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "top_k": args.top_k,
            "embedding_model": config.EMBEDDING_MODEL,
            "generation_model": config.GENERATION_MODEL,
        },
        "metricas": {m: round(float(df[m].mean()), 4) for m in present},
    }
    (out_dir / f"{args.tag}_resumen.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== RESUMEN ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
