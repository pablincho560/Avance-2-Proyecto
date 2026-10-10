# 🎓 Tutor Académico RAG — Avance 2

Asistente experto basado en **RAG** que responde preguntas únicamente con el material de estudio (PDFs) cargado, mantiene el historial de la conversación, muestra las fuentes de cada respuesta y se despliega en la nube.

- **URL pública de la aplicación:** `PENDIENTE: pega aquí tu URL`
- **Repositorio:** `PENDIENTE: pega aquí el enlace de GitHub`

## Flujo RAG implementado

```
Documentos (PDF/DOCX/TXT/HTML)
   │ 1. Ingesta (pypdf, python-docx, BeautifulSoup)
   ▼
Texto por página
   │ 2. Chunking recursivo (800 car., solape 100)
   ▼
Chunks + metadatos (documento, página, chunk_id)
   │ 3. Embeddings (gemini-embedding-001, RETRIEVAL_DOCUMENT)
   ▼
Índice vectorial persistente (ChromaDB, similitud coseno)
   ▲
   │ 5. Recuperación top_k=4 (embedding de la consulta, RETRIEVAL_QUERY)
Pregunta (reescrita como autónoma si es seguimiento)
   │ 6. Generación (Gemini + System Prompt, Few-Shot y delimitadores del Avance 1)
   ▼
Respuesta + fuentes citadas [Fuente N]
```

| Etapa | Decisión | Justificación |
|---|---|---|
| Corpus | Guías de estudio en PDF | Pertinentes al dominio de tutor académico |
| Chunking | 800 caracteres, solape 100, separadores `\n\n`, `\n`, `. ` | Mantiene ideas completas sin diluir la relevancia |
| Embeddings | `gemini-embedding-001` | Multilingüe (español) y misma API que el LLM |
| Índice | ChromaDB persistente | Simple, local, soporta metadatos para citar fuentes |
| Recuperación | Similitud coseno, `top_k=4` | Equilibrio entre cobertura y ruido |
| Generación | Gemini con prompt restringido al contexto | Reduce alucinaciones; responde "no encuentro..." fuera del corpus |
| Seguimiento | Reescritura de la pregunta con el historial | Permite que preguntas como "¿y cómo se aplica?" recuperen bien |

## Estructura

```
app.py                      # Interfaz Streamlit
src/config.py               # Parámetros y manejo de claves
src/llm.py                  # Cliente Gemini (embeddings y generación)
src/ingest.py               # Ingesta + chunking + indexado
src/rag.py                  # Recuperación + generación + historial
src/prompts.py              # System prompt, few-shot, formato de salida
evaluation/eval_questions.json   # 16 preguntas (12 en corpus, 4 fuera de alcance)
evaluation/run_ragas.py          # Evaluación con Ragas
evaluation/compare_results.py    # Comparación antes/después
data/                       # Documentos fuente
docs/                       # PDF de entrega y capturas
```

## Instalación y ejecución local

```bash
git clone <URL_DEL_REPO>
cd tutor-rag
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # y escribe tu GOOGLE_API_KEY
# Coloca tus documentos en data/

python -m src.ingest        # construye el índice (opcional: la app lo crea si no existe)
python -m src.rag           # prueba por consola
streamlit run app.py        # interfaz de chat
```

## Evaluación con Ragas

1. Edita `evaluation/eval_questions.json` con preguntas y respuestas de referencia reales de tu corpus (las 4 últimas son fuera de alcance).
2. Corrida base:
   ```bash
   python -m evaluation.run_ragas --tag baseline
   ```
3. Iteración de mejora (cambia un parámetro, por ejemplo chunk y top_k):
   ```bash
   python -m evaluation.run_ragas --tag mejora --chunk-size 500 --chunk-overlap 80 --top-k 6
   ```
4. Comparación:
   ```bash
   python -m evaluation.compare_results --before baseline --after mejora
   ```

Los resultados quedan en `evaluation/results/` (`*_resumen.json` y el gráfico PNG).

### Resultados
_Pega aquí la tabla de métricas, el análisis de la métrica más baja y la comparación antes/después._

## Despliegue (Streamlit Community Cloud)

1. Sube el repositorio a GitHub (sin `.env` ni `secrets.toml`; ya están en `.gitignore`).
2. En <https://share.streamlit.io> → **New app** → elige el repo, rama `main` y archivo `app.py`.
3. En **Advanced settings → Secrets** pega:
   ```toml
   GOOGLE_API_KEY = "tu_clave"
   ```
4. Despliega. En el primer arranque la app construye el índice a partir de `data/`.
5. Copia la URL pública en la parte superior de este README.

## Seguridad
Las claves se gestionan con variables de entorno o Secrets de la plataforma; nunca se incluyen en el repositorio. Al modelo solo se le envían los fragmentos recuperados necesarios para cada consulta, nunca el corpus completo.
