"""Interfaz de chat (Streamlit) con historial, preguntas de seguimiento y fuentes."""
import streamlit as st

from src import config
from src.rag import RAGPipeline

st.set_page_config(page_title="Tutor Académico RAG", page_icon="🎓")
st.title("🎓 Tutor Académico RAG")
st.caption("Responde solo con base en el material de estudio cargado.")


@st.cache_resource(show_spinner="Preparando la base de conocimiento...")
def load_pipeline() -> RAGPipeline:
    pipe = RAGPipeline()
    pipe.ensure_index()
    return pipe


try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"No se pudo iniciar el asistente: {e}")
    st.stop()

with st.sidebar:
    st.header("Configuración")
    st.write(f"Modelo: `{config.GENERATION_MODEL}`")
    st.write(f"Embeddings: `{config.EMBEDDING_MODEL}`")
    st.write(f"Chunk: {pipeline.chunk_size} / solape {pipeline.chunk_overlap}")
    st.write(f"top_k: {pipeline.top_k}")
    if st.button("🗑️ Nueva conversación"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(fuentes: list[dict]) -> None:
    with st.expander(f"📚 Fuentes ({len(fuentes)})"):
        for i, f in enumerate(fuentes, 1):
            st.markdown(f"**[Fuente {i}] {f['documento']} — p.{f['pagina']}** (similitud {f['similitud']})")
            st.caption(f["texto"])


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("fuentes"):
            render_sources(m["fuentes"])

if prompt := st.chat_input("Escribe tu pregunta..."):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en el material..."):
            result = pipeline.answer(prompt, history)
        st.markdown(result["respuesta"])
        render_sources(result["fuentes"])

    st.session_state.messages.append(
        {"role": "assistant", "content": result["respuesta"], "fuentes": result["fuentes"]}
    )
