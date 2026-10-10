from src.rag import RAGPipeline

pipe = RAGPipeline()
preguntas = [
    "¿Qué es un árbol binario de búsqueda (BST)?",
    "¿Qué recorrido de un BST devuelve los elementos en orden ascendente?",
    "¿Qué es un algoritmo de ordenamiento?",
    "¿Cómo funciona el ordenamiento burbuja (bubble sort)?",
]
for q in preguntas:
    print("\nPREGUNTA:", q)
    for f in pipe.retrieve(q):
        print(f"  {f['documento']} chunk{f['chunk_id']} sim={f['similitud']}")
    print("RESPUESTA:", pipe.answer(q)["respuesta"][:200])