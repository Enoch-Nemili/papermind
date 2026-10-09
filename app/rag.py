"""
PaperMind — RETRIEVE and GENERATE.

Given a question: retrieve the most similar chunks from pgvector, then have the
configured LLM (Ollama locally / Gemini in the cloud) answer using them.

    python -m app.rag "What is retrieval-augmented generation?"
"""

import os
import sys

from app.config import get_chat_model
from app.embed_store import get_vector_store


def answer_question(question, k=4):
    """Retrieve the top-k relevant chunks, then have the LLM answer using them."""
    store = get_vector_store()

    # --- RETRIEVE ---
    results = store.similarity_search(question, k=k)

    context = "\n\n".join(
        f"[Source: {os.path.basename(doc.metadata.get('source', 'unknown'))}, "
        f"page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in results
    )

    prompt = f"""You are a helpful research assistant answering questions about a library of AI/ML papers.
Answer the question using ONLY the context below. If the answer is not in the context, say you don't know.
Cite the source file(s) you used.

Context:
{context}

Question: {question}

Answer:"""

    # --- GENERATE ---
    llm = get_chat_model()
    response = llm.invoke(prompt)
    return response.content, results


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is retrieval-augmented generation?"
    print(f"Q: {question}\n")
    answer, sources = answer_question(question)
    print("A:", answer)
    print("\n--- Sources used ---")
    for doc in sources:
        name = os.path.basename(doc.metadata.get("source", "unknown"))
        print(f"  - {name} (page {doc.metadata.get('page', '?')})")
