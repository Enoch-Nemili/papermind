"""
PaperMind — RAG Steps 4 & 5: RETRIEVE and GENERATE.

Given a question, this:
  1) embeds the question and finds the most similar chunks in pgvector (RETRIEVE),
  2) hands those chunks + the question to llama3.2 to write a grounded answer (GENERATE).

Try it from the project root (venv active):
    python -m app.rag "What is retrieval-augmented generation?"
"""

import os
import sys

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_postgres import PGVector

from app.embed_store import CONNECTION, COLLECTION


def get_vector_store():
    """Connect to the pgvector collection we filled in the embed step."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION,
        connection=CONNECTION,
        use_jsonb=True,
    )


def answer_question(question, k=4):
    """Retrieve the top-k relevant chunks, then have the LLM answer using them."""
    store = get_vector_store()

    # --- RETRIEVE ---
    # Embed the question and find the k most similar chunks.
    results = store.similarity_search(question, k=k)

    # Stitch the retrieved chunks into one block of context, tagged with their source.
    context = "\n\n".join(
        f"[Source: {os.path.basename(doc.metadata.get('source', 'unknown'))}, "
        f"page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in results
    )

    # --- Build the prompt ---
    prompt = f"""You are a helpful research assistant answering questions about a library of AI/ML papers.
Answer the question using ONLY the context below. If the answer is not in the context, say you don't know.
Cite the source file(s) you used.

Context:
{context}

Question: {question}

Answer:"""

    # --- GENERATE ---
    # Send the prompt to the local llama3.2 model and get its written answer.
    llm = ChatOllama(model="llama3.2")
    response = llm.invoke(prompt)

    return response.content, results


if __name__ == "__main__":
    # Take the question from the command line (or use a default).
    question = " ".join(sys.argv[1:]) or "What is retrieval-augmented generation?"
    print(f"Q: {question}\n")

    answer, sources = answer_question(question)

    print("A:", answer)
    print("\n--- Sources used ---")
    for doc in sources:
        name = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", "?")
        print(f"  - {name} (page {page})")
