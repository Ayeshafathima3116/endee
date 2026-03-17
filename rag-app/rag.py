"""
rag.py — RAG Query Engine.

Takes a user question, embeds it, retrieves relevant chunks from Endee,
and sends context + question to the Groq LLM to generate a grounded answer.
"""

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from endee import Endee
from groq import Groq

load_dotenv()

ENDEE_HOST  = os.getenv("ENDEE_HOST", "http://localhost:8080")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
INDEX_NAME  = "documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
LLM_MODEL = "llama-3.3-70b-versatile"  # free on Groq

# ─────────────────────────────────────────────
#  Singletons (load once per process)
# ─────────────────────────────────────────────
_embedder = None
_endee_index = None
_groq_client = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _get_endee_index():
    global _endee_index
    if _endee_index is None:
        client = Endee()
        client.set_base_url(f"{ENDEE_HOST}/api/v1")
        _endee_index = client.get_index(name=INDEX_NAME)
    return _endee_index


def _get_groq():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


# ─────────────────────────────────────────────
#  RAG pipeline
# ─────────────────────────────────────────────
def answer_question(question: str) -> dict:
    """
    Full RAG pipeline:
      1. Embed the question
      2. Search Endee for the top-K most similar chunks
      3. Build a prompt with the retrieved context
      4. Call Groq LLM and return the answer + sources
    """
    if not question.strip():
        return {"answer": "Please ask a question.", "sources": []}

    # 1. Embed query
    embedder = _get_embedder()
    query_vector = embedder.encode(question).tolist()

    # 2. Search Endee
    try:
        index = _get_endee_index()
        results = index.query(vector=query_vector, top_k=TOP_K)
    except Exception as e:
        return {
            "answer": f"Error querying Endee: {str(e)}. Make sure Endee is running (docker compose up -d).",
            "sources": []
        }

    if not results:
        return {
            "answer": "I couldn't find any relevant documents. Please ingest some documents first.",
            "sources": []
        }

    # 3. Build context from results
    context_parts = []
    sources = []
    seen_sources = set()

    for r in results:
        meta = r.get("meta", {}) if isinstance(r, dict) else getattr(r, "meta", {})
        text = meta.get("text", "")
        source = meta.get("source", "Unknown")

        if text:
            context_parts.append(f"[Source: {source}]\n{text}")

        if source not in seen_sources:
            seen_sources.add(source)
            sources.append(source)

    context = "\n\n---\n\n".join(context_parts)

    # 4. Call Groq LLM
    prompt = f"""You are a helpful AI assistant. Use ONLY the context below to answer the question.
If the answer is not found in the context, say "I don't have enough information to answer this based on the provided documents."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    try:
        groq = _get_groq()
        response = groq.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Error calling LLM: {str(e)}"

    return {
        "answer": answer,
        "sources": sources,
        "context_chunks": len(results),
    }


# ─────────────────────────────────────────────
#  CLI test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import json
    q = input("Ask a question: ")
    result = answer_question(q)
    print("\n" + "="*60)
    print("ANSWER:", result["answer"])
    print("SOURCES:", result["sources"])
