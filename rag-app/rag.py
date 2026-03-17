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
def stream_answer(question: str):
    """
    Streaming version of the RAG pipeline.
    Yields dicts for metadata and individual text chunks.
    """
    if not question.strip():
        yield {"answer": "Please ask a question."}
        return

    # 1. Embed query
    embedder = _get_embedder()
    query_vector = embedder.encode(question).tolist()

    # 2. Search Endee
    try:
        index = _get_endee_index()
        results = index.query(vector=query_vector, top_k=TOP_K)
    except Exception as e:
        yield {"answer": f"Error querying Endee: {str(e)}"}
        return

    if not results:
        yield {"answer": "I couldn't find any relevant documents."}
        return

    # 3. Build context and sources
    context_parts = []
    unique_sources = {} # {name: first_preview}
    
    for r in results:
        meta = r.get("meta", {}) if isinstance(r, dict) else getattr(r, "meta", {})
        text = meta.get("text", "")
        source = meta.get("source", "Unknown")

        if text:
            context_parts.append(f"[Source: {source}]\n{text}")
            if source not in unique_sources:
                unique_sources[source] = text[:500]

    sources_with_text = [{"name": name, "preview": preview} for name, preview in unique_sources.items()]
    context = "\n\n---\n\n".join(context_parts)

    # Yield metadata first (sources)
    yield {"sources": sources_with_text, "context_chunks": len(results)}

    # 4. Call Groq with streaming=True
    prompt = f"""You are DocuMind, a professional AI knowledge assistant. 
Use the provided CONTEXT to answer the user's QUESTION accurately and concisely.
If the answer isn't in the context, state that you don't have that information.

Structure your response as follows:
1. Your detailed answer.
2. The exact marker: ---KNOWLEDGE_EXTRAS---
3. FOLLOW_UPS: ["Smart Follow-up 1?", "Smart Follow-up 2?", "Smart Follow-up 3?"]
4. The exact marker: ---END_EXTRAS---

CRITICAL: 
- Provide exactly 3 diverse, curiosity-driven follow-up questions.
- Use ONLY double quotes for the JSON parts.
- Do NOT output anything after ---END_EXTRAS---.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    try:
        groq = _get_groq()
        stream = groq.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024,
            stream=True
        )
        accumulated_answer = ""
        extras_marker = "---KNOWLEDGE_EXTRAS---"
        for chunk in stream:
            content = chunk.choices[0].delta.content or ""
            if content:
                accumulated_answer += content
                if extras_marker in accumulated_answer:
                    # Send only the part before the marker
                    clean_content = content.split(extras_marker)[0]
                    if clean_content:
                        yield {"answer_chunk": clean_content}
                    break 
                yield {"answer_chunk": content}

        # Final extras parsing
        if extras_marker in accumulated_answer:
            yield {"answer_cleanup": True}
            extras_part = accumulated_answer.split(extras_marker)[-1].split("---END_EXTRAS---")[0]
            
            # More robust parsing for FOLLOW_UPS
            try:
                import re
                follow_ups = []
                
                f_match = re.search(r'FOLLOW_UPS:\s*(\[.*?\])', extras_part, re.DOTALL)
                if f_match:
                    try:
                        follow_ups = json.loads(f_match.group(1))
                    except:
                        # Fallback for single quotes
                        follow_ups = json.loads(f_match.group(1).replace("'", '"'))
                
                if follow_ups:
                    yield {"suggestions": follow_ups}
            except Exception as e:
                print(f"Extras parsing error: {e}")
                pass
    except Exception as e:
        yield {"error": str(e)}

def answer_question(question: str) -> dict:
    """
    Synchronous wrapper for stream_answer.
    Maintains compatibility with non-streaming consumers.
    """
    full_answer = ""
    sources = []
    chunks = 0
    for chunk in stream_answer(question):
        if "sources" in chunk:
            sources = [s["name"] for s in chunk["sources"]]
            chunks = chunk["context_chunks"]
        if "answer_chunk" in chunk:
            full_answer += chunk["answer_chunk"]
        if "answer" in chunk: # For simple string returns
            full_answer = chunk["answer"]
    
    return {
        "answer": full_answer,
        "sources": sources,
        "context_chunks": chunks
    }


# ─────────────────────────────────────────────
#  CLI test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import json
    q = input("Ask a question: ")
    print("\nSTREAMING RESPONSE:")
    print("-" * 20)
    for chunk in stream_answer(q):
        if "answer_chunk" in chunk:
            print(chunk["answer_chunk"], end="", flush=True)
        elif "sources" in chunk:
            print(f"\n[Sources: {', '.join([s['name'] for s in chunk['sources']])}]")
    print("\n" + "-" * 20)
