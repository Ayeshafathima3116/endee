"""
ingest.py — Document ingestion pipeline for RAG Q&A system.

Reads .txt and .pdf files from the docs/ folder, splits them into chunks,
embeds each chunk using sentence-transformers, and stores them in Endee.

Usage:
    python ingest.py
"""

import os
import uuid
import time
from pathlib import Path
from dotenv import load_dotenv

# Sentence-transformers for local, free embeddings
from sentence_transformers import SentenceTransformer

# Endee Python client
from endee import Endee, Precision

# PDF support
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("PyPDF2 not installed. PDF files will be skipped.")

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
load_dotenv()

ENDEE_HOST = os.getenv("ENDEE_HOST", "http://localhost:8080")
INDEX_NAME = "documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # dim=384, fast, free, local
EMBEDDING_DIM = 384
CHUNK_SIZE = 300        # words per chunk
CHUNK_OVERLAP = 50      # word overlap between chunks
DOCS_DIR = Path(__file__).parent / "docs"


# ─────────────────────────────────────────────
#  Embedding model (loads once)
# ─────────────────────────────────────────────
print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
print("Model loaded!")


# ─────────────────────────────────────────────
#  Endee client
# ─────────────────────────────────────────────
def get_endee_client() -> Endee:
    client = Endee()
    client.set_base_url(f"{ENDEE_HOST}/api/v1")
    return client


def ensure_index(client: Endee):
    """Create the Endee index if it doesn't exist yet."""
    try:
        existing = [idx["name"] for idx in client.list_indexes()]
        if INDEX_NAME not in existing:
            print(f"Creating Endee index '{INDEX_NAME}'...")
            client.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIM,
                space_type="cosine",
                precision=Precision.INT8,
            )
            print("Index created!")
        else:
            print(f"Index '{INDEX_NAME}' already exists.")
    except Exception as e:
        print(f"Error creating index: {e}")
        raise


# ─────────────────────────────────────────────
#  Text utilities
# ─────────────────────────────────────────────
def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    if not PDF_SUPPORT:
        return ""
    text = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping word-level chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


# ─────────────────────────────────────────────
#  Main ingestion
# ─────────────────────────────────────────────
def ingest_documents():
    if not DOCS_DIR.exists():
        print(f"Docs directory not found: {DOCS_DIR}")
        return

    files = list(DOCS_DIR.glob("*.txt")) + list(DOCS_DIR.glob("*.pdf"))
    if not files:
        print("No .txt or .pdf files found in docs/")
        return

    client = get_endee_client()
    ensure_index(client)
    index = client.get_index(name=INDEX_NAME)

    total_chunks = 0
    for file_path in files:
        print(f"\nProcessing: {file_path.name}")

        # Read content
        if file_path.suffix == ".pdf":
            content = read_pdf(file_path)
        else:
            content = read_txt(file_path)

        if not content.strip():
            print(f"  Skipping {file_path.name} (empty)")
            continue

        # Chunk
        chunks = chunk_text(content)
        print(f"  Split into {len(chunks)} chunks")

        # Embed in batches of 32
        vectors_to_upsert = []
        for i, chunk in enumerate(chunks):
            embedding = embedder.encode(chunk).tolist()
            vectors_to_upsert.append({
                "id": f"{file_path.stem}_{i}_{uuid.uuid4().hex[:8]}",
                "vector": embedding,
                "meta": {
                    "source": file_path.name,
                    "chunk_index": i,
                    "text": chunk,
                }
            })

        # Upsert to Endee
        index.upsert(vectors_to_upsert)
        total_chunks += len(chunks)
        print(f"  ✓ Indexed {len(chunks)} chunks from '{file_path.name}'")

    print(f"\n✅ Done! Indexed {total_chunks} total chunks into Endee index '{INDEX_NAME}'.")


if __name__ == "__main__":
    ingest_documents()
