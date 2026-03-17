# 🧠 DocuMind — AI-Powered Document Q&A with Endee Vector Database

> **RAG (Retrieval-Augmented Generation)** chatbot that lets you upload documents and ask questions about them in natural language. Powered by **[Endee](https://github.com/endee-io/endee)** as the vector database, `sentence-transformers` for embeddings, and **Groq LLM** for answer generation.

---

## 📋 Project Overview

DocuMind is a complete RAG (Retrieval-Augmented Generation) pipeline that:

1. **Ingests** `.txt` and `.pdf` documents — splits them into overlapping chunks
2. **Embeds** each chunk using `all-MiniLM-L6-v2` (local, free, 384 dimensions)
3. **Stores** the vector embeddings in **Endee** vector database
4. **Retrieves** the most semantically relevant chunks at query time using cosine similarity search
5. **Generates** grounded, cited answers using the **Groq LLM API** (free tier)

The result is a document Q&A chatbot that gives accurate answers with source citations — no hallucination, no made-up facts.

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────┐
│                        User Browser                        │
│               (Beautiful Dark-Mode Chat UI)                │
└────────────────────┬───────────────────────────────────────┘
                     │  HTTP
┌────────────────────▼───────────────────────────────────────┐
│                   Flask Web Server (app.py)                 │
│   GET /     →  Chat UI                                      │
│   POST /ask →  RAG pipeline                                 │
│   POST /ingest → Trigger document indexing                  │
│   GET /status  → Health check                               │
└────────────────────┬───────────────────────────────────────┘
                     │
         ┌───────────┴───────────────────────────┐
         │                                       │
┌────────▼──────────────┐           ┌────────────▼──────────┐
│  sentence-transformers │           │   Groq LLM API        │
│  all-MiniLM-L6-v2     │           │   llama-3.3-70b       │
│  (local, free, 384-d)  │           │   (free tier)         │
└────────┬──────────────┘           └────────────┬──────────┘
         │ embed documents                        │ generate answer
         │ embed user query                       │ from retrieved context
         │                                        │
┌────────▼──────────────────────────────────────┐│
│           Endee Vector Database               ││
│   ────────────────────────────────────────   ││
│   • Index: "documents" (cosine, dim=384)      ││
│   • Upsert: chunked document vectors          ││
│   • Query: top-K semantic search              ◄┘│
│   • Payload filtering: source, chunk_index    │
│   • Running via Docker on port 8080           │
└───────────────────────────────────────────────┘
```

---

## 🚀 Why Endee?

Endee is used as the core retrieval layer of this RAG pipeline because:

| Feature | Benefit |
|---|---|
| **High-performance C++ core** | Millisecond query latency even with large indexes |
| **Cosine similarity search** | Ideal for semantic text embedding search |
| **Payload filtering** | Filter by document source, date, or any metadata at search time |
| **Docker deployment** | One command to start: `docker compose up -d` |
| **Python SDK** | Clean client API: `create_index()`, `upsert()`, `query()` |
| **INT8 precision** | Memory-efficient storage without sacrificing accuracy |
| **Up to 1B vectors/node** | Production-ready scale |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Vector Database | **Endee** (open-source, Docker) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (local, free) |
| LLM | **Groq API** — `llama-3.3-70b-versatile` (free tier) |
| Backend | **Python + Flask** |
| Frontend | HTML / CSS / Vanilla JavaScript (dark mode chat UI) |
| Document Parsing | `PyPDF2` for PDF, built-in for `.txt` |

---

## 📁 Project Structure

```
rag-app/
├── docker-compose.yml    # Runs Endee vector database
├── app.py                # Flask web application
├── ingest.py             # Document chunking, embedding & indexing into Endee
├── rag.py                # RAG query engine (embed → Endee search → LLM)
├── templates/
│   └── index.html        # Chat UI
├── docs/                 # Put your .txt or .pdf files here
│   ├── artificial_intelligence.txt
│   └── vector_databases.txt
├── requirements.txt
└── .env.example          # Environment variables template
```

---

## ⚙️ Setup Instructions

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- Python 3.9+ installed
- A free [Groq API key](https://console.groq.com) (takes 1 minute)

---

### Step 1 — Clone & Navigate

```bash
git clone https://github.com/YOUR_USERNAME/endee.git
cd endee/rag-app
```

---

### Step 2 — Start Endee Vector Database

```bash
docker compose up -d
```

Verify Endee is running:
```bash
curl http://localhost:8080/api/v1/indexes
# Expected: []
```

Or open [http://localhost:8080](http://localhost:8080) in your browser to see the Endee dashboard.

---

### Step 3 — Configure Environment

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your Groq API key
# GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```

Get your free Groq API key at: https://console.groq.com

---

### Step 4 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ The first run will download the `all-MiniLM-L6-v2` model (~90MB). Subsequent runs use the cached model.

---

### Step 5 — Add Your Documents

Place your `.txt` or `.pdf` files in the `docs/` folder. Two sample documents are included:
- `docs/artificial_intelligence.txt` — Overview of AI, ML, Deep Learning, RAG
- `docs/vector_databases.txt` — Deep dive into vector DBs, Endee, semantic search

---

### Step 6 — Index Documents into Endee

```bash
python ingest.py
```

This will:
- Read all files from `docs/`
- Split them into overlapping chunks (300 words, 50-word overlap)
- Embed each chunk locally
- Upsert all vectors into an Endee index called `documents`

Expected output:
```
Loading embedding model 'all-MiniLM-L6-v2'...
Model loaded!
Processing: artificial_intelligence.txt
  Split into 14 chunks
  ✓ Indexed 14 chunks from 'artificial_intelligence.txt'
...
✅ Done! Indexed 28 total chunks into Endee index 'documents'.
```

---

### Step 7 — Start the Web App

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 💬 Demo Questions to Try

Once running with the sample documents:

- *"What is Retrieval-Augmented Generation?"*
- *"How does Endee handle vector similarity search?"*
- *"What are the main types of machine learning?"*
- *"Explain cosine similarity and why it's used for text embeddings"*
- *"What is the difference between supervised and unsupervised learning?"*
- *"How does payload filtering work in Endee?"*

---

## 🔧 How Endee is Used

### Creating the Index (ingest.py)
```python
from endee import Endee, Precision

client = Endee()
client.set_base_url("http://localhost:8080/api/v1")

client.create_index(
    name="documents",
    dimension=384,          # all-MiniLM-L6-v2 output dimension
    space_type="cosine",    # cosine similarity metric
    precision=Precision.INT8  # memory-efficient quantization
)
```

### Upserting Document Vectors (ingest.py)
```python
index = client.get_index(name="documents")
index.upsert([{
    "id": "doc1_chunk0",
    "vector": embedding,    # 384-d float list from sentence-transformers
    "meta": {
        "source": "filename.txt",
        "chunk_index": 0,
        "text": "The actual chunk text goes here..."
    }
}])
```

### Semantic Search at Query Time (rag.py)
```python
query_vector = embedder.encode(user_question).tolist()
results = index.query(vector=query_vector, top_k=5)

for r in results:
    print(r["meta"]["text"])   # retrieved context
    print(r["similarity"])     # cosine similarity score
```

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Chat UI |
| `POST` | `/ask` | `{"question": "..."}` → `{"answer": "...", "sources": [...]}` |
| `POST` | `/ingest` | Trigger document re-indexing |
| `GET` | `/status` | Endee health + vector count |

---

## 🛑 Stopping the Application

```bash
# Stop Flask: Ctrl+C in the terminal

# Stop Endee
docker compose down

# Data is persisted in a Docker volume; run 'docker compose up -d' to restart
```

---

## 📄 License

This project is built on top of **Endee**, which is licensed under the Apache License 2.0.
See the [LICENSE](../LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Endee](https://github.com/endee-io/endee) — Open-source vector database
- [sentence-transformers](https://www.sbert.net/) — Free local text embeddings
- [Groq](https://groq.com/) — Ultra-fast LLM inference
- [Flask](https://flask.palletsprojects.com/) — Lightweight Python web framework
