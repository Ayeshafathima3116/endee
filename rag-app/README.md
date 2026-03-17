# 🧠 DocuMind: Intelligent RAG Knowledge Base

DocuMind is a high-performance Retrieval-Augmented Generation (RAG) application built to demonstrate the power of **Endee** as a vector database. It allows users to upload documents (PDFs, Text) or crawl web URLs to build a searchable, private intelligence core.

## 🚀 Features
- **Semantic RAG**: Context-aware answers powered by Llama 3 (via Groq) and Endee.
- **Web & URL Ingestion**: Headless scraping of webpages to expand your knowledge base.
- **Document Lifecycle Manager**: Full CRUD operations on indexed documents and their vector embeddings.
- **Smart Follow-ups**: AI-driven suggestions for deep-dive exploration.
- **Premium UI**: Modern, dark-mode interface with streaming responses and real-time metrics.

## 🏗️ System Design
DocuMind follows a modular RAG architecture:

1.  **Frontend**: Vanilla JavaScript + SSE (Server-Sent Events) for real-time streaming and a responsive dark-themed UI.
2.  **API Layer**: Flask (Python) handles document processing, web scraping, and RAG orchestration.
3.  **Vector Core**: [Endee](https://github.com/endee-io/endee) manages high-dimensional vector storage and similarity search.
4.  **LLM Engine**: Groq (Llama-3.3-70b) provides lighting-fast reasoning and response generation.
5.  **Embeddings**: `sentence-transformers` (all-MiniLM-L6-v2) converts text into 384-dimensional vectors.

## 🔌 Use of Endee
Endee is the backbone of DocuMind's intelligence. It is used for:
-   **Dynamic Indexing**: Documents are chunked and converted into vectors, then pushed to an Endee index.
-   **Semantic Retrieval**: Queries are embedded in real-time and matched against the Endee vector space using cosine similarity.
-   **Targeted Deletion**: Leverages Endee's `delete_with_filter` to purge specific document vectors by metadata (`source` field) when a user removes a file.

## 🛠️ Setup Instructions

### 1. Prerequisites
- Python 3.9+
- Docker (for running Endee)
- Groq API Key

### 2. Mandatory Repository Steps
1. Star the [Endee Repository](https://github.com/endee-io/endee).
2. Fork the repository to your account.
3. Clone your fork and move these project files into it.

### 3. Backend Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "GROQ_API_KEY=your_key_here" > .env
```

### 4. Running Endee
```bash
docker compose up -d
```

### 5. Start Application
```bash
python app.py
```
Visit `http://localhost:5000` to start chatting!

---
*Developed for the Endee AI/ML Project Assessment.*
