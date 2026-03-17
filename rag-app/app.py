"""
app.py — Flask web application for the RAG Document Q&A system.

Routes:
  GET  /         → Chat UI
  POST /ask      → RAG pipeline (question → answer + sources)
  POST /ingest   → Trigger document ingestion
  GET  /status   → Check Endee server health & indexed doc count
"""

import os
import json
import subprocess
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

from rag import answer_question

load_dotenv()

ENDEE_HOST = os.getenv("ENDEE_HOST", "http://localhost:8080")

app = Flask(__name__)
CORS(app)


# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the chat UI."""
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    """
    Accepts JSON: { "question": "..." }
    Returns JSON: { "answer": "...", "sources": [...], "context_chunks": N }
    """
    data = request.get_json(force=True)
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    result = answer_question(question)
    return jsonify(result)


@app.route("/ingest", methods=["POST"])
def ingest():
    """Trigger document ingestion by running ingest.py as a subprocess."""
    try:
        proc = subprocess.run(
            ["python", "ingest.py"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        output = proc.stdout + proc.stderr
        success = proc.returncode == 0
        return jsonify({"success": success, "output": output})
    except subprocess.TimeoutExpired:
        return jsonify({"success": False, "output": "Ingestion timed out after 5 minutes."})
    except Exception as e:
        return jsonify({"success": False, "output": str(e)})


@app.route("/status", methods=["GET"])
def status():
    """Check Endee health and index stats."""
    import requests as req
    try:
        r = req.get(f"{ENDEE_HOST}/api/v1/indexes", timeout=3)
        indexes = r.json() if r.ok else []
        doc_index = next((i for i in indexes if i["name"] == "documents"), None)
        return jsonify({
            "endee_running": True,
            "index_exists": doc_index is not None,
            "vector_count": doc_index.get("count", 0) if doc_index else 0,
        })
    except Exception:
        return jsonify({
            "endee_running": False,
            "index_exists": False,
            "vector_count": 0,
        })


if __name__ == "__main__":
    print("🚀 Starting RAG Q&A Server...")
    print("   Open http://localhost:5000 in your browser")
    print("   Make sure Endee is running: docker compose up -d")
    app.run(debug=True, host="0.0.0.0", port=5000)
