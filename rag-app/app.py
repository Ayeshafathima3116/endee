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
import re
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv

from endee import Endee
from rag import stream_answer

load_dotenv()

ENDEE_HOST = os.getenv("ENDEE_HOST", "http://localhost:8080")
INDEX_NAME = "documents"

app = Flask(__name__)
CORS(app)

# Initialize Endee client
client = Endee()
client.set_base_url(f"{ENDEE_HOST}/api/v1")


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
    Returns SSE stream of JSON chunks.
    """
    data = request.get_json(force=True)
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    def generate():
        for chunk in stream_answer(question):
            yield f"data: {json.dumps(chunk)}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route("/upload", methods=["POST"])
def upload():
    """Accepts multipart/form-data with 'file'. Saves to docs/ directory."""
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400

    if file:
        docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
        if not os.path.exists(docs_dir):
            os.makedirs(docs_dir)
        
        filepath = os.path.join(docs_dir, file.filename)
        file.save(filepath)
        return jsonify({"success": True, "filename": file.filename})

@app.route("/ingest-url", methods=["POST"])
def ingest_url():
    """Fetches content from a URL, cleans it, and saves it as a .txt file for ingestion."""
    data = request.get_json(force=True)
    url = data.get("url", "").strip()

    if not url:
        return jsonify({"success": False, "error": "No URL provided"}), 400

    try:
        # 1. Fetch
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # 2. Parse & Clean
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove noise
        for ex in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            ex.decompose()
            
        title = soup.title.string if soup.title else "web_content"
        text = soup.get_text(separator='\n')
        
        # Clean whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # 3. Save
        filename = re.sub(r'[^\w\s-]', '', title).strip().lower().replace(' ', '_')
        if not filename: filename = "web_extract_" + str(hash(url))[:8]
        filename += ".txt"
        
        docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
        if not os.path.exists(docs_dir): os.makedirs(docs_dir)
        
        filepath = os.path.join(docs_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"SOURCE URL: {url}\n\n{text}")
            
        return jsonify({"success": True, "filename": filename})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


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


@app.route("/documents", methods=["GET"])
def list_documents():
    """Returns a list of all documents currently in the docs/ directory."""
    docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
    if not os.path.exists(docs_dir):
        return jsonify({"documents": []})
    
    files = []
    for f in os.listdir(docs_dir):
        if f.endswith(('.txt', '.pdf', '.docx')):
            path = os.path.join(docs_dir, f)
            try:
                stats = os.stat(path)
                files.append({
                    "name": f,
                    "size": stats.st_size,
                    "modified": stats.st_mtime
                })
            except: pass
    return jsonify({"documents": files})


@app.route("/documents/<path:filename>", methods=["DELETE"])
def delete_document(filename):
    """
    1. Delete document vectors from Endee using filter.
    2. Delete the file from the docs/ directory.
    """
    try:
        # 1. Purge from Endee
        idx = client.get_index(name=INDEX_NAME)
        # Endee expects filter as an array: [{"field": {"$eq": value}}]
        idx.delete_with_filter(filter=[{"source": {"$eq": filename}}])
        
        # 2. Delete from disk
        docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
        filepath = os.path.join(docs_dir, filename)
        
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({"success": True, "message": f"Purged {filename} from knowledge base."})
        else:
            return jsonify({"success": True, "message": f"Purged {filename} from vectors, but file not found on disk."})
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/status", methods=["GET"])
def status():
    """Check Endee health and index stats."""
    try:
        # Use existing global client
        index_list_resp = client.list_indexes()
        indexes = index_list_resp.get('indexes', [])
        
        # Try to get more accurate count from describe() if index exists
        vector_count = 0
        doc_found = False
        for idx in indexes:
            if idx.get("name") == INDEX_NAME:
                doc_found = True
                # Describe for real-time count
                try:
                    target_index = client.get_index(name=INDEX_NAME)
                    description = target_index.describe()
                    # Check both 'count' (from describe) and 'total_elements' (from metadata)
                    vector_count = description.get('count') or description.get('total_elements', 0)
                except:
                    # Fallback to list_indexes count if describe fails
                    vector_count = idx.get("total_elements", 0)
                break
                
        # Count documents on disk
        docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
        doc_count = 0
        if os.path.exists(docs_dir):
            doc_count = len([f for f in os.listdir(docs_dir) if f.endswith(('.txt', '.pdf', '.docx'))])
            
        return jsonify({
            "endee_running": True,
            "index_exists": doc_found,
            "vector_count": vector_count,
            "doc_count": doc_count
        })
    except Exception as e:
        print(f"Status check error: {e}")
        return jsonify({
            "endee_running": False,
            "index_exists": False,
            "vector_count": 0,
            "doc_count": 0
        })


if __name__ == "__main__":
    print("🚀 Starting RAG Q&A Server...")
    print("   Open http://localhost:5000 in your browser")
    print("   Make sure Endee is running: docker compose up -d")
    app.run(debug=True, host="0.0.0.0", port=5000)
