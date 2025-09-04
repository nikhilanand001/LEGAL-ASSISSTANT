from flask import Flask, render_template, request, stream_with_context, Response, redirect, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import os
import tempfile
import fitz  # PyMuPDF
from llama_index.vector_stores.chroma import ChromaVectorStore
from index_loader import retriever
from llm_client import run_llm_query
from embed_and_index import chunk_text_hybrid, embedder

from chromadb import Client
from chromadb.config import Settings

import nltk
nltk.download('punkt')      # still needed
nltk.download('punkt_tab')  # new resource


load_dotenv()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# ========== HELPER FUNCTIONS ==========

def build_prompt(user_query, chunks):
    context = "\n".join(f"{i+1}. {chunk.text if hasattr(chunk, 'text') else chunk}" for i, chunk in enumerate(chunks))
    return f"""You are a legal assistant. Use the following context to answer the question.

Context:
{context}

Question: {user_query}
Answer:"""

def extract_text_from_file(file_path, extension):
    if extension == ".pdf":
        doc = fitz.open(file_path)
        return "\n".join([page.get_text() for page in doc])

    elif extension == ".docx":
        from docx import Document
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])

    elif extension == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    else:
        return ""

# ========== MAIN PAGE ROUTES ==========

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", answer=None, context=[])

@app.route("/query", methods=["POST"])
def query():
    user_query = request.form["query"].strip()

    # Retrieve relevant chunks from default index
    nodes = retriever.retrieve(user_query)
    prompt = build_prompt(user_query, nodes)

    def generate():
        for token in run_llm_query(prompt):
            yield token

    return Response(stream_with_context(generate()), mimetype="text/plain")

# ========== FILE UPLOAD ROUTE ==========

@app.route("/upload", methods=["GET", "POST"])

def upload():
    if request.method == "POST":
        file = request.files["document"]
        if file.filename == "":
            return "No file selected."

        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        _, ext = os.path.splitext(filename)
        ext = ext.lower()

        if ext not in [".pdf", ".docx", ".txt"]:
            return "Unsupported file type. Only PDF, DOCX, and TXT are allowed."

        text = extract_text_from_file(save_path, ext)
        if not text.strip():
            return "Failed to extract text. The file might be empty or unreadable."

        # Chunk and embed text
        chunks = chunk_text_hybrid(text)

               # Initialize persistent Chroma client if not already
        if "CHROMA_CLIENT" not in app.config:
            app.config["CHROMA_CLIENT"] = Client(Settings(anonymized_telemetry=False))
        chroma_client = app.config["CHROMA_CLIENT"]

        # Delete previous collection if exists
        try:
            chroma_client.delete_collection("upload_doc")
        except:
            pass

        collection = chroma_client.create_collection(name="upload_doc")

        embeddings = embedder.encode(chunks, convert_to_numpy=True)
        ids = [f"upload_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"chunk_id": id_} for id_ in ids]
        collection.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)
        app.config["TEMP_COLLECTION"] = collection
        app.config["UPLOAD_CHUNKS"] = chunks
        app.config["UPLOAD_TEXTS"] = [chunk.text if hasattr(chunk, "text") else chunk for chunk in chunks]
        return render_template("index.html", message="Upload successful. Now ask your question.")

    return render_template("index.html")

# ========== HANDLE QUERY ON UPLOADED DOC ==========

@app.route("/ask_upload", methods=["POST"])
def ask_upload():
    user_query = request.form["query"].strip()
    collection = app.config.get("TEMP_COLLECTION")
    texts = app.config.get("UPLOAD_TEXTS", [])

    if not collection or not texts:
        return "No uploaded document found. Please upload a document first."

    query_embedding = embedder.encode([user_query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    # Get document indices returned by Chroma
    ids = results["ids"][0]
    indices = [int(id_.split("_")[-1]) for id_ in ids]  # Assuming format: upload_chunk_0, etc.
    top_chunks = [texts[i] for i in indices]

    prompt = build_prompt(user_query, [type('Chunk', (), {"text": chunk}) for chunk in top_chunks])
    
    def generate():
        for token in run_llm_query(prompt):
            yield token

    return Response(stream_with_context(generate()), mimetype="text/plain")

# ========== UPLOAD AND QUERY IN ONE SHOT ==========

@app.route("/upload_and_query", methods=["POST"])
def upload_and_query():
    file = request.files.get("document")
    user_query = request.form.get("query", "").strip()

    if not file or file.filename == "":
        return "No file selected."
    if not user_query:
        return "No query provided."

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    if ext not in [".pdf", ".docx", ".txt"]:
        return "Unsupported file type. Only PDF, DOCX, and TXT are allowed."

    text = extract_text_from_file(save_path, ext)
    if not text.strip():
        return "Failed to extract text."

    chunks = chunk_text_hybrid(text)
    temp_client = Client(Settings(anonymized_telemetry=False))
    try:
        temp_client.delete_collection("temp_upload_combined")
    except:
        pass
    temp_collection = temp_client.create_collection(name="temp_upload_combined")

    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    ids = [f"combined_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"chunk_id": id_} for id_ in ids]
    temp_collection.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)

    query_embedding = embedder.encode([user_query])[0]
    results = temp_collection.query(query_embeddings=[query_embedding], n_results=3)
    top_chunks = results["documents"][0]

    context = "\n".join(top_chunks)
    prompt = f"""You are a legal assistant. Use the following context from the uploaded document to answer the question.

Context:
{context}

Question: {user_query}
Answer:"""

    def generate():
        for token in run_llm_query(prompt):
            yield token

    return Response(stream_with_context(generate()), mimetype="text/plain")

# ========== MAIN ==========
if __name__ == "__main__":
    app.run(debug=True)



