from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import sqlite3
import uuid

app = FastAPI()

# ---------- DATABASE ----------
conn = sqlite3.connect("rag.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT,
    chunk_index INTEGER,
    chunk_text TEXT
)
""")

conn.commit()

# ---------- EMBEDDINGS ----------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("rag_chunks")

# ---------- HELPERS ----------
def extract_text(file, filename):
    if filename.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join([p.extract_text() for p in reader.pages])
    return file.read().decode("utf-8")

def chunk_text(text, size=1000, overlap=500):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

# ---------- API MODELS ----------
class AskRequest(BaseModel):
    document_id: str
    question: str

# ---------- UPLOAD ----------
@app.post("/upload")
def upload(file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())
    text = extract_text(file.file, file.filename)

    cursor.execute(
        "INSERT INTO documents VALUES (?, ?)",
        (doc_id, file.filename)
    )

    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        chunk_id = str(uuid.uuid4())

        cursor.execute(
            "INSERT INTO chunks VALUES (?, ?, ?, ?)",
            (chunk_id, doc_id, i, chunk)
        )

        embedding = embedder.encode(chunk).tolist()

        # store document_id in metadata
        collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"document_id": doc_id}]
        )

    conn.commit()

    return {
        "document_id": doc_id,
        "status": "uploaded"
    }

# ---------- ASK ----------
@app.post("/ask")
def ask(req: AskRequest):
    query_embedding = embedder.encode(req.question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "distances"]
    )

    # No chunks found at all
    if not results["documents"] or not results["documents"][0]:
        return {
            "answer": "Invalid question. Answer not found in the uploaded document.",
            "citations": []
        }

    chunks = results["documents"][0]
    ids = results["ids"][0]
    distances = results["distances"][0]

    best_chunk = None
    best_similarity = 0
    best_citation = None

    # Find best matching chunk
    for chunk, cid, dist in zip(chunks, ids, distances):
        similarity = 1 - dist  # cosine similarity
        if similarity > best_similarity:
            best_similarity = similarity
            best_chunk = chunk
            best_citation = cid

    SIMILARITY_THRESHOLD = 0.4

    # Question is related to document
    if best_chunk and best_similarity >= SIMILARITY_THRESHOLD:
        return {
            "answer": "Answer from the uploaded document:\n\n" + best_chunk[:800],
            "citations": [best_citation]
        }

    # Question is outside document
    return {
        "answer": "Invalid question. Answer not found in the uploaded document.",
        "citations": []
    } 