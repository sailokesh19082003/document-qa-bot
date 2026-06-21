"""
ingest.py
---------
This is the INDEXING half of the RAG pipeline.

Run this once (and again any time you add/change documents in /data):
    python src/ingest.py

What it does, step by step:
1. Scans the /data folder for .pdf, .docx, and .txt files.
2. Extracts text page-by-page (PDF) or as whole-document text (DOCX/TXT),
   keeping track of WHICH file and WHICH page each piece of text came from.
3. Splits that text into overlapping "chunks" so the LLM only ever sees
   small, relevant pieces instead of whole documents.
4. Sends all chunks to Gemini's embedding model IN A SINGLE BATCH CALL
   (not one-by-one in a loop -- batching is faster and required by the
   assignment).
5. Saves everything into a local, disk-persistent ChromaDB collection so
   we never have to repeat the embedding step again.
"""

import os
import sys
from pypdf import PdfReader
from docx import Document as DocxDocument
from tqdm import tqdm
import chromadb
from google import genai

sys.path.append(os.path.dirname(__file__))
from config import (
    GEMINI_API_KEY, EMBEDDING_MODEL, DATA_DIR, DB_DIR,
    COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP,
)

client = genai.Client(api_key=GEMINI_API_KEY)


# ---------------------------------------------------------------------
# STEP 1: Document extraction
# ---------------------------------------------------------------------

def extract_pdf_pages(file_path: str) -> list[dict]:
    """Extract text page-by-page from a PDF, tracking page numbers."""
    extracted = []
    file_name = os.path.basename(file_path)
    reader = PdfReader(file_path)
    for index, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            clean_text = " ".join(text.split())  # collapse extra whitespace
            extracted.append({
                "text": clean_text,
                "metadata": {"source": file_name, "page": index + 1},
            })
    return extracted


def extract_docx_pages(file_path: str) -> list[dict]:
    """Extract text from a DOCX file. DOCX has no fixed 'pages', so we
    treat the whole document as page 1 and rely on chunking afterwards."""
    file_name = os.path.basename(file_path)
    doc = DocxDocument(file_path)
    full_text = " ".join(p.text for p in doc.paragraphs if p.text.strip())
    if not full_text.strip():
        return []
    return [{"text": full_text, "metadata": {"source": file_name, "page": 1}}]


def extract_txt_pages(file_path: str) -> list[dict]:
    """Extract text from a plain .txt file."""
    file_name = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        return []
    return [{"text": " ".join(text.split()), "metadata": {"source": file_name, "page": 1}}]


def load_all_documents(data_dir: str) -> list[dict]:
    """Scan the data directory and extract text from every supported file."""
    all_pages = []
    files = sorted(os.listdir(data_dir))
    for fname in files:
        path = os.path.join(data_dir, fname)
        ext = fname.lower().split(".")[-1]
        try:
            if ext == "pdf":
                pages = extract_pdf_pages(path)
            elif ext == "docx":
                pages = extract_docx_pages(path)
            elif ext == "txt":
                pages = extract_txt_pages(path)
            else:
                continue  # skip unsupported file types
            print(f"  Extracted {len(pages)} page(s) from {fname}")
            all_pages.extend(pages)
        except Exception as e:
            print(f"  Error reading {fname}: {e}")
    return all_pages


# ---------------------------------------------------------------------
# STEP 2: Recursive-style chunking with overlap
# ---------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split text into overlapping chunks, preferring to break on paragraph
    or sentence boundaries rather than cutting mid-word whenever possible.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # Try to end the chunk at a natural boundary (space) so we don't
        # cut a word in half.
        if end < text_length:
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        chunks.append(text[start:end].strip())

        # Slide the window forward, leaving the overlap in place.
        start += (chunk_size - chunk_overlap)

    return [c for c in chunks if c]


def chunk_extracted_pages(pages: list[dict], chunk_size: int, chunk_overlap: int) -> list[dict]:
    """Apply chunk_text() to every extracted page and carry metadata over."""
    all_chunks = []
    for page in pages:
        pieces = chunk_text(page["text"], chunk_size, chunk_overlap)
        for i, piece in enumerate(pieces):
            all_chunks.append({
                "text": piece,
                "metadata": {
                    "source": page["metadata"]["source"],
                    "page": page["metadata"]["page"],
                    "chunk_index": i,
                },
            })
    return all_chunks


# ---------------------------------------------------------------------
# STEP 3: Batch embedding + persisting to ChromaDB
# ---------------------------------------------------------------------

def embed_texts_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts in batches using Gemini's embedding model.
    Batching (instead of one API call per chunk) is required by the
    assignment and is much faster / cheaper.
    """
    all_embeddings = []
    batch_size = 100  # Gemini API batch limit per request

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
        batch = texts[i:i + batch_size]
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
        )
        all_embeddings.extend([e.values for e in result.embeddings])

    return all_embeddings


def save_to_vector_db(chunks: list[dict], db_path: str = DB_DIR):
    """Embed all chunks (batched) and store them in a persistent ChromaDB collection."""
    if not chunks:
        print("No chunks to index. Add documents to the data/ folder first.")
        return

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [f"id_{i}" for i in range(len(chunks))]

    print(f"\nEmbedding {len(texts)} chunks in batches...")
    embeddings = embed_texts_batch(texts)

    db_client = chromadb.PersistentClient(path=db_path)

    # Start fresh each time ingest.py is run, so re-running never duplicates data.
    try:
        db_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = db_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"Successfully indexed {len(chunks)} chunks into '{COLLECTION_NAME}' at {db_path}/")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main():
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not found. Did you create a .env file? See README.md.")
        return

    print(f"Scanning '{DATA_DIR}/' for documents...")
    pages = load_all_documents(DATA_DIR)
    print(f"\nTotal pages/sections extracted: {len(pages)}")

    print("\nChunking text...")
    chunks = chunk_extracted_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Total chunks created: {len(chunks)}")

    save_to_vector_db(chunks)


if __name__ == "__main__":
    main()
