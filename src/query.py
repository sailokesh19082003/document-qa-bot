"""
query.py
--------
This is the QUERYING half of the RAG pipeline.

Given a user's natural-language question, this module:
1. Embeds the question using the SAME embedding model used during ingestion.
2. Searches the persisted ChromaDB collection for the top-k most similar chunks.
3. Builds a "grounded" prompt that forces the LLM to answer ONLY from those
   chunks (to prevent hallucination).
4. Calls Gemini to generate the final answer.
5. Returns the answer together with clear source citations.

It does NOT re-embed or re-index any documents -- that already happened in
ingest.py. This keeps indexing and querying cleanly separated, as required
by the assignment.
"""

import os
import sys
import chromadb
from google import genai

sys.path.append(os.path.dirname(__file__))
from config import (
    GEMINI_API_KEY, EMBEDDING_MODEL, GENERATION_MODEL,
    DB_DIR, COLLECTION_NAME, TOP_K,
)

client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = (
    "You are a precise document Q&A assistant for a health and nutrition "
    "knowledge base. Use ONLY the provided context to answer the user's "
    "question. Cite the source (filename and page number) inline next to "
    "every fact you state, in the format (filename, Page X). "
    "If the answer cannot be found in the context, say exactly: "
    "'I cannot find the answer in the provided documents.' "
    "Do not use your own outside knowledge to answer."
)


def embed_query(query: str) -> list[float]:
    """Embed a single user query using the same model used for the documents."""
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[query],
    )
    return result.embeddings[0].values


def retrieve_chunks(query: str, k: int = TOP_K, db_path: str = DB_DIR):
    """Embed the query and fetch the top-k most similar chunks from ChromaDB."""
    db_client = chromadb.PersistentClient(path=db_path)
    collection = db_client.get_collection(COLLECTION_NAME)

    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return documents, metadatas, distances


def build_prompt(query: str, documents: list[str], metadatas: list[dict]) -> str:
    """Assemble the retrieved chunks and citations into a grounded prompt."""
    context_blocks = []
    for doc, meta in zip(documents, metadatas):
        citation = f"Source: {meta['source']}, Page: {meta['page']}"
        context_blocks.append(f"[{citation}]\nContext: {doc}")

    context_payload = "\n\n---\n\n".join(context_blocks)

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT INFORMATION:\n{context_payload}\n\n"
        f"USER QUESTION: {query}\n\n"
        f"GROUNDED ANSWER:"
    )


def query_rag_pipeline(user_query: str, k: int = TOP_K, db_path: str = DB_DIR) -> dict:
    """Full query pipeline: retrieve -> build prompt -> generate answer."""
    documents, metadatas, distances = retrieve_chunks(user_query, k, db_path)

    if not documents:
        return {
            "answer": "I cannot find the answer in the provided documents.",
            "citations": [],
            "raw_context": [],
        }

    prompt = build_prompt(user_query, documents, metadatas)

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )

    citations = [
        f"{meta['source']}, Page {meta['page']}" for meta in metadatas
    ]

    return {
        "answer": response.text,
        "citations": citations,
        "raw_context": documents,
        "distances": distances,
    }


if __name__ == "__main__":
    # Quick manual test: python src/query.py "your question here"
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        result = query_rag_pipeline(q)
        print("\nANSWER:\n", result["answer"])
        print("\nCITATIONS:")
        for c in result["citations"]:
            print(" -", c)
    else:
        print("Usage: python src/query.py \"your question here\"")
