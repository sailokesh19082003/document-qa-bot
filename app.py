"""
app.py
------
Streamlit web UI for the Document Q&A Bot.

This is the file Streamlit Community Cloud will run when deployed.
Run locally with:
    streamlit run app.py
"""

import os
import sys
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from config import GEMINI_API_KEY, DB_DIR
from query import query_rag_pipeline

st.set_page_config(page_title="Health & Nutrition Q&A Bot", page_icon="🥗", layout="centered")

st.title("🥗 Health & Nutrition Document Q&A Bot")
st.caption("Ask questions grounded strictly in the indexed documents — powered by RAG + Gemini.")

# ---- Setup checks ----
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is not set. Add it in your `.env` file (local) or in "
             "Streamlit Cloud's **Secrets** settings (deployed).")
    st.stop()

if not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
    st.error("No vector database found. Run `python src/ingest.py` first to index your documents.")
    st.stop()

# ---- Sidebar ----
with st.sidebar:
    st.header("About")
    st.write(
        "This bot answers questions using only the content of the documents "
        "in the `/data` folder. It will not use outside knowledge, and it "
        "will tell you honestly when an answer isn't in the documents."
    )
    top_k = st.slider("Number of chunks to retrieve (k)", min_value=1, max_value=8, value=4)

# ---- Chat history ----
if "history" not in st.session_state:
    st.session_state.history = []

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        if entry["citations"]:
            with st.expander("📚 Sources used"):
                for cite, snippet in zip(entry["citations"], entry["raw_context"]):
                    st.markdown(f"**{cite}**")
                    st.caption(snippet[:300] + ("..." if len(snippet) > 300 else ""))

# ---- New question ----
question = st.chat_input("Ask a question about the indexed documents...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating a grounded answer..."):
            result = query_rag_pipeline(question, k=top_k)
        st.write(result["answer"])
        if result["citations"]:
            with st.expander("📚 Sources used"):
                seen = set()
                for cite, snippet in zip(result["citations"], result["raw_context"]):
                    if cite not in seen:
                        seen.add(cite)
                        st.markdown(f"**{cite}**")
                        st.caption(snippet[:300] + ("..." if len(snippet) > 300 else ""))

    st.session_state.history.append({
        "question": question,
        "answer": result["answer"],
        "citations": result["citations"],
        "raw_context": result["raw_context"],
    })
