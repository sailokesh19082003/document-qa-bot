# Health & Nutrition Document Q&A Bot (RAG)

🔗 **Live Demo:** https://document-app-bot-bbeuh94vht5zwz74agyxjw.streamlit.app/


A command-line and web-based Q&A bot that answers questions strictly from a
local collection of health & nutrition documents, using Retrieval-Augmented
Generation (RAG). The bot never relies on the LLM's own training knowledge —
every answer is grounded in retrieved document chunks and comes with source
citations (filename + page number).

Built as part of an AI Engineering Internship assignment.

## Tech Stack

| Component | Tool / Library | Version |
|---|---|---|
| Language | Python | 3.11+ |
| PDF parsing | `pypdf` | >=4.0.0 |
| DOCX parsing | `python-docx` | >=1.1.0 |
| Vector database | `chromadb` (local, disk-persistent) | >=0.5.0 |
| Embedding model | Google `gemini-embedding-001` | via `google-genai` SDK |
| Generation model | Google `gemini-2.5-flash` | via `google-genai` SDK |
| Env management | `python-dotenv` | >=1.0.1 |
| Progress bars | `tqdm` | >=4.66.0 |
| Web UI | `streamlit` | >=1.30.0 |

> **Note on model choice:** Google has retired the `text-embedding-004`
> model and the legacy `google-generativeai` SDK that older tutorials
> reference. This project uses the current, supported equivalents:
> `gemini-embedding-001` for embeddings and `gemini-2.5-flash` for
> generation, accessed through the new unified `google-genai` SDK.

## Architecture Overview

```
 data/ (PDF, DOCX)
       |
       v
 [ingest.py]  --extract text page-by-page-->  [chunk_text()]
       |                                              |
       |                                  overlapping ~1000-char chunks
       |                                              |
       v                                              v
 [embed_texts_batch()] --batched Gemini embedding calls--> [ChromaDB: db/]
                                                                  |
                                                    (persisted to disk, once)

 User question
       |
       v
 [query.py: embed_query()] --> [ChromaDB similarity search, top-k]
       |
       v
 [build_prompt()] --strict grounding instructions + retrieved chunks-->
       |
       v
 [Gemini 2.5 Flash] --> Grounded answer + inline citations
       |
       v
 [main.py CLI]  or  [app.py Streamlit UI]
```

Ingestion (`ingest.py`) and querying (`query.py`) are kept as two separate
scripts. You only run `ingest.py` when your documents change; `query.py` /
`main.py` / `app.py` just load the already-built database from disk, so no
repeated embedding cost on every run.

## Chunking Strategy

**Fixed-size chunking with overlap, snapped to word boundaries.**

- Chunk size: **1000 characters**
- Overlap: **200 characters**
- Each chunk attempts to end at the nearest space rather than cutting a
  word in half.

**Why this approach:** Fixed-size chunking is simple, predictable, and fast
to implement correctly, which matters for a 3-day assignment. 1000
characters (roughly 150-200 words) is large enough to preserve a paragraph's
worth of context but small enough to keep retrieval precise and avoid
diluting the LLM's prompt with irrelevant text. The 200-character overlap
ensures that if a key fact sits right on a chunk boundary, it still appears
fully in at least one chunk. Every chunk keeps the source filename and page
number in its metadata, so citations are always traceable back to the exact
page.

## Embedding Model & Vector Database

- **Embedding model:** `gemini-embedding-001` (Google). Chosen because the
  project already uses Gemini for generation, keeping the API surface and
  billing in one place, and because embedding calls are batched (multiple
  chunks sent per API request) rather than looped one at a time, which is
  both faster and required by the assignment.
- **Vector database:** `ChromaDB`, running locally with `PersistentClient`,
  storing data on disk under `db/`. Chosen because it requires no separate
  server process, persists automatically between runs, and has a simple
  Python API well-suited to a small, single-machine project like this one.
  Cosine similarity (`hnsw:space: cosine`) is used for nearest-neighbor
  search.

## Setup Instructions

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd document-qa-bot
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your API key
```bash
cp .env.example .env
```
Open `.env` and paste your Gemini API key:
```
GEMINI_API_KEY=your_actual_key_here
```
Get a free key at: https://aistudio.google.com/apikey

### 5. Index the documents (run once)
```bash
python src/ingest.py
```
This reads everything in `data/`, chunks it, embeds it in batches, and
saves the vector database to `db/`. You only need to re-run this if you
add or change documents.

### 6. Run the bot

**Command-line interface (required):**
```bash
python src/main.py
```

**Streamlit web UI (bonus):**
```bash
streamlit run app.py
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Your Google Gemini API key. Get one free at https://aistudio.google.com/apikey. **Never commit this to git** — it belongs only in your local `.env` file (already excluded via `.gitignore`) or in Streamlit Cloud's Secrets manager when deployed. |

## Example Queries

The knowledge base covers five documents on health and nutrition topics.
Try questions like:

1. **"How much protein should an athlete eat per day?"** — answered from
   `macronutrients_guide.pdf`.
2. **"What are the symptoms of vitamin D deficiency?"** — answered from
   `micronutrients_and_deficiencies.pdf`.
3. **"How much fluid should I drink during exercise lasting over an hour?"**
   — answered from `hydration_and_exercise_nutrition.pdf`.
4. **"How does caffeine affect sleep?"** — answered from
   `sleep_stress_and_nutrition.docx`.
5. **"What foods support a healthy gut microbiome?"** — answered from
   `gut_health_and_microbiome.docx`.
6. **"What is the capital of France?"** (out-of-scope test) — the bot
   should respond: *"I cannot find the answer in the provided documents."*

## Known Limitations

- **Fixed-size chunking** can occasionally split a sentence mid-thought even
  with word-boundary snapping, which may slightly reduce answer precision
  for facts that span a chunk boundary.
- **DOCX page numbers are not tracked** — Word documents don't have a fixed
  pagination the way PDFs do (page count depends on viewer/zoom), so DOCX
  citations report "Page 1" as a placeholder rather than a true page number.
- **No re-ranking step** — retrieval relies purely on cosine similarity
  from the embedding model; a cross-encoder re-ranker could improve
  precision further but was out of scope for this assignment.
- **Single-turn retrieval** — each question is embedded and retrieved
  independently; the bot doesn't yet use conversation history to inform
  retrieval for follow-up questions (e.g., "what about for women?" after a
  previous question), even though the Streamlit UI does display chat
  history.
- **English-only** — extraction and embedding have only been tested on
  English-language documents.

## Project Structure

```
document-qa-bot/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── app.py                 # Streamlit web UI
├── data/                  # 5 source documents (3 PDF, 2 DOCX)
├── db/                    # Persistent ChromaDB storage (generated by ingest.py)
└── src/
    ├── __init__.py
    ├── config.py           # Constants: models, paths, chunk size, top-k
    ├── ingest.py            # Indexing pipeline
    ├── query.py             # Retrieval + generation pipeline
    └── main.py              # CLI interactive loop
```

## Deployment (Streamlit Community Cloud)

1. Push this repository to a **public** GitHub repo.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click "New app", select your repo, branch `main`, and set the main file
   path to `app.py`.
4. Under **Advanced settings > Secrets**, add:
   ```
   GEMINI_API_KEY = "your_actual_key_here"
   ```
5. Deploy. Note: the `db/` folder must already contain your indexed data —
   run `python src/ingest.py` locally first and commit the resulting `db/`
   folder, since Streamlit Cloud cannot run a one-off indexing script for
   you before app startup.
