"""
config.py
---------
Central place for all configuration constants used across the project.
Keeping these in one file means you only have to change a value once
(e.g. chunk size) instead of hunting through every script.
"""

import os
from dotenv import load_dotenv

# Load variables from the .env file into the environment
load_dotenv()

# ---- API Key ----
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ---- Models ----
# NOTE: The original Gemini "text-embedding-004" and the old
# "google-generativeai" SDK have been retired by Google.
# This project uses the current, supported replacements:
#   - Embedding model : gemini-embedding-001
#   - Generation model: gemini-2.5-flash
#   - SDK             : google-genai (the new unified SDK)
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-2.5-flash"

# ---- Paths ----
DATA_DIR = "data"
DB_DIR = "db"
COLLECTION_NAME = "health_nutrition_knowledge_base"

# ---- Chunking ----
CHUNK_SIZE = 1000        # characters per chunk
CHUNK_OVERLAP = 200      # overlap between consecutive chunks

# ---- Retrieval ----
TOP_K = 4                # number of chunks to retrieve per query
