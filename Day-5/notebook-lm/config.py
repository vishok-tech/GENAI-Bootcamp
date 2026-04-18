# =============================================================================
# config.py — Central configuration for NotebookLM Replica
# =============================================================================

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
STORAGE_DIR   = BASE_DIR / "storage"
UPLOADS_DIR   = STORAGE_DIR / "uploads"
CHROMA_DIR    = STORAGE_DIR / "chroma_db"
NOTES_DIR     = STORAGE_DIR / "notes"

# Create dirs if they don't exist
for d in [UPLOADS_DIR, CHROMA_DIR, NOTES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Ollama ───────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL         = "llama3.2:3b"
EMBED_MODEL       = "nomic-embed-text"

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "notebooklm_docs"

# ── RAG ──────────────────────────────────────────────────────────────────────
CHUNK_SIZE        = 1000
CHUNK_OVERLAP     = 200
TOP_K             = 4

# ── Tavily ───────────────────────────────────────────────────────────────────
TAVILY_API_KEY    = os.getenv("TAVILY_API_KEY", "")

# ── App ──────────────────────────────────────────────────────────────────────
APP_TITLE         = "NotebookLM Replica"
APP_ICON          = "📓"
