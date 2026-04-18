# =============================================================================
# core/document_processor.py — PDF loading, chunking, metadata attachment
# =============================================================================

import hashlib
from datetime import datetime
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import CHUNK_SIZE, CHUNK_OVERLAP, UPLOADS_DIR


def get_file_hash(file_path: str) -> str:
    """Generate MD5 hash of a file for deduplication."""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def save_uploaded_file(uploaded_file) -> str:
    """
    Save a Streamlit UploadedFile to the uploads directory.
    Returns the saved file path.
    """
    save_path = UPLOADS_DIR / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(save_path)


def load_and_chunk_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF and split it into chunks with metadata.

    Metadata attached per chunk:
      - filename   : original PDF filename
      - page_number: page the chunk came from (1-indexed)
      - upload_date: ISO timestamp of when it was processed
      - file_hash  : MD5 hash for deduplication
      - source     : full file path
    """
    filename    = Path(file_path).name
    upload_date = datetime.now().isoformat()
    file_hash   = get_file_hash(file_path)

    # Load PDF pages
    loader = PyPDFLoader(file_path)
    pages  = loader.load()  # each page is a Document with page metadata

    # Chunk with overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators    = ["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    # Enrich metadata on every chunk
    for chunk in chunks:
        # PyPDFLoader sets page as 0-indexed int
        raw_page = chunk.metadata.get("page", 0)
        chunk.metadata.update({
            "filename"   : filename,
            "page_number": int(raw_page) + 1,   # human-readable 1-indexed
            "upload_date": upload_date,
            "file_hash"  : file_hash,
            "source"     : file_path,
        })

    return chunks


def list_uploaded_pdfs() -> List[str]:
    """Return list of PDF filenames already in the uploads directory."""
    return [f.name for f in UPLOADS_DIR.glob("*.pdf")]
