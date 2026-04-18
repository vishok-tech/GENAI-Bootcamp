# =============================================================================
# core/vector_store.py — ChromaDB vector store operations
# =============================================================================

from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from config import CHROMA_DIR, CHROMA_COLLECTION, EMBED_MODEL, OLLAMA_BASE_URL


def get_embeddings() -> OllamaEmbeddings:
    """Return the Ollama nomic-embed-text embedding model."""
    return OllamaEmbeddings(
        model   = EMBED_MODEL,
        base_url= OLLAMA_BASE_URL,
    )


def get_vector_store() -> Chroma:
    """
    Return a persistent ChromaDB vector store.
    Creates it if it doesn't exist yet.
    """
    return Chroma(
        collection_name     = CHROMA_COLLECTION,
        embedding_function  = get_embeddings(),
        persist_directory   = str(CHROMA_DIR),
    )


def add_documents(chunks: List[Document]) -> int:
    """
    Add document chunks to ChromaDB.
    Returns the number of chunks added.
    """
    vs = get_vector_store()
    vs.add_documents(chunks)
    return len(chunks)


def similarity_search(
    query           : str,
    selected_docs   : Optional[List[str]] = None,
    k               : int = 4,
) -> List[Document]:
    """
    Retrieve top-k relevant chunks for a query.

    If selected_docs is provided, filters results to only those filenames
    using ChromaDB's metadata where filter.
    """
    vs = get_vector_store()

    if selected_docs:
        # ChromaDB where filter — only retrieve from selected documents
        where_filter = {"filename": {"$in": selected_docs}}
        results = vs.similarity_search(
            query  = query,
            k      = k,
            filter = where_filter,
        )
    else:
        results = vs.similarity_search(query=query, k=k)

    return results


def get_stored_filenames() -> List[str]:
    """
    Return a list of unique filenames already indexed in ChromaDB.
    Used to show which documents are available to search.
    """
    vs         = get_vector_store()
    collection = vs._collection
    # Get all metadata and extract unique filenames
    data       = collection.get(include=["metadatas"])
    filenames  = set()
    for meta in data.get("metadatas", []):
        if meta and "filename" in meta:
            filenames.add(meta["filename"])
    return sorted(list(filenames))


def delete_document(filename: str) -> int:
    """
    Remove all chunks belonging to a specific document from ChromaDB.
    Returns the number of chunks deleted.
    """
    vs         = get_vector_store()
    collection = vs._collection
    results    = collection.get(
        where   = {"filename": filename},
        include = ["documents"],
    )
    ids = results.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)
