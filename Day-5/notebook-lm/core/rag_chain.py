# =============================================================================
# core/rag_chain.py — RAG chain: retrieve → format → generate answer
# =============================================================================

from typing import List, Optional
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

from config import LLM_MODEL, OLLAMA_BASE_URL, TOP_K
from core.prompts import RAG_PROMPT, WEB_SEARCH_PROMPT
from core.vector_store import similarity_search


def get_llm() -> OllamaLLM:
    """Return the Ollama llama3.2:3b LLM."""
    return OllamaLLM(
        model    = LLM_MODEL,
        base_url = OLLAMA_BASE_URL,
        temperature = 0.1,
    )


def format_context(docs: List[Document]) -> str:
    """
    Format retrieved document chunks into a readable context string
    with source information.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        filename = doc.metadata.get("filename", "Unknown")
        page     = doc.metadata.get("page_number", "?")
        parts.append(
            f"[Chunk {i} — {filename}, Page {page}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def format_citations(docs: List[Document]) -> str:
    """Build a citation string from retrieved documents."""
    seen     = set()
    citations = []
    for doc in docs:
        filename = doc.metadata.get("filename", "Unknown")
        page     = doc.metadata.get("page_number", "?")
        key      = f"{filename}::{page}"
        if key not in seen:
            seen.add(key)
            citations.append(f"📄 {filename} — Page {page}")
    return "\n".join(citations)


def rag_answer(
    question      : str,
    selected_docs : Optional[List[str]] = None,
) -> dict:
    """
    Full RAG pipeline:
      1. Retrieve relevant chunks from ChromaDB (filtered by selected docs)
      2. Format context
      3. Generate answer using LLM
      4. Return answer + citations + source chunks

    Returns:
        {
            "answer"    : str,
            "citations" : str,
            "sources"   : List[Document],
            "context"   : str,
        }
    """
    # Step 1 — Retrieve
    docs = similarity_search(question, selected_docs=selected_docs, k=TOP_K)

    if not docs:
        return {
            "answer"   : "I couldn't find relevant information in the selected documents. "
                         "Please make sure you've uploaded and selected the right PDFs.",
            "citations": "",
            "sources"  : [],
            "context"  : "",
        }

    # Step 2 — Format context
    context = format_context(docs)

    # Step 3 — Generate answer
    llm    = get_llm()
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=RAG_PROMPT,
    )
    chain  = prompt | llm
    answer = chain.invoke({"context": context, "question": question})

    # Step 4 — Build citations
    citations = format_citations(docs)

    return {
        "answer"   : str(answer),
        "citations": citations,
        "sources"  : docs,
        "context"  : context,
    }


def web_search_answer(question: str, search_results: str) -> str:
    """
    Generate an answer from web search results using the LLM.
    """
    llm    = get_llm()
    prompt = PromptTemplate(
        input_variables=["search_results", "question"],
        template=WEB_SEARCH_PROMPT,
    )
    chain  = prompt | llm
    answer = chain.invoke({
        "search_results": search_results,
        "question"      : question,
    })
    return str(answer)
