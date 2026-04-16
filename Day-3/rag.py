"""
Day 3 — RAG: Vector Store, Semantic Retriever & RAG Chain
GenAI Bootcamp | Nunnari Academy

Prerequisites (run once):
    pip install langchain langchain-community langchain-text-splitters
                chromadb pypdf langchain-ollama

Ollama models needed (run once):
    ollama pull nomic-embed-text
    ollama pull qwen2.5:1.5b
"""

from datetime import date

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

PDF_FILES = [
    {
        "path": "paper1.pdf",
        "source_type": "research_paper",
        "upload_date": str(date.today()),
    },
    {
        "path": "textbook_chapter.pdf",
        "source_type": "textbook",
        "upload_date": str(date.today()),
    },
]

EMBED_MODEL  = "nomic-embed-text"   # Ollama embedding model
CHAT_MODEL   = "qwen2.5:1.5b"       # Ollama chat model
CHROMA_DIR   = "./chroma_db"         # where ChromaDB persists on disk
COLLECTION   = "day3_rag"
TOP_K        = 3                     # chunks to retrieve per query


# ─────────────────────────────────────────────
# REUSE Day 2 helpers — Load, Split, Metadata
# ─────────────────────────────────────────────

def load_and_chunk(pdf_configs: list[dict]) -> list:
    """Load PDFs, split into chunks, and attach metadata (Day 2 pipeline)."""
    all_docs = []
    for config in pdf_configs:
        print(f"📄 Loading: {config['path']}")
        loader = PyPDFLoader(config["path"])
        docs   = loader.load()
        for doc in docs:
            doc.metadata["_source_type"] = config["source_type"]
            doc.metadata["_upload_date"]  = config["upload_date"]
        all_docs.extend(docs)
        print(f"   → {len(docs)} page(s) loaded")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks   = splitter.split_documents(all_docs)

    for chunk in chunks:
        src      = chunk.metadata.get("source", "unknown")
        filename = src.split("/")[-1].split("\\")[-1]
        chunk.metadata["filename"]    = filename
        chunk.metadata["page_number"] = chunk.metadata.get("page", 0) + 1
        chunk.metadata["upload_date"] = chunk.metadata.pop("_upload_date", str(date.today()))
        chunk.metadata["source_type"] = chunk.metadata.pop("_source_type", "unknown")

    print(f"\n✅ {len(chunks)} chunks ready\n")
    return chunks


# ─────────────────────────────────────────────
# EXERCISE 1 — Store Chunks in ChromaDB
# ─────────────────────────────────────────────

def build_vector_store(chunks: list) -> Chroma:
    """
    Generate embeddings with nomic-embed-text via Ollama
    and persist them in a ChromaDB vector store.
    """
    print("🔢 Generating embeddings and storing in ChromaDB …")
    print(f"   Model  : {EMBED_MODEL}")
    print(f"   Chunks : {len(chunks)}")
    print(f"   Dir    : {CHROMA_DIR}\n")

    embeddings   = OllamaEmbeddings(model=EMBED_MODEL)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
    )

    print(f"✅ Vector store built — {vector_store._collection.count()} vectors stored\n")
    return vector_store


# ─────────────────────────────────────────────
# EXERCISE 2 — Semantic Retriever
# ─────────────────────────────────────────────

def build_retriever(vector_store: Chroma):
    """Return a retriever that fetches the top-k most similar chunks."""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
    print(f"✅ Retriever ready (top-k = {TOP_K})\n")
    return retriever


# ─────────────────────────────────────────────
# EXERCISE 3 — RAG Chain
# ─────────────────────────────────────────────

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:""",
)


def format_docs(docs: list) -> str:
    """Concatenate retrieved chunks into a single context string."""
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('filename','?')} | "
        f"Page {d.metadata.get('page_number','?')} | "
        f"Type: {d.metadata.get('source_type','?')}]\n{d.page_content}"
        for d in docs
    )


def build_rag_chain(retriever):
    """
    RAG chain:
        question → retriever → prompt (context + question) → LLM → answer
    """
    llm = ChatOllama(model=CHAT_MODEL, temperature=0)

    chain = (
        {
            "context":  retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    print(f"✅ RAG chain ready (LLM: {CHAT_MODEL})\n")
    return chain


# ─────────────────────────────────────────────
# EXERCISE 4 — Test with Multiple Questions
# ─────────────────────────────────────────────

TEST_QUESTIONS = [
    "What is the Transformer model and why was it proposed?",
    "What are the three types of machine learning?",
    "What is the difference between overfitting and underfitting?",
    "How does self-attention work in the Transformer architecture?",
    "What are the steps in a machine learning pipeline?",
]


def run_rag_tests(retriever, chain):
    print("=" * 60)
    print("EXERCISE 4 — RAG Question Answering Tests")
    print("=" * 60)

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n{'─'*60}")
        print(f"Q{i}: {question}")
        print("─" * 60)

        # Show retrieved chunks
        retrieved_docs = retriever.invoke(question)
        print(f"\n📚 Retrieved {len(retrieved_docs)} chunk(s):")
        for j, doc in enumerate(retrieved_docs, 1):
            meta = doc.metadata
            print(f"\n  Chunk {j} — {meta.get('filename','?')} | "
                  f"Page {meta.get('page_number','?')} | "
                  f"Type: {meta.get('source_type','?')}")
            print(f"  Preview: {doc.page_content[:150].strip()!r} …")

        # Generate answer
        print(f"\n🤖 Answer:")
        answer = chain.invoke(question)
        print(answer)

    print(f"\n{'='*60}")
    print("✅ All RAG tests complete!")
    print("=" * 60)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════╗")
    print("║   🔍  Day 3 — RAG Pipeline               ║")
    print("╚══════════════════════════════════════════╝\n")

    # Day 2 pipeline
    chunks = load_and_chunk(PDF_FILES)

    # Exercise 1 — Vector store
    vector_store = build_vector_store(chunks)

    # Exercise 2 — Retriever
    retriever = build_retriever(vector_store)

    # Exercise 3 — RAG chain
    chain = build_rag_chain(retriever)

    # Exercise 4 — Tests
    run_rag_tests(retriever, chain)
