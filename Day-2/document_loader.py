"""
Day 2 — Document Loading, Chunking, Metadata & Filtering
GenAI Bootcamp | Nunnari Academy
"""

from datetime import date
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ─────────────────────────────────────────────
# CONFIG — update these paths to your PDF files
# ─────────────────────────────────────────────

PDF_FILES = [
    {
        "path": "paper1.pdf",           # ← replace with your actual file path
        "source_type": "research_paper",
        "upload_date": str(date.today()),
    },
    {
        "path": "textbook_chapter.pdf", # ← replace with your actual file path
        "source_type": "textbook",
        "upload_date": str(date.today()),
    },
]


# ─────────────────────────────────────────────
# EXERCISE 1 — Load PDFs
# ─────────────────────────────────────────────

def load_pdfs(pdf_configs: list[dict]) -> list:
    """Load all PDF files using PyPDFLoader and return raw documents."""
    all_docs = []
    for config in pdf_configs:
        print(f"📄 Loading: {config['path']}")
        loader = PyPDFLoader(config["path"])
        docs = loader.load()
        print(f"   → {len(docs)} page(s) loaded")

        # Temporarily store extra config on each doc for use in Exercise 3
        for doc in docs:
            doc.metadata["_source_type"] = config["source_type"]
            doc.metadata["_upload_date"] = config["upload_date"]

        all_docs.extend(docs)

    print(f"\n✅ Total pages loaded: {len(all_docs)}\n")
    return all_docs


# ─────────────────────────────────────────────
# EXERCISE 2 — Split into Chunks
# ─────────────────────────────────────────────

def split_documents(docs: list) -> list:
    """Split documents into chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Total chunks after splitting: {len(chunks)}\n")
    return chunks


# ─────────────────────────────────────────────
# EXERCISE 3 — Attach Metadata
# ─────────────────────────────────────────────

def attach_metadata(chunks: list) -> list:
    """
    Enrich each chunk's metadata with:
    - filename     : name of the source PDF
    - page_number  : page the chunk came from (1-indexed)
    - upload_date  : date the PDF was loaded
    - source_type  : e.g. 'textbook', 'research_paper', 'notes'
    """
    for chunk in chunks:
        source_path = chunk.metadata.get("source", "unknown")
        filename = source_path.split("/")[-1].split("\\")[-1]  # works on Win & Mac/Linux

        chunk.metadata["filename"]    = filename
        chunk.metadata["page_number"] = chunk.metadata.get("page", 0) + 1  # LangChain pages are 0-indexed
        chunk.metadata["upload_date"] = chunk.metadata.pop("_upload_date", str(date.today()))
        chunk.metadata["source_type"] = chunk.metadata.pop("_source_type", "unknown")

    print(f"✅ Metadata attached to {len(chunks)} chunks\n")
    return chunks


# ─────────────────────────────────────────────
# EXERCISE 4 — Filter Function
# ─────────────────────────────────────────────

def filter_chunks(chunks: list, **filters) -> list:
    """
    Return only the chunks whose metadata matches ALL provided key-value filters.

    Example usage:
        filter_chunks(chunks, filename="paper1.pdf")
        filter_chunks(chunks, filename="paper1.pdf", page_number=3)
        filter_chunks(chunks, source_type="textbook")
    """
    results = []
    for chunk in chunks:
        if all(chunk.metadata.get(key) == value for key, value in filters.items()):
            results.append(chunk)
    return results


# ─────────────────────────────────────────────
# EXERCISE 5 — Test Block
# ─────────────────────────────────────────────

def run_tests(chunks: list):
    print("=" * 55)
    print("EXERCISE 5 — Filter Tests")
    print("=" * 55)

    # Grab the actual filenames present in chunks (works with any PDF names)
    filenames = list({c.metadata["filename"] for c in chunks})
    source_types = list({c.metadata["source_type"] for c in chunks})

    # ── Test 1: Filter by filename ──────────────────────────
    print(f"\n📌 Test 1: Filter by filename = '{filenames[0]}'")
    result = filter_chunks(chunks, filename=filenames[0])
    print(f"   Chunks found: {len(result)}")
    if result:
        print(f"   Sample chunk (first 120 chars): {result[0].page_content[:120]!r}")

    # ── Test 2: Filter by filename + page_number ────────────
    print(f"\n📌 Test 2: Filter by filename = '{filenames[0]}' AND page_number = 1")
    result = filter_chunks(chunks, filename=filenames[0], page_number=1)
    print(f"   Chunks found: {len(result)}")
    if result:
        print(f"   Sample chunk (first 120 chars): {result[0].page_content[:120]!r}")

    # ── Test 3: Filter by source_type ───────────────────────
    print(f"\n📌 Test 3: Filter by source_type = '{source_types[0]}'")
    result = filter_chunks(chunks, source_type=source_types[0])
    print(f"   Chunks found: {len(result)}")

    # ── Test 4: Filter that returns no results ───────────────
    print(f"\n📌 Test 4: Filter with a page_number that doesn't exist (page_number=9999)")
    result = filter_chunks(chunks, page_number=9999)
    print(f"   Chunks found: {len(result)}  ← expected 0")

    # ── Test 5: Show metadata of first chunk overall ─────────
    print(f"\n📌 Test 5: Metadata snapshot of first chunk overall")
    first = chunks[0]
    for key, value in first.metadata.items():
        if not key.startswith("_"):
            print(f"   {key}: {value}")

    print("\n" + "=" * 55)
    print("✅ All tests complete!")
    print("=" * 55)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1 — Load
    raw_docs = load_pdfs(PDF_FILES)

    # Step 2 — Split
    chunks = split_documents(raw_docs)

    # Step 3 — Attach metadata
    chunks = attach_metadata(chunks)

    # Step 4 & 5 — Filter function is defined above; tests run here
    run_tests(chunks)
