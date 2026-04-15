# Day 2 - Document Loading, Chunking & Metadata Filtering

## Overview
This Task focuses on processing PDF documents using LangChain by loading, splitting, enriching with metadata, and filtering chunks based on specific conditions.

---

## 🚀 Tasks Completed

### 1. Load PDFs
Used `PyPDFLoader` to load multiple PDF files such as research papers and textbook chapters.

### 2. Split into Chunks
Applied `RecursiveCharacterTextSplitter` with:
- `chunk_size = 1000`
- `chunk_overlap = 200`

This ensures efficient handling of large documents for downstream tasks.

### 3. Attach Metadata
Each chunk is enriched with:
- `filename`
- `page_number`
- `upload_date`
- `source_type` (e.g., textbook, research_paper)

### 4. Filter Function
Implemented a flexible filtering function:


5. Testing

Performed multiple test cases to validate:

Filtering by filename
Filtering by page number
Filtering by source type
Handling edge cases (no results)

### Tech Stack

Python
LangChain
PyPDFLoader
RecursiveCharacterTextSplitter

How to Run

pip install langchain langchain-community

python document_loader.py

```python






