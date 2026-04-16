# Day 3 - RAG Pipeline (Vector Store, Retriever & LLM)

## 📌 Overview
This task implements a **Retrieval-Augmented Generation (RAG)** pipeline using LangChain, Ollama, and ChromaDB.

---

## 🚀 Tasks Completed

### 1. Store Chunks in Vector Database
- Generated embeddings using `nomic-embed-text`
- Stored chunks in **ChromaDB**
- Enabled persistent vector storage

### 2. Build Semantic Retriever
- Created a retriever to fetch **top-k relevant chunks**
- Based on semantic similarity (not keyword search)

### 3. Build RAG Chain
- Combined:
  - Retriever
  - Prompt template
  - LLM (`qwen2.5:1.5b`)
- Generated context-aware answers

### 4. Testing with Multiple Questions
- Asked multiple questions
- Printed:
  - Retrieved chunks
  - Final generated answers

---

## 🛠️ Tech Stack
- Python
- LangChain
- Ollama
- ChromaDB
- PyPDFLoader

---

## 📂 File
- `rag.py`

---

## 🧠 Key Learning
Semantic search retrieves information based on **meaning**, not keywords — forming the backbone of modern RAG systems.

---

## ▶️ How to Run

```bash
pip install langchain langchain-community langchain-text-splitters chromadb pypdf langchain-ollama

ollama pull nomic-embed-text
ollama pull qwen2.5:1.5b

python rag.py