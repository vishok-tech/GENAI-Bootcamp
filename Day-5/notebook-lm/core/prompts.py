# =============================================================================
# core/prompts.py — All LLM prompts for the NotebookLM Replica
# =============================================================================

# ── RAG answer prompt ─────────────────────────────────────────────────────────
RAG_PROMPT = """You are a helpful research assistant. Answer the user's question
using ONLY the provided document context below.

Always cite your sources at the end using the format:
📄 Source: [filename], Page [page_number]

If the context does not contain enough information to answer the question,
say: "I couldn't find relevant information in the selected documents."

Context:
{context}

Question: {question}

Answer:"""

# ── Web search synthesis prompt ───────────────────────────────────────────────
WEB_SEARCH_PROMPT = """You are a helpful research assistant. The user asked a
question and you searched the web for current information.

Synthesize the search results below into a clear, concise answer.
Always mention that this information comes from a web search.

Search Results:
{search_results}

Question: {question}

Answer:"""

# ── Intent classification prompt ──────────────────────────────────────────────
INTENT_PROMPT = """Classify the user's message into one of these intents:
- "document_search": User wants information from uploaded documents
- "web_search": User wants current/recent information from the internet
- "save_note": User wants to save something as a note
- "general": General conversation or greeting

Reply with ONLY the intent label, nothing else.

Message: {message}
Intent:"""

# ── Note summary prompt ───────────────────────────────────────────────────────
NOTE_SUMMARY_PROMPT = """Summarize the following answer into a concise markdown note.
Include a short title (# Title), key points as bullet points, and any source citations.

Answer to summarize:
{answer}

Markdown Note:"""
