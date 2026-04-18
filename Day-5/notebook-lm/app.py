# =============================================================================
# app.py — NotebookLM Replica — Main Streamlit Application
# =============================================================================
# A simplified NotebookLM clone built with:
#   Frontend    : Streamlit (3-panel layout)
#   LLM         : Ollama llama3.2:3b (local, free)
#   Framework   : LangChain
#   Vector DB   : ChromaDB (persistent, local)
#   Embeddings  : Ollama nomic-embed-text
#   Web Search  : Tavily API
#   Orchestration: LangGraph (classify → retrieve/search → generate → save)
#
# Run: streamlit run app.py
# =============================================================================

import streamlit as st

# ── Page config — must be the FIRST Streamlit call ───────────────────────────
st.set_page_config(
    page_title  = "NotebookLM Replica",
    page_icon   = "📓",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global font & background ───────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Hide default Streamlit chrome ─────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Top bar ────────────────────────────────────────────────────────────── */
.top-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 0 1rem;
    border-bottom: 1px solid rgba(128,128,128,0.15);
    margin-bottom: 1rem;
}
.top-bar h1 {
    margin: 0;
    font-size: 1.4rem;
    font-weight: 600;
    letter-spacing: -0.02em;
}
.top-bar .subtitle {
    font-size: 0.8rem;
    opacity: 0.55;
    margin: 0;
}

/* ── Notes panel header ─────────────────────────────────────────────────── */
.notes-header {
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    opacity: 0.6;
    margin-bottom: 0.5rem;
}

/* ── Chat message styling ───────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 0.25rem;
}

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,0.12);
}

/* ── Expander cards ─────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid rgba(128,128,128,0.15);
    border-radius: 10px;
    margin-bottom: 0.5rem;
}

/* ── Dividers ───────────────────────────────────────────────────────────── */
hr { opacity: 0.15 !important; }

/* ── Code blocks (for Mermaid) ──────────────────────────────────────────── */
code { font-family: 'DM Mono', monospace; font-size: 0.8rem; }

/* ── Scrollable notes panel ─────────────────────────────────────────────── */
.notes-scroll {
    max-height: calc(100vh - 200px);
    overflow-y: auto;
    padding-right: 4px;
}
</style>
""", unsafe_allow_html=True)


# ── Imports ───────────────────────────────────────────────────────────────────
from components.sidebar import render_sidebar
from components.chat    import render_chat
from components.notes   import render_notes_panel


# ── Layout: Sidebar + Main (Chat + Notes) ─────────────────────────────────────
# Sidebar is rendered by Streamlit's native sidebar.
# Main area is split: 65% chat | 35% notes

def main():
    # Render sidebar and get settings
    settings = render_sidebar()

    # Main area — split into chat and notes columns
    chat_col, notes_col = st.columns([65, 35], gap="large")

    with chat_col:
        # Top bar
        st.markdown("""
        <div class="top-bar">
            <span style="font-size:1.6rem">📓</span>
            <div>
                <h1>NotebookLM Replica</h1>
                <p class="subtitle">Ollama · LangChain · ChromaDB · LangGraph</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Chat area
        render_chat(
            selected_docs = settings["selected_docs"],
            web_search_on = settings["web_search_on"],
        )

    with notes_col:
        st.markdown(
            "<div class='notes-header'>Saved Notes</div>",
            unsafe_allow_html=True,
        )
        render_notes_panel()


if __name__ == "__main__":
    main()
