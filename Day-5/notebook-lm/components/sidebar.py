# =============================================================================
# components/sidebar.py — Left sidebar: PDF upload, document list, settings
# =============================================================================

import streamlit as st
from pathlib import Path

from config import UPLOADS_DIR
from core.document_processor import save_uploaded_file, load_and_chunk_pdf
from core.vector_store import add_documents, get_stored_filenames, delete_document
from utils.helpers import format_file_size


def render_sidebar() -> dict:
    """
    Render the sidebar and return current settings as a dict:
      {
        "selected_docs" : List[str],
        "web_search_on" : bool,
      }
    """
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 1rem 0 0.5rem'>
            <span style='font-size:2rem'>📓</span>
            <h2 style='margin:0; font-size:1.3rem; font-weight:700;
                       color: var(--text-color)'>NotebookLM</h2>
            <p style='margin:0; font-size:0.75rem; opacity:0.6'>
                Powered by Ollama + LangGraph
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── PDF Upload Section ────────────────────────────────────────────────
        st.markdown("### 📄 Upload Documents")
        uploaded_files = st.file_uploader(
            label      = "Upload PDF files",
            type       = ["pdf"],
            accept_multiple_files = True,
            label_visibility      = "collapsed",
            help = "Upload one or more PDF files to chat with",
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Check if already indexed
                stored = get_stored_filenames()
                if uploaded_file.name in stored:
                    st.info(f"✓ {uploaded_file.name} already indexed")
                    continue

                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        # Save to disk
                        file_path = save_uploaded_file(uploaded_file)

                        # Chunk and index
                        chunks    = load_and_chunk_pdf(file_path)
                        n_added   = add_documents(chunks)

                        st.success(
                            f"✅ {uploaded_file.name}\n"
                            f"→ {n_added} chunks indexed"
                        )
                    except Exception as e:
                        st.error(f"❌ Failed to process {uploaded_file.name}: {str(e)}")

        st.divider()

        # ── Document List with Checkboxes ─────────────────────────────────────
        st.markdown("### 📚 Your Documents")
        stored_docs = get_stored_filenames()

        if not stored_docs:
            st.caption("No documents uploaded yet.\nUpload a PDF above to get started.")
            selected_docs = []
        else:
            # Initialize selection state
            if "selected_docs" not in st.session_state:
                st.session_state.selected_docs = stored_docs.copy()

            selected_docs = []
            for doc in stored_docs:
                # Get file size if available
                file_path = UPLOADS_DIR / doc
                size_str  = format_file_size(file_path.stat().st_size) \
                            if file_path.exists() else ""

                col1, col2 = st.columns([5, 1])
                with col1:
                    checked = st.checkbox(
                        label = f"**{doc}**" + (f"\n{size_str}" if size_str else ""),
                        value = doc in st.session_state.get("selected_docs", stored_docs),
                        key   = f"doc_{doc}",
                    )
                with col2:
                    if st.button("🗑", key=f"del_{doc}",
                                  help=f"Remove {doc} from index"):
                        with st.spinner("Removing..."):
                            delete_document(doc)
                            # Remove from local uploads too
                            if file_path.exists():
                                file_path.unlink()
                        st.rerun()

                if checked:
                    selected_docs.append(doc)

            if not selected_docs:
                st.warning("⚠️ No documents selected. Select at least one to search.")

        st.divider()

        # ── Chat Settings ─────────────────────────────────────────────────────
        st.markdown("### ⚙️ Settings")

        web_search_on = st.toggle(
            label = "🌐 Web Search",
            value = False,
            help  = "Enable Tavily web search for real-time information",
        )

        if web_search_on:
            from config import TAVILY_API_KEY
            if not TAVILY_API_KEY:
                st.warning("Add TAVILY_API_KEY to .env to use web search")
            else:
                st.caption("✓ Tavily API connected")

        st.divider()

        # ── Clear Chat Button ─────────────────────────────────────────────────
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages  = []
            st.session_state.chat_history = []
            st.rerun()

        # ── LangGraph Diagram ─────────────────────────────────────────────────
        with st.expander("🔀 View LangGraph"):
            try:
                from core.graph import get_graph_mermaid
                mermaid_str = get_graph_mermaid()
                st.code(mermaid_str, language="text")
                st.caption("Copy to https://mermaid.live to visualize")
            except Exception as e:
                st.caption(f"Graph unavailable: {e}")

    return {
        "selected_docs": selected_docs,
        "web_search_on": web_search_on,
    }
