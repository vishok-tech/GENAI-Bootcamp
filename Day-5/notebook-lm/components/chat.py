# =============================================================================
# components/chat.py — Chat area: conversation interface with RAG + Agent
# =============================================================================

import streamlit as st
from typing import List, Optional

from core.graph import build_graph


def initialize_chat():
    """Initialize chat session state on first load."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role"   : "assistant",
                "content": (
                    "👋 Hello! I'm your NotebookLM assistant.\n\n"
                    "**To get started:**\n"
                    "1. Upload one or more PDF files in the sidebar\n"
                    "2. Select the documents you want to search\n"
                    "3. Ask me anything about your documents!\n\n"
                    "You can also enable **Web Search** in the sidebar "
                    "to search the internet for additional context."
                ),
            }
        ]


def render_chat(
    selected_docs : Optional[List[str]] = None,
    web_search_on : bool = False,
):
    """
    Render the chat interface.
    Handles message display, user input, and agent invocation.
    """
    initialize_chat()

    # ── Display chat history ──────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show sources if available
            if msg.get("citations"):
                with st.expander("📎 Sources", expanded=False):
                    st.markdown(msg["citations"])

            # Show save note button for assistant messages
            if msg["role"] == "assistant" and msg.get("content"):
                col1, col2, col3 = st.columns([1, 1, 8])
                with col1:
                    if st.button(
                        "💾 Save",
                        key     = f"save_{hash(msg['content'][:50])}",
                        help    = "Save this response as a note",
                        type    = "secondary",
                    ):
                        _save_note_inline(msg["content"])

    # ── Chat input ────────────────────────────────────────────────────────────
    user_input = st.chat_input(
        placeholder = "Ask about your documents... (e.g. 'What is the main argument?')",
    )

    if user_input:
        _handle_user_message(
            user_input    = user_input,
            selected_docs = selected_docs,
            web_search_on = web_search_on,
        )


def _handle_user_message(
    user_input    : str,
    selected_docs : Optional[List[str]],
    web_search_on : bool,
):
    """Process user input through the LangGraph pipeline and display response."""

    # Add user message to history
    st.session_state.messages.append({
        "role"   : "user",
        "content": user_input,
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Run LangGraph pipeline
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                graph = build_graph()

                # Initial state for the graph
                initial_state = {
                    "question"      : user_input,
                    "intent"        : "",
                    "context"       : "",
                    "search_results": "",
                    "answer"        : "",
                    "citations"     : "",
                    "note_saved"    : False,
                    "selected_docs" : selected_docs if selected_docs else None,
                    "web_search_on" : web_search_on,
                    "error"         : None,
                }

                result    = graph.invoke(initial_state)
                answer    = result.get("answer", "I couldn't generate a response.")
                citations = result.get("citations", "")
                intent    = result.get("intent", "")

                # Add intent badge
                intent_badge = {
                    "document_search": "📄 Document Search",
                    "web_search"     : "🌐 Web Search",
                    "save_note"      : "💾 Note Saved",
                    "general"        : "💬 General",
                }.get(intent, "")

                if intent_badge:
                    st.caption(intent_badge)

                st.markdown(answer)

                if citations:
                    with st.expander("📎 Sources", expanded=False):
                        st.markdown(citations)

                # Save note button
                col1, col2, col3 = st.columns([1, 1, 8])
                with col1:
                    if st.button(
                        "💾 Save",
                        key  = f"save_new_{hash(answer[:50])}",
                        help = "Save this response as a note",
                        type = "secondary",
                    ):
                        _save_note_inline(answer)

            except Exception as e:
                answer    = f"❌ An error occurred: {str(e)}\n\nPlease check that Ollama is running."
                citations = ""
                st.error(answer)

    # Add assistant response to history
    st.session_state.messages.append({
        "role"     : "assistant",
        "content"  : answer,
        "citations": citations,
    })


def _save_note_inline(content: str):
    """Save a note directly from the chat without going through the graph."""
    from core.agents import save_note
    try:
        result = save_note.invoke(content)
        st.toast(result, icon="💾")
        st.rerun()
    except Exception as e:
        st.toast(f"Failed to save note: {e}", icon="❌")
