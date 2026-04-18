# =============================================================================
# core/graph.py — LangGraph workflow for NotebookLM Replica
# =============================================================================
# Nodes:
#   classify_intent   → decide what kind of request this is
#   retrieve_documents → RAG search in ChromaDB
#   web_search        → Tavily search
#   generate_response → LLM answer generation
#   save_note         → save markdown note
#
# Conditional routing:
#   classify_intent → {retrieve_documents | web_search | save_note | generate_response}
#   retrieve_documents → generate_response
#   web_search → generate_response
#   generate_response → END
#   save_note → END
# =============================================================================

from typing import TypedDict, Optional, List
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

from config import LLM_MODEL, OLLAMA_BASE_URL, TAVILY_API_KEY
from core.prompts import INTENT_PROMPT, RAG_PROMPT, WEB_SEARCH_PROMPT
from core.vector_store import similarity_search
from core.rag_chain import format_context, format_citations, get_llm


# ── Graph State ───────────────────────────────────────────────────────────────

class GraphState(TypedDict):
    """
    State passed between LangGraph nodes.
    Each field is updated by the node that processes it.
    """
    question        : str
    intent          : str
    context         : str
    search_results  : str
    answer          : str
    citations       : str
    note_saved      : bool
    selected_docs   : Optional[List[str]]
    web_search_on   : bool
    error           : Optional[str]


# ── Node 1: Classify Intent ───────────────────────────────────────────────────

def classify_intent(state: GraphState) -> GraphState:
    """
    Use the LLM to classify the user's intent.
    Possible intents: document_search, web_search, save_note, general
    """
    llm    = get_llm()
    prompt = PromptTemplate(
        input_variables=["message"],
        template=INTENT_PROMPT,
    )
    chain  = prompt | llm

    try:
        raw_intent = chain.invoke({"message": state["question"]})
        intent     = str(raw_intent).strip().lower()

        # Normalize intent to known values
        if "web" in intent:
            intent = "web_search"
        elif "save" in intent or "note" in intent:
            intent = "save_note"
        elif "document" in intent or "search" in intent or "pdf" in intent:
            intent = "document_search"
        else:
            intent = "general"

        # Override: if web search is disabled, fall back to document search
        if intent == "web_search" and not state.get("web_search_on", False):
            intent = "document_search"

    except Exception as e:
        intent = "document_search"  # safe default

    return {**state, "intent": intent}


# ── Node 2: Retrieve Documents ────────────────────────────────────────────────

def retrieve_documents(state: GraphState) -> GraphState:
    """
    Retrieve relevant chunks from ChromaDB filtered by selected documents.
    """
    try:
        docs      = similarity_search(
            query         = state["question"],
            selected_docs = state.get("selected_docs"),
            k             = 4,
        )
        context   = format_context(docs)
        citations = format_citations(docs)
    except Exception as e:
        context   = ""
        citations = ""

    return {**state, "context": context, "citations": citations}


# ── Node 3: Web Search ────────────────────────────────────────────────────────

def web_search(state: GraphState) -> GraphState:
    """
    Run Tavily web search and store results in state.
    """
    try:
        # NEW
        from langchain_tavily import TavilySearch
        import os
        os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

        tool = TavilySearch(max_results=4)
        results = tool.invoke({"query": state["question"]})

        if isinstance(results, list):
            formatted = []
            for r in results:
                if isinstance(r, dict):
                    title   = r.get("title", "")
                    content = r.get("content", "")
                    formatted.append(f"**{title}**\n{content}")
            search_results = "\n\n---\n\n".join(formatted)
        else:
            search_results = str(results)

    except Exception as e:
        search_results = f"Web search failed: {str(e)}"

    return {**state, "search_results": search_results}


# ── Node 4: Generate Response ─────────────────────────────────────────────────

def generate_response(state: GraphState) -> GraphState:
    """
    Generate the final answer using the LLM.
    Uses document context if available, web results if from web search,
    or falls back to general knowledge.
    """
    llm = get_llm()

    try:
        if state.get("search_results"):
            # Web search path
            prompt = PromptTemplate(
                input_variables=["search_results", "question"],
                template=WEB_SEARCH_PROMPT,
            )
            answer = (prompt | llm).invoke({
                "search_results": state["search_results"],
                "question"      : state["question"],
            })
        elif state.get("context"):
            # RAG path
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=RAG_PROMPT,
            )
            answer = (prompt | llm).invoke({
                "context" : state["context"],
                "question": state["question"],
            })
        else:
            # General fallback
            answer = llm.invoke(state["question"])

        answer = str(answer)

    except Exception as e:
        answer = f"I encountered an error generating a response: {str(e)}"

    return {**state, "answer": answer}


# ── Node 5: Save Note ─────────────────────────────────────────────────────────

def save_note_node(state: GraphState) -> GraphState:
    """
    Save the current answer or question as a note.
    """
    content = state.get("answer") or state.get("question", "")
    if not content:
        return {**state, "note_saved": False,
                "answer": "Nothing to save — no content available."}

    result = save_note.invoke(content)
    return {**state, "note_saved": True, "answer": result}


# ── Conditional Router ────────────────────────────────────────────────────────

def route_intent(state: GraphState) -> str:
    """
    Decide which node to go to after classify_intent.
    Returns the name of the next node.
    """
    intent = state.get("intent", "general")
    if intent == "save_note":
        return "save_note"
    elif intent == "web_search" and state.get("web_search_on", False):
        return "web_search"
    else:
        # Default: document search (works for general too)
        return "retrieve_documents"


# ── Build Graph ───────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Assemble and compile the LangGraph workflow.
    Returns a compiled graph ready to invoke.
    """
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("classify_intent",    classify_intent)
    graph.add_node("retrieve_documents", retrieve_documents)
    graph.add_node("web_search",         web_search)
    graph.add_node("generate_response",  generate_response)
    graph.add_node("save_note",          save_note_node)

    # Entry point
    graph.set_entry_point("classify_intent")

    # Conditional routing after intent classification
    graph.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "retrieve_documents": "retrieve_documents",
            "web_search"        : "web_search",
            "save_note"         : "save_note",
        }
    )

    # After retrieval → generate
    graph.add_edge("retrieve_documents", "generate_response")

    # After web search → generate
    graph.add_edge("web_search", "generate_response")

    # Both paths end after generation
    graph.add_edge("generate_response", END)
    graph.add_edge("save_note",         END)

    return graph.compile()


def get_graph_mermaid() -> str:
    """Return the Mermaid diagram string for the LangGraph workflow."""
    try:
        compiled = build_graph()
        return compiled.get_graph().draw_mermaid()
    except Exception as e:
        return f"graph TD\n    A[classify_intent] --> B{{route}}\n    B -->|document| C[retrieve_documents]\n    B -->|web| D[web_search]\n    B -->|note| E[save_note]\n    C --> F[generate_response]\n    D --> F\n    F --> G([END])\n    E --> G"
