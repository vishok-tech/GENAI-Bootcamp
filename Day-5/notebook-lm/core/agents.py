# =============================================================================
# core/agents.py — LangChain ReAct agent with Document, Web, and Note tools
# =============================================================================

from datetime import datetime
from typing import List, Optional
from langchain_core.tools import tool          # ← use langchain_core not langchain
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

from config import LLM_MODEL, OLLAMA_BASE_URL, NOTES_DIR, TAVILY_API_KEY
from core.rag_chain import rag_answer, get_llm
from core.prompts import NOTE_SUMMARY_PROMPT

# Agent imports at the bottom to avoid circular issues
try:
    from langchain.agents import AgentExecutor
    from langchain.agents.react.agent import create_react_agent
    from langchain import hub
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False


# ── Tool 1: Document Search ───────────────────────────────────────────────────

def make_document_search_tool(selected_docs: Optional[List[str]] = None):
    """
    Factory that creates a document search tool bound to the selected documents.
    The tool is recreated each time so it uses the current document selection.
    """
    @tool
    def document_search(query: str) -> str:
        """
        Search through the uploaded PDF documents for information.
        Use this when the user asks about content in their documents.
        Input: a search query string.
        """
        result = rag_answer(query, selected_docs=selected_docs)
        answer = result["answer"]
        citations = result["citations"]
        if citations:
            return f"{answer}\n\nSources:\n{citations}"
        return answer

    return document_search


# ── Tool 2: Web Search ──────────────────────────────────────
# 
# ──────────────────

def make_web_search_tool():
    """Create a Tavily web search tool."""
    try:
        # NEW
        from langchain_tavily import TavilySearch
        import os
        os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

        tool = TavilySearch(
            max_results        = 4,
            search_depth       = "basic",
            include_answer     = True,
            include_raw_content= False,
        )

        @tool
        def web_search(query: str) -> str:
            """
            Search the web for current, real-time information.
            Use this when the user asks about recent events or needs
            information not found in the uploaded documents.
            Input: a search query string.
            """
            results = tavily_tool.invoke({"query": query})
            if isinstance(results, list):
                formatted = []
                for r in results:
                    if isinstance(r, dict):
                        title   = r.get("title", "")
                        content = r.get("content", "")
                        url     = r.get("url", "")
                        formatted.append(f"**{title}**\n{content}\nURL: {url}")
                return "\n\n---\n\n".join(formatted)
            return str(results)

        return web_search

    except Exception as e:
        @tool
        def web_search(query: str) -> str:
            """Web search tool (currently unavailable — Tavily key not set)."""
            return f"Web search unavailable: {str(e)}. Please add TAVILY_API_KEY to .env"
        return web_search


# ── Tool 3: Save Note ─────────────────────────────────────────────────────────

@tool
def save_note(content: str) -> str:
    """
    Save a response or piece of information as a markdown note file.
    Use this when the user asks to save or remember something.
    Input: the text content to save as a note.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"note_{timestamp}.md"
    filepath  = NOTES_DIR / filename

    # Generate a summarized markdown note using LLM
    try:
        llm    = get_llm()
        prompt = PromptTemplate(
            input_variables=["answer"],
            template=NOTE_SUMMARY_PROMPT,
        )
        chain   = prompt | llm
        md_note = chain.invoke({"answer": content})
        md_note = str(md_note)
    except Exception:
        # Fallback: save raw content if LLM fails
        md_note = f"# Note\n\n{content}"

    # Add metadata header
    full_note = f"---\nSaved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n---\n\n{md_note}"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(full_note)

    return f"✅ Note saved as `{filename}`"


# ── Agent Builder ─────────────────────────────────────────────────────────────

def build_agent(
    selected_docs   : Optional[List[str]] = None,
    web_search_on   : bool = False,
) -> AgentExecutor:
    """
    Build a ReAct agent with the appropriate tools.

    Tools included:
      - document_search (always)
      - web_search (only if web_search_on=True)
      - save_note (always)
    """
    tools = [
        make_document_search_tool(selected_docs),
        save_note,
    ]

    if web_search_on and TAVILY_API_KEY:
        tools.append(make_web_search_tool())

    llm = OllamaLLM(
        model       = LLM_MODEL,
        base_url    = OLLAMA_BASE_URL,
        temperature = 0.1,
    )

    # ReAct prompt from LangChain hub (works well with llama3.2:3b)
    try:
        prompt = hub.pull("hwchase17/react")
    except Exception:
        # Fallback ReAct prompt if hub is unavailable
        from langchain.prompts import PromptTemplate
        prompt = PromptTemplate.from_template(
            "Answer the following questions as best you can. You have access to the following tools:\n\n"
            "{tools}\n\n"
            "Use the following format:\n\n"
            "Question: the input question you must answer\n"
            "Thought: you should always think about what to do\n"
            "Action: the action to take, should be one of [{tool_names}]\n"
            "Action Input: the input to the action\n"
            "Observation: the result of the action\n"
            "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
            "Thought: I now know the final answer\n"
            "Final Answer: the final answer to the original input question\n\n"
            "Begin!\n\n"
            "Question: {input}\n"
            "Thought:{agent_scratchpad}"
        )

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent          = agent,
        tools          = tools,
        verbose        = False,
        max_iterations = 5,
        handle_parsing_errors = True,
        return_intermediate_steps = False,
    )
