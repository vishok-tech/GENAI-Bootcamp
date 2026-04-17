"""
Day 4 — Tool Calling, Manual Agent Loop & ReAct Agent
GenAI Bootcamp | Nunnari Academy

Prerequisites:
    pip install langchain langchain-community langchain-ollama
                langgraph tavily-python

Tavily API key (free at tavily.com):
    Set environment variable: TAVILY_API_KEY=tvly-xxxx

Ollama model needed:
    ollama pull llama3.2:3b
"""

import json
import os
from datetime import datetime

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
load_dotenv()

# ── Tavily needs an API key ────────────────────────────────────────────────────
# Either set it in your shell:  $env:TAVILY_API_KEY="tvly-xxxx"
# Or hard-code it here (not recommended for production):
# os.environ["TAVILY_API_KEY"] = "tvly-xxxx"

CHAT_MODEL = "llama3.2:3b"
NOTES_FILE = "notes.json"


# ─────────────────────────────────────────────
# EXERCISE 1 — Define Three Tools
# ─────────────────────────────────────────────

# ── Tool 1: Web Search via Tavily ─────────────────────────────────────────────
@tool
def web_search(query: str) -> str:
    """Search the web for real-time information on any topic using Tavily."""
    print(f"  🔍 [web_search] Searching: '{query}'")
    search = TavilySearch(max_results=3)
    results = search.invoke(query)
    return str(results)


# ── Tool 2: Summarize via ChatOllama ──────────────────────────────────────────
@tool
def summarize(text: str) -> str:
    """Summarize a given piece of text into a short paragraph using an LLM."""
    print(f"  📝 [summarize] Summarizing {len(text)} characters …")
    llm = ChatOllama(model=CHAT_MODEL, temperature=0)
    prompt = (
        "Please summarize the following text in one short paragraph "
        "(3-5 sentences). Be concise and capture the key points.\n\n"
        f"Text:\n{text}"
    )
    response = llm.invoke(prompt)
    return response.content


# ── Tool 3: Save Notes ────────────────────────────────────────────────────────
@tool
def notes(text: str) -> str:
    """
    Convert the latest AI response into a saved note with a title and content.
    Saves to a local notes.json file and returns a confirmation.
    """
    print(f"  🗒️  [notes] Saving note …")
    llm = ChatOllama(model=CHAT_MODEL, temperature=0)

    # Ask the LLM to extract a title + content from the text
    extract_prompt = (
        "Extract a short title (max 8 words) and clean content from the text below. "
        "Respond ONLY with valid JSON in this exact format, no extra text:\n"
        '{"title": "...", "content": "..."}\n\n'
        f"Text:\n{text}"
    )
    raw = llm.invoke(extract_prompt).content.strip()

    try:
        note_data = json.loads(raw)
    except json.JSONDecodeError:
        note_data = {"title": "Untitled Note", "content": text}

    note_data["timestamp"] = datetime.now().isoformat()

    # Load existing notes, append, save
    existing = []
    if os.path.exists(NOTES_FILE):
        with open(NOTES_FILE, "r") as f:
            existing = json.load(f)
    existing.append(note_data)
    with open(NOTES_FILE, "w") as f:
        json.dump(existing, f, indent=2)

    return f"✅ Note saved — Title: '{note_data['title']}'"


# ── Tool registry (used by the manual loop) ───────────────────────────────────
TOOLS = {
    "web_search": web_search,
    "summarize":  summarize,
    "notes":      notes,
}


# ── Standalone tool tests ──────────────────────────────────────────────────────
def test_tools_standalone():
    print("\n" + "="*60)
    print("EXERCISE 1 — Standalone Tool Tests")
    print("="*60)

    print("\n📌 Testing web_search …")
    r = web_search.invoke("latest news on LangChain framework")
    print(r[:300], "…")

    print("\n📌 Testing summarize …")
    sample = (
        "Large language models (LLMs) are trained on vast corpora of text data "
        "using self-supervised learning. They learn to predict the next token in "
        "a sequence, which allows them to generate coherent text, answer questions, "
        "write code, and perform many other tasks. Models like GPT-4 and Claude "
        "have billions of parameters and require enormous compute to train."
    )
    r = summarize.invoke(sample)
    print(r)

    print("\n📌 Testing notes …")
    r = notes.invoke("The Transformer architecture introduced self-attention, which allows "
                     "models to weigh the importance of different words in a sequence when "
                     "making predictions. This replaced recurrent networks and enabled "
                     "massive parallelization during training.")
    print(r)


# ─────────────────────────────────────────────
# EXERCISE 2 — Manual Tool-Calling Loop
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful AI assistant with access to these tools:

1. web_search(query: str) — Search the web for real-time information.
2. summarize(text: str) — Summarize a long piece of text into a short paragraph.
3. notes(text: str) — Save the latest response as a titled note to a file.

To call a tool, respond ONLY with valid JSON in this exact format:
{"tool": "tool_name", "args": {"param": "value"}}

When you have a complete final answer and do NOT need any more tools, respond ONLY with:
{"final_answer": "your answer here"}

Rules:
- Always use a tool if the user needs real-time info or summarization.
- Never mix tool calls with plain text — respond with ONLY the JSON.
- After receiving a tool result, decide if you need another tool or can give the final answer.
"""


def run_manual_agent(user_query: str) -> str:
    """
    Manual agent loop:
    1. LLM decides which tool to call (JSON output)
    2. We execute the tool
    3. Feed result back to LLM
    4. Repeat until LLM returns {"final_answer": "..."}
    """
    print(f"\n{'─'*60}")
    print(f"🧠 Query: {user_query}")
    print(f"{'─'*60}")

    llm = ChatOllama(model=CHAT_MODEL, temperature=0)

    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_query},
    ]

    max_iterations = 6

    for iteration in range(max_iterations):
        print(f"\n  [Loop iteration {iteration + 1}]")

        response = llm.invoke(messages)
        raw = response.content.strip()
        print(f"  LLM raw output: {raw[:200]}")

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # LLM gave plain text — treat as final answer
            print("  ⚠️  Non-JSON response — treating as final answer")
            return raw

        # Final answer
        if "final_answer" in parsed:
            print(f"\n✅ Final Answer: {parsed['final_answer']}")
            return parsed["final_answer"]

        # Tool call
        if "tool" in parsed:
            tool_name = parsed.get("tool")
            tool_args = parsed.get("args", {})

            print(f"  🔧 Tool picked: {tool_name}  |  Args: {tool_args}")

            if tool_name not in TOOLS:
                tool_result = f"Error: unknown tool '{tool_name}'"
            else:
                try:
                    tool_result = TOOLS[tool_name].invoke(tool_args)
                except Exception as e:
                    tool_result = f"Tool error: {e}"

            print(f"  📤 Tool result (first 300 chars): {str(tool_result)[:300]}")

            # Feed result back into the conversation
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"Tool result for {tool_name}:\n{tool_result}\n\nNow continue."
            })
        else:
            # Unexpected JSON shape
            return f"Unexpected response format: {raw}"

    return "Max iterations reached without a final answer."


# ─────────────────────────────────────────────
# EXERCISE 3 — Test with Three Queries
# ─────────────────────────────────────────────

TEST_QUERIES = [
    "What is the latest news on OpenAI?",
    (
        "Summarize this paragraph: "
        "Retrieval-Augmented Generation (RAG) is an AI framework that combines "
        "a retrieval system with a generative language model. Instead of relying "
        "solely on the model's parametric knowledge, RAG fetches relevant documents "
        "from an external knowledge base and uses them as context when generating "
        "a response. This makes the system more accurate, up-to-date, and less prone "
        "to hallucination."
    ),
    "Find the latest news on AI agents and summarize it",
]


def run_agent_tests():
    print("\n" + "="*60)
    print("EXERCISE 3 — Manual Agent Loop: Three Query Tests")
    print("="*60)

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n\n{'='*60}")
        print(f"Query {i}: {query[:80]}{'…' if len(query) > 80 else ''}")
        answer = run_manual_agent(query)
        print(f"\n🎯 Answer: {answer}")


# ─────────────────────────────────────────────
# EXERCISE 4 (Bonus) — ReAct Agent with LangGraph
# ─────────────────────────────────────────────

def run_react_agent_tests():
    """Recreate the same behavior using LangGraph's create_react_agent."""
    try:
        from langgraph.prebuilt import create_react_agent
    except ImportError:
        print("\n⚠️  langgraph not installed. Run: pip install langgraph")
        return

    print("\n" + "="*60)
    print("EXERCISE 4 (Bonus) — ReAct Agent via LangGraph")
    print("="*60)

    llm    = ChatOllama(model=CHAT_MODEL, temperature=0)
    tools  = [web_search, summarize, notes]
    agent  = create_react_agent(llm, tools)

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n{'─'*60}")
        print(f"ReAct Query {i}: {query[:80]}{'…' if len(query) > 80 else ''}")
        result = agent.invoke({"messages": [("user", query)]})
        final  = result["messages"][-1].content
        print(f"🎯 ReAct Answer: {final}")

        # Show which tools were invoked
        tool_calls = [
            m for m in result["messages"]
            if hasattr(m, "tool_calls") and m.tool_calls
        ]
        if tool_calls:
            for tc in tool_calls:
                for call in tc.tool_calls:
                    print(f"   🔧 Tool used: {call['name']}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════╗")
    print("║   🤖  Day 4 — Tool Calling & AI Agents       ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"   Model : {CHAT_MODEL}")
    print(f"   Tools : web_search | summarize | notes\n")

    # Exercise 1 — Test each tool standalone
    test_tools_standalone()

    # Exercises 2 & 3 — Manual agent loop with three queries
    run_agent_tests()

    # Exercise 4 (Bonus) — ReAct agent via LangGraph
    run_react_agent_tests()
