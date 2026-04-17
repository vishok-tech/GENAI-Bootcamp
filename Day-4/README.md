# Day 4 — Tool Calling, Manual Agent Loop & ReAct Agent

**Generative AI Bootcamp | Nunnari Academy**

---

## Overview

This project implements a fully functional AI Agent with Tool Calling using **LangChain**, **Ollama (llama3.2:3b)**, **Tavily Search**, and **LangGraph**. The agent dynamically decides which tool to invoke, executes it, processes the result, and loops until it produces a final answer.

---

## Prerequisites

### Install Dependencies

```bash
pip install langchain langchain-community langchain-ollama langgraph tavily-python python-dotenv
```

### Ollama Model

```bash
ollama pull llama3.2:3b
```

### Tavily API Key

Get a free API key at [tavily.com](https://tavily.com) and set it as an environment variable:

```bash
# Windows (PowerShell)
$env:TAVILY_API_KEY="tvly-xxxx"

# macOS / Linux
export TAVILY_API_KEY="tvly-xxxx"
```

Or add it to a `.env` file in the project root:

```
TAVILY_API_KEY=tvly-xxxx
```

---

## Project Structure

```
Day-4/
├── tool_calling.py     # Main script with all exercises
├── notes.json          # Auto-generated file where notes are saved
├── .env                # Your API keys (not committed to Git)
└── README.md
```

---

## Exercises

### Exercise 1 — Define Three Tools

Three tools are defined using the LangChain `@tool` decorator:

| Tool | Description |
|------|-------------|
| `web_search(query: str)` | Searches the web using Tavily and returns real-time results |
| `summarize(text: str)` | Sends text to ChatOllama and returns a short paragraph summary |
| `notes(text: str)` | Extracts a title and content from text and saves it to `notes.json` |

Each tool is tested standalone before connecting to any agent.

---

### Exercise 2 — Manual Tool-Calling Loop

A manual agent loop is built where:

1. The LLM receives a system prompt listing available tools
2. The LLM responds with a structured JSON tool call:
   ```json
   {"tool": "tool_name", "args": {"param": "value"}}
   ```
3. The code parses the JSON, executes the matching tool, and feeds the result back to the LLM
4. The loop continues until the LLM returns:
   ```json
   {"final_answer": "your answer here"}
   ```

Maximum iterations are capped at **6** to prevent infinite loops.

---

### Exercise 3 — Test with Three Queries

The agent loop is tested with three queries:

| Query | Tools Invoked |
|-------|--------------|
| `"What is the latest news on OpenAI?"` | `web_search` → `summarize` → `notes` → `final_answer` |
| `"Summarize this paragraph: ..."` | `final_answer` (directly, no search needed) |
| `"Find the latest news on AI agents and summarize it"` | `web_search` → `summarize` → `notes` → `final_answer` |

---

### Exercise 4 (Bonus) — ReAct Agent via LangGraph

The same tool-calling behavior is recreated using `create_react_agent` from LangGraph with the same three tools and queries.

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(llm, tools)
result = agent.invoke({"messages": [("user", query)]})
```

> **Note:** `create_react_agent` has been deprecated in LangGraph V1.0 and moved to `langchain.agents`. Update your import to `from langchain.agents import create_agent` for future compatibility.

---

## Sample Output

### Exercise 1 — Standalone Tool Tests

```
📌 Testing web_search …
  🔍 [web_search] Searching: 'latest news on LangChain framework'
  {'results': [{'title': 'LangChain Announces Enterprise Agentic AI Platform ...'}]}

📌 Testing summarize …
  📝 [summarize] Summarizing 354 characters …
  Large language models are trained on vast amounts of text data ...

📌 Testing notes …
  🗒️  [notes] Saving note …
  ✅ Note saved — Title: 'Transformer Architecture Overview'
```

### Exercise 3 — Query 1 (OpenAI News)

```
[Loop iteration 1] → Tool: web_search
[Loop iteration 2] → Tool: summarize
[Loop iteration 3] → Tool: notes
[Loop iteration 4] → final_answer

✅ Answer: OpenAI has established its first permanent office in London,
marking an expansion of its operations in the UK ...
```

---

## Saved Notes (notes.json)

Notes are automatically saved after each agent run:

```json
[
  {
    "title": "Transformer Architecture Overview",
    "content": "self-attention replaces recurrent networks for prediction",
    "timestamp": "2026-04-17T00:55:48.013758"
  },
  {
    "title": "OpenAI Expands Operations in London",
    "content": "OpenAI has established its first permanent office in London ...",
    "timestamp": "2026-04-17T00:55:59.983989"
  },
  {
    "title": "AI Agents Surpass Human Performance",
    "content": "Achieving an average score of 66% on a benchmark test.",
    "timestamp": "2026-04-17T00:56:14.935456"
  }
]
```

---

## Key Takeaway

> Tools are just Python functions — the LLM outputs a structured JSON request and your code does the actual execution. This simple idea is the backbone of every autonomous AI agent system.

---

## Run the Script

```bash
python tool_calling.py
```

---

## Acknowledgements

Special thanks to **Shivaprakash Srinivasan** from **Nunnari (நுண்ணறி) Labs** for conducting this hands-on, discussion-driven bootcamp.

---

## Tags

`#GenerativeAI` `#AIAgents` `#ToolCalling` `#LangChain` `#LangGraph` `#Ollama` `#NunnariAcademy` `#FreeBootcamp2026`
