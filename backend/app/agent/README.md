# Agent Module

LangGraph state, graph, nodes, prompts, and tools will live here.

Planned files:

- `state.py`
- `graph.py`
- `nodes.py`
- `tools.py`
- `prompts.py`

Current files:

- `prompts.py` contains the RAG chat system prompt and prompt builder.
- `llm.py` contains LLM client abstractions, an offline extractive fallback, and an OpenAI chat client.
- `state.py` defines shared workflow state.
- `nodes.py` contains deterministic workflow nodes.
- `graph.py` orchestrates the stateful agent and can compile a LangGraph workflow when `langgraph` is installed.
