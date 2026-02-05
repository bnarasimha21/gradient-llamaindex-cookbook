# 05 - ReAct Agent with Knowledge Base

An agent that reasons about when to search the KB vs respond directly.

## What This Shows

- `GradientKBRetriever` — Wrapped as a searchable tool
- `GradientAI` — **Agent's reasoning brain**
- `ReActAgent` — Reasoning + Acting loop
- `QueryEngineTool` — Wraps query engine as agent tool

## How ReAct Works

```
User Query
    ↓
┌─────────────────────┐
│   THOUGHT           │ ← LLM reasons about the query
│   "I need to search │
│    for Droplet info"│
└─────────────────────┘
    ↓
┌─────────────────────┐
│   ACTION            │ ← Agent decides to use tool
│   search_kb(        │
│     "Droplet"       │
│   )                 │
└─────────────────────┘
    ↓
┌─────────────────────┐
│   OBSERVATION       │ ← Tool returns results
│   "A Droplet is..." │
└─────────────────────┘
    ↓
┌─────────────────────┐
│   THOUGHT           │ ← LLM reasons about results
│   "I have enough    │
│    info to answer"  │
└─────────────────────┘
    ↓
┌─────────────────────┐
│   FINAL ANSWER      │
└─────────────────────┘
```

## Code Highlights

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool

# Wrap query engine as a tool
kb_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="digitalocean_docs",
    description="Search DigitalOcean documentation for technical information"
)

# Create ReAct agent
agent = ReActAgent.from_tools(
    tools=[kb_tool],
    llm=llm,
    verbose=True  # See reasoning steps
)

# Agent decides when to search
response = agent.chat("What's the difference between Droplets and Kubernetes?")
```

## Run

```bash
python main.py
```

## Key Points

- **LLM drives decisions** — GradientAI is the agent's brain
- **Transparent reasoning** — Verbose mode shows thought process
- **Selective retrieval** — Agent only searches when needed
