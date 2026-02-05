# 07 - Router Query Engine

Route queries to different KB sections or query engines based on intent.

## What This Shows

- `GradientKBRetriever` × N — **Multiple retriever instances**
- `GradientAI` — Powers routing decisions + generation
- `RouterQueryEngine` — Routes to appropriate engine
- `LLMSingleSelector` — LLM-based query classification

## Use Case

When your KB has distinct sections (Droplets, Kubernetes, Databases), 
route queries to the most relevant section for better precision.

## Architecture

```
                    ┌─────────────────┐
                    │  User Query     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  LLM Selector   │ ← GradientAI classifies query
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ Droplet Docs  │   │ Kubernetes    │   │ Database Docs │
│ Query Engine  │   │ Query Engine  │   │ Query Engine  │
└───────────────┘   └───────────────┘   └───────────────┘
```

## Code Highlights

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool

# Multiple query engines for different topics
# (In practice, these might point to different KBs or use different filters)
compute_engine = get_query_engine()  # Droplets, VMs
k8s_engine = get_query_engine()      # Kubernetes
db_engine = get_query_engine()       # Databases

# Wrap as tools with descriptions
tools = [
    QueryEngineTool.from_defaults(
        query_engine=compute_engine,
        description="Questions about Droplets, VMs, and compute resources"
    ),
    QueryEngineTool.from_defaults(
        query_engine=k8s_engine,
        description="Questions about Kubernetes clusters and containers"
    ),
    QueryEngineTool.from_defaults(
        query_engine=db_engine,
        description="Questions about managed databases (PostgreSQL, MySQL, Redis)"
    ),
]

# Router selects the best engine
router = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(llm=llm),
    query_engine_tools=tools
)

# Automatically routes to appropriate engine
response = router.query("How do I scale my Kubernetes cluster?")
```

## Run

```bash
python main.py
```

## Key Points

- **Multiple retrievers showcase flexibility** — Same integration, different configs
- **LLM-powered routing** — GradientAI decides where to route
- **Improved precision** — Narrower search scope = better results
