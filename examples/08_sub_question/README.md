# 08 - Sub-Question Query Engine

Break complex queries into simpler sub-questions, answer each, then synthesize.

## What This Shows

- `GradientKBRetriever` — Answers sub-questions
- `GradientAI` — **Generates sub-questions + synthesizes answers**
- `SubQuestionQueryEngine` — Orchestrates decomposition and synthesis

## Use Case

Complex questions often require multiple pieces of information:

> "Compare Droplets and App Platform for hosting a Python web app"

This breaks into:
1. What are Droplets good for?
2. What is App Platform?
3. How do they compare for Python apps?
4. What are the tradeoffs?

## Architecture

```
┌────────────────────────────────────────┐
│         Complex User Query             │
└───────────────────┬────────────────────┘
                    │
        ┌───────────▼───────────┐
        │  LLM Decomposes into  │ ← GradientAI generates sub-questions
        │    Sub-Questions      │
        └───────────┬───────────┘
                    │
    ┌───────────────┼───────────────┐
    │               │               │
    ▼               ▼               ▼
┌───────┐      ┌───────┐       ┌───────┐
│  Q1   │      │  Q2   │       │  Q3   │
│ RAG   │      │ RAG   │       │ RAG   │
└───┬───┘      └───┬───┘       └───┬───┘
    │               │               │
    ▼               ▼               ▼
┌───────┐      ┌───────┐       ┌───────┐
│  A1   │      │  A2   │       │  A3   │
└───────┴──────┴───────┴───────┴───────┘
                    │
        ┌───────────▼───────────┐
        │  LLM Synthesizes      │ ← GradientAI combines answers
        │  Final Response       │
        └───────────────────────┘
```

## Code Highlights

```python
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool

# Wrap query engine as tool
tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="digitalocean_docs",
    description="DigitalOcean documentation"
)

# Sub-question engine
sub_q_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=[tool],
    llm=llm,
    verbose=True  # See sub-questions
)

# Complex query gets decomposed automatically
response = sub_q_engine.query(
    "Compare Droplets vs App Platform for a high-traffic Python API"
)
```

## Run

```bash
python main.py
```

## Key Points

- **Automatic decomposition** — LLM figures out sub-questions
- **Parallel answering** — Sub-questions can run concurrently
- **Better reasoning** — Complex queries get thorough treatment
