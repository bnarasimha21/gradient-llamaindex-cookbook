# 01 - Simple RAG Query Engine

The most basic RAG pattern: retrieve relevant documents and generate a response.

## What This Shows

- `GradientKBRetriever` — Fetches documents from Gradient Knowledge Base
- `GradientAI` — Generates responses using Gradient LLM
- `RetrieverQueryEngine` — LlamaIndex's orchestration layer

## Architecture

```
Query → Retriever → Relevant Docs → LLM → Response
```

## Code

```python
from llama_index.retrievers.digitalocean.gradientai import GradientKBRetriever
from llama_index.llms.digitalocean.gradientai import GradientAI
from llama_index.core.query_engine import RetrieverQueryEngine

# Initialize components
retriever = GradientKBRetriever(
    knowledge_base_id="kb-xxx",
    api_token="your-token",
    num_results=5
)

llm = GradientAI(
    model="llama3.3-70b-instruct",
    model_access_key="your-key"
)

# Create query engine
query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    llm=llm
)

# Query
response = query_engine.query("How do I create a Droplet?")
print(response)
```

## Run

```bash
python main.py
```

## Key Points

- **5 lines of setup** — minimal boilerplate
- **Automatic context injection** — retriever results become LLM context
- **Metadata preserved** — document IDs, scores available in response
