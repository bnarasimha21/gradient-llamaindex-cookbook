# 04 - Reranking for Better Retrieval

Improve retrieval quality by reranking initial results.

## What This Shows

- `GradientKBRetriever` — Initial retrieval
- `SentenceTransformerRerank` — Free, local reranking
- Node postprocessing pipeline

## Why Rerank?

Initial retrieval (BM25, embeddings) optimizes for recall — get many relevant docs.
Reranking optimizes for precision — put the BEST docs first.

```
Query → Retrieve Top 10 → Rerank → Keep Top 3 → Generate
```

## Reranking Options

| Option | Cost | Quality | Speed |
|--------|------|---------|-------|
| `SentenceTransformerRerank` | Free | Good | ~100ms |
| `CohereRerank` | $0.001/search | Excellent | ~200ms |
| `ColbertRerank` | Free | Very Good | ~150ms |

## Code Highlights

```python
from llama_index.core.postprocessor import SentenceTransformerRerank

# Reranker (runs locally, free)
reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=3  # Keep top 3 after reranking
)

# Add to query engine
query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,  # Gets 10 docs
    llm=llm,
    node_postprocessors=[reranker]  # Reranks to top 3
)
```

## Run

```bash
# Install reranking dependencies
pip install sentence-transformers

python main.py
```

## Key Points

- **Two-stage retrieval** — Recall then precision
- **Free local option** — No API keys needed
- **Measurable improvement** — Compare with/without reranking
