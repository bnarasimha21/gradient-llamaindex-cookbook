# 11 - Async Batch Processing

Process multiple queries concurrently for high-throughput applications.

## What This Shows

- `GradientKBRetriever.aretrieve()` — **Async retrieval**
- `GradientAI` async methods — Async LLM calls
- `asyncio.gather()` — Concurrent execution
- Production batch patterns

## Why Async?

| Approach | 5 Queries | 50 Queries |
|----------|-----------|------------|
| Sequential | ~15 sec | ~150 sec |
| Async Concurrent | ~5 sec | ~15 sec |

Async lets you overlap I/O wait times — while one request waits for the API, others can run.

## Code Highlights

```python
import asyncio

# Async retrieval
async def async_retrieve(query: str):
    nodes = await retriever.aretrieve(query)
    return nodes

# Process multiple queries concurrently
async def process_batch(queries: list[str]):
    tasks = [async_retrieve(q) for q in queries]
    results = await asyncio.gather(*tasks)
    return results

# Run
queries = ["How do I...?", "What is...?", "Where can I...?"]
results = asyncio.run(process_batch(queries))
```

## Patterns Shown

1. **Simple async retrieval** — Single query, async
2. **Batch retrieval** — Multiple queries, concurrent
3. **Full async RAG** — Retrieve + generate, all async
4. **Rate-limited batch** — Respect API limits with semaphores

## Run

```bash
# Basic async demo
python main.py

# Batch processing with timing
python main.py --batch 10
```

## Key Points

- **Native async support** — Both retriever and LLM support async
- **Significant speedup** — 3-10x faster for batch workloads
- **Production-ready** — Semaphores for rate limiting included
