# 03 - Streaming Responses

Real-time token streaming for better user experience.

## What This Shows

- `GradientAI` — **Streaming generation capability**
- `StreamingResponse` — Token-by-token output
- Async streaming patterns

## Why Streaming?

- **Better UX** — Users see responses as they generate
- **Lower perceived latency** — First token appears in ~200ms vs waiting 5s+
- **Early termination** — Stop generation if user navigates away

## Code Highlights

```python
# Sync streaming
response = llm.stream_complete("Explain Kubernetes")
for token in response:
    print(token.delta, end="", flush=True)

# Async streaming
async for token in await llm.astream_complete("Explain Kubernetes"):
    print(token.delta, end="", flush=True)

# With query engine
response = query_engine.query("How do I deploy?", streaming=True)
response.print_response_stream()
```

## Run

```bash
# Basic streaming
python main.py

# Async streaming
python main.py --async
```

## Key Points

- **GradientAI supports streaming natively** — No extra setup
- **Works with query engines** — Full RAG with streaming output
- **Async-first** — Production-ready patterns included
