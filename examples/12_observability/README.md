# 12 - Observability & Tracing

Debug and monitor your RAG pipeline with callbacks and tracing.

## What This Shows

- `GradientKBRetriever` — Emits retrieval events
- `GradientAI` — Emits LLM events  
- `CallbackManager` — Central event handling
- `LlamaDebugHandler` — Built-in debugging (free)

## Why Observability?

- **Debug issues** — See exactly what was retrieved and generated
- **Optimize latency** — Find slow components
- **Monitor production** — Track costs, errors, patterns
- **Improve quality** — Understand why bad responses happen

## Events Captured

| Event Type | Data |
|------------|------|
| `RETRIEVE` | Query, nodes returned, scores |
| `LLM` | Prompt, completion, tokens, latency |
| `QUERY` | End-to-end query timing |
| `EMBEDDING` | Embedding calls (if applicable) |

## Code Highlights

```python
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

# Debug handler (free, built-in)
debug_handler = LlamaDebugHandler()
callback_manager = CallbackManager([debug_handler])

# Attach to components
retriever = GradientKBRetriever(
    ...,
    callback_manager=callback_manager
)

llm = GradientAI(
    ...,
    callback_manager=callback_manager
)

# Query (events captured automatically)
response = query_engine.query("How do I...?")

# Inspect events
print(debug_handler.get_event_time_info())  # Timing
print(debug_handler.get_llm_inputs_outputs())  # LLM details
```

## Run

```bash
python main.py
```

## Integration Options

| Tool | Cost | Features |
|------|------|----------|
| `LlamaDebugHandler` | Free | Basic tracing, timing |
| Phoenix (Arize) | Free tier | Visual traces, evals |
| LangFuse | Free tier | Traces, prompts, scores |
| Weights & Biases | Free tier | Experiment tracking |

## Key Points

- **Both components emit events** — Full pipeline visibility
- **Zero-config start** — LlamaDebugHandler works immediately
- **Production-ready** — Integrate with your observability stack
