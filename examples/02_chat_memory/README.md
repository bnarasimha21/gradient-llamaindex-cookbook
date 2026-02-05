# 02 - Chat Engine with Memory

Conversational RAG that remembers chat history across turns.

## What This Shows

- `GradientKBRetriever` — Context retrieval
- `GradientAI` — Chat responses
- `CondensePlusContextChatEngine` — Conversation management
- `ChatMemoryBuffer` — Session history storage

## Architecture

```
User Message → Condense (with history) → Retrieve → Generate → Response
     ↑                                                              |
     └──────────────── Memory Buffer ←─────────────────────────────┘
```

## How It Works

1. **Condense**: Rewrites user query using chat history for context
2. **Retrieve**: Fetches relevant docs with condensed query  
3. **Generate**: LLM responds with retrieved context + chat history
4. **Remember**: Response added to memory for next turn

## Code Highlights

```python
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer

# Memory stores last 10 messages
memory = ChatMemoryBuffer.from_defaults(token_limit=4096)

# Chat engine with memory
chat_engine = CondensePlusContextChatEngine.from_defaults(
    retriever=retriever,
    llm=llm,
    memory=memory,
    verbose=True
)

# Multi-turn conversation
response1 = chat_engine.chat("What is a Droplet?")
response2 = chat_engine.chat("How do I resize it?")  # "it" refers to Droplet
```

## Run

```bash
python main.py
```

## Key Points

- **Context-aware follow-ups** — "it", "that", "the same" work naturally
- **Configurable memory** — Token limit prevents context overflow
- **Verbose mode** — See the condensed queries for debugging
