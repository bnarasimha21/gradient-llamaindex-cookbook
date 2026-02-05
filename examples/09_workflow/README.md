# 09 - Workflow: Research Assistant

Multi-step workflow that retrieves, summarizes, and fact-checks.

## What This Shows

- `GradientKBRetriever` — Retrieval step
- `GradientAI` — Multiple LLM steps (summarize, verify, respond)
- `Workflow` — LlamaIndex's workflow orchestration
- `@step` decorators — Define workflow steps
- `Context` — Pass data between steps

## Why Workflows?

For complex AI tasks, you need more than retrieve→generate:

- **Multi-step processing** — Retrieve → Summarize → Verify → Format
- **Conditional logic** — Different paths based on results
- **Error handling** — Graceful failures at each step
- **Observability** — Track each step's performance

## This Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    Research Assistant                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐│
│  │ Retrieve │ → │ Summarize│ → │  Verify  │ → │ Format ││
│  │   Docs   │    │  Points  │    │  Facts   │    │Response││
│  └──────────┘    └──────────┘    └──────────┘    └────────┘│
│       ↓               ↓               ↓              ↓      │
│  GradientKB      GradientAI      GradientAI     GradientAI │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Code Highlights

```python
from llama_index.core.workflow import Workflow, step, Context, Event, StartEvent, StopEvent

class ResearchWorkflow(Workflow):
    
    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> RetrievalEvent:
        nodes = await retriever.aretrieve(ev.query)
        ctx.data["nodes"] = nodes
        return RetrievalEvent(nodes=nodes)
    
    @step
    async def summarize(self, ctx: Context, ev: RetrievalEvent) -> SummaryEvent:
        summary = await llm.acomplete(f"Summarize: {ev.nodes}")
        return SummaryEvent(summary=summary)
    
    @step
    async def verify(self, ctx: Context, ev: SummaryEvent) -> VerifiedEvent:
        verified = await llm.acomplete(f"Verify accuracy: {ev.summary}")
        return VerifiedEvent(content=verified)
    
    @step
    async def respond(self, ctx: Context, ev: VerifiedEvent) -> StopEvent:
        return StopEvent(result=ev.content)

# Run workflow
workflow = ResearchWorkflow()
result = await workflow.run(query="How do I set up a load balancer?")
```

## Run

```bash
python main.py
```

## Key Points

- **Modern LlamaIndex pattern** — Workflows are the future
- **Async-first** — Built for production performance
- **Type-safe events** — Clear data contracts between steps
- **Both components shine** — Retriever and LLM used meaningfully
