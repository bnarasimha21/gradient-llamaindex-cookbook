# 10 - RAG Evaluation

Measure your RAG pipeline's quality: retrieval relevance and answer faithfulness.

## What This Shows

- `GradientKBRetriever` — Being evaluated
- `GradientAI` — Can serve as evaluator judge LLM
- LlamaIndex evaluation framework
- Key RAG metrics

## Why Evaluate?

- **Measure improvement** — Know if changes help or hurt
- **Find failure modes** — Identify where retrieval or generation fails
- **Production confidence** — Quantify quality before deployment

## Key Metrics

| Metric | What It Measures | Good Score |
|--------|------------------|------------|
| **Faithfulness** | Is the answer grounded in retrieved docs? | > 0.8 |
| **Relevancy** | Are retrieved docs relevant to the query? | > 0.7 |
| **Answer Relevancy** | Does the answer address the question? | > 0.8 |
| **Context Precision** | Are relevant docs ranked higher? | > 0.7 |

## Code Highlights

```python
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    BatchEvalRunner
)

# Create evaluators
faithfulness = FaithfulnessEvaluator(llm=llm)
relevancy = RelevancyEvaluator(llm=llm)

# Evaluate a single response
response = query_engine.query("How do I create a Droplet?")

faith_result = await faithfulness.aevaluate_response(response=response)
print(f"Faithfulness: {faith_result.score}")  # 0.0 - 1.0

rel_result = await relevancy.aevaluate_response(
    query="How do I create a Droplet?",
    response=response
)
print(f"Relevancy: {rel_result.score}")
```

## Run

```bash
python main.py
```

## Key Points

- **LLM-as-judge** — GradientAI evaluates GradientAI outputs
- **Batch evaluation** — Run over many queries for reliable metrics
- **Actionable insights** — Low scores point to specific issues
