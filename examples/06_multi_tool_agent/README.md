# 06 - Multi-Tool Agent

An agent with multiple tools: KB search + calculator + date/time.

## What This Shows

- `GradientKBRetriever` — One of multiple tools
- `GradientAI` — **Agent brain choosing between tools**
- `FunctionTool` — Custom Python functions as tools
- Tool orchestration and selection

## Tools in This Example

| Tool | Purpose | Type |
|------|---------|------|
| `digitalocean_docs` | Search KB for documentation | QueryEngineTool |
| `calculator` | Math calculations | FunctionTool |
| `current_datetime` | Get current date/time | FunctionTool |

## How Multi-Tool Works

```
User: "If a Droplet costs $48/month, what's the annual cost?"

Agent Reasoning:
  THOUGHT: I need to search for Droplet pricing
  ACTION: digitalocean_docs("Droplet pricing")
  OBSERVATION: "Droplets start at $4/month..."
  
  THOUGHT: User mentioned $48/month, I should calculate annual
  ACTION: calculator("48 * 12")
  OBSERVATION: 576
  
  THOUGHT: I have enough information
  ANSWER: A Droplet costing $48/month would cost $576 annually.
```

## Code Highlights

```python
from llama_index.core.tools import FunctionTool

# Custom calculator tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        result = eval(expression)  # In production, use a safe parser
        return str(result)
    except:
        return "Error: Invalid expression"

calc_tool = FunctionTool.from_defaults(
    fn=calculate,
    name="calculator",
    description="Calculate math expressions. Input: math expression string."
)

# Agent with multiple tools
agent = ReActAgent.from_tools(
    tools=[kb_tool, calc_tool, datetime_tool],
    llm=llm
)
```

## Run

```bash
python main.py
```

## Key Points

- **Tool selection is LLM-driven** — GradientAI decides which tool fits
- **Composable** — Add any FunctionTool for new capabilities
- **Multi-step reasoning** — Agent can use multiple tools per query
