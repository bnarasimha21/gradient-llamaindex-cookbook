# 🦙 Gradient LlamaIndex Cookbook

Production-ready examples for building AI applications with [LlamaIndex](https://www.llamaindex.ai/) and [DigitalOcean Gradient](https://www.digitalocean.com/products/gradient).

## What's Inside

12 examples showing how to use the LlamaIndex integrations for DigitalOcean Gradient:

| # | Example | Description | Complexity |
|---|---------|-------------|------------|
| 01 | [Simple RAG](examples/01_simple_rag/) | Basic retrieval-augmented generation | ⭐ |
| 02 | [Chat with Memory](examples/02_chat_memory/) | Conversational RAG with session history | ⭐⭐ |
| 03 | [Streaming](examples/03_streaming/) | Real-time token streaming for better UX | ⭐⭐ |
| 04 | [Reranking](examples/04_reranking/) | Improve retrieval with reranking | ⭐⭐ |
| 05 | [ReAct Agent](examples/05_react_agent/) | Agent that reasons and acts with KB | ⭐⭐⭐ |
| 06 | [Multi-Tool Agent](examples/06_multi_tool_agent/) | Agent with multiple tools | ⭐⭐⭐ |
| 07 | [Router](examples/07_router_query_engine/) | Route queries to different KB sections | ⭐⭐⭐ |
| 08 | [Sub-Questions](examples/08_sub_question/) | Break complex queries into parts | ⭐⭐⭐ |
| 09 | [Workflow](examples/09_workflow/) | Multi-step research assistant | ⭐⭐⭐⭐ |
| 10 | [Evaluation](examples/10_evaluation/) | Measure RAG quality | ⭐⭐⭐ |
| 11 | [Async Batch](examples/11_async_batch/) | Process multiple queries concurrently | ⭐⭐ |
| 12 | [Observability](examples/12_observability/) | Trace and debug your RAG pipeline | ⭐⭐ |

## Prerequisites

- Python 3.10+
- DigitalOcean account with Gradient access
- A Gradient Knowledge Base (we use DigitalOcean docs in examples)

## Quick Start

### 1. Install dependencies

```bash
# Clone the repo
git clone https://github.com/digitalocean/gradient-llamaindex-cookbook.git
cd gradient-llamaindex-cookbook

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### 2. Set up credentials

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
DIGITALOCEAN_ACCESS_TOKEN=your-digitalocean-access-token
MODEL_ACCESS_KEY=your-model-access-key
GRADIENT_KB_ID=kb-your-knowledge-base-uuid
```

### 3. Run an example

```bash
cd examples/01_simple_rag
python main.py
```

## Required Packages

The examples use these LlamaIndex integrations:

```bash
pip install llama-index-retrievers-digitalocean-gradientai
pip install llama-index-llms-digitalocean-gradientai
```

## Knowledge Base Setup

These examples assume a Knowledge Base populated with DigitalOcean documentation. You can:

1. **Use your own KB** — Update queries in examples to match your content
2. **Create a DO docs KB** — Upload DigitalOcean documentation to a new Knowledge Base

### Creating a Knowledge Base

1. Go to [DigitalOcean Gradient Console](https://cloud.digitalocean.com/gradient)
2. Create a new Knowledge Base
3. Upload your documents (PDFs, text files, or use web crawler)
4. Copy the Knowledge Base ID to your `.env`

## Example Structure

Each example follows the same structure:

```
examples/XX_example_name/
├── README.md      # Explanation and usage
├── main.py        # Main runnable example
└── (optional additional files)
```

## Running Tests

```bash
# Run all tests (mocked, no credentials needed)
pytest

# Run integration tests (requires credentials)
pytest -m integration

# Run tests for a specific example
pytest tests/test_01_simple_rag.py
```

## Contributing

Contributions welcome! Please read our [Contributing Guide](CONTRIBUTING.md).

## Related Packages

- [llama-index-retrievers-digitalocean-gradientai](https://pypi.org/project/llama-index-retrievers-digitalocean-gradientai/) — Retriever integration
- [llama-index-llms-digitalocean-gradientai](https://pypi.org/project/llama-index-llms-digitalocean-gradientai/) — LLM integration

## License

MIT License — see [LICENSE](LICENSE) for details.

---

Built with ❤️ by DigitalOcean for the LlamaIndex community.
