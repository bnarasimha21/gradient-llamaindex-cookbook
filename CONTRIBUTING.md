# Contributing to Gradient LlamaIndex Cookbook

Thanks for your interest in contributing! 🎉

## Ways to Contribute

- **Add examples** — New use cases or patterns
- **Improve docs** — Clearer explanations, better READMEs
- **Fix bugs** — Found an issue? PRs welcome
- **Add tests** — More coverage is always good

## Development Setup

```bash
# Clone
git clone https://github.com/digitalocean/gradient-llamaindex-cookbook.git
cd gradient-llamaindex-cookbook

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Set up credentials
cp .env.example .env
# Edit .env with your credentials
```

## Running Tests

```bash
# Unit tests (mocked, no credentials needed)
pytest

# Integration tests (requires credentials)
pytest -m integration

# With coverage
pytest --cov=shared --cov=examples
```

## Adding a New Example

1. Create a new folder: `examples/XX_example_name/`
2. Add `README.md` explaining the example
3. Add `main.py` with runnable code
4. Add tests in `tests/test_XX_example_name.py`
5. Update main `README.md` table

### Example Structure

```
examples/XX_your_example/
├── README.md       # What it shows, how to run
├── main.py         # Main runnable code
└── (optional)      # Additional files if needed
```

### Example README Template

```markdown
# XX - Your Example Name

Brief description.

## What This Shows

- Component 1 — What it does
- Component 2 — What it does

## Code Highlights

\`\`\`python
# Key code snippet
\`\`\`

## Run

\`\`\`bash
python main.py
\`\`\`

## Key Points

- Point 1
- Point 2
```

## Code Style

- Use `black` for formatting
- Use `ruff` for linting
- Follow existing patterns in other examples

```bash
# Format
black .

# Lint
ruff check . --fix
```

## Pull Request Process

1. Fork the repo
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit PR with clear description

## Questions?

Open an issue or reach out to the DigitalOcean community.
