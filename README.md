# Linki

Linki is a small Typer-based CLI scaffold with LangChain tools restricted to a workspace.

## Usage

```bash
pip install -e .
Linki "Summarize this workspace" --workspace ./workspace
```

Set provider API keys in a `.env` file or in the environment before running model calls.

```bash
Linki "Summarize this workspace" --provider openai --workspace ./workspace
Linki "Summarize this workspace" --provider deepseek --workspace ./workspace
Linki "Create a Tetris game as a single HTML file" --provider deepseek --workspace ./workspace --max-loops 10
```

Provider defaults:

- `openai`: `OPENAI_MODEL`, default `gpt-4o-mini`
- `deepseek`: `DEEPSEEK_MODEL`, default `deepseek-v4-flash`
