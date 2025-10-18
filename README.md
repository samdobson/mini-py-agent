# MiniAgent

A minimal implementation of an agentic AI assistant. Made as simple as possible for educational purposes.

Read the accompanying blog post at https://samdobson.uk/posts/how-to-build-an-agent/

## What It Does

MiniAgent is an interactive command-line agent that can list, read and edit files, using Claude Haiku as the model.

The agent maintains conversation context across multiple turns and automatically executes tool calls as needed to fulfil user requests.

## What It Shows

This implementation demonstrates the core concepts of building an AI agent:

1. **Conversation Management** - Maintaining multi-turn dialogue with tool results
2. **Agentic Loop** - The request ’ inference ’ tool execution ’ response cycle
3. **Tool Use Pattern** - How to define tools with schemas and connect them to Python functions

## Key Components

- `Agent` class - Orchestrates the conversation loop and tool execution
- Tool definitions (`READ_FILE_TOOL`, `LIST_FILES_TOOL`, `EDIT_FILE_TOOL`) - Define capabilities available to the model
- `_run_inference()` - Makes API calls to the LLM with tool schemas
- `_execute_tool()` - Routes tool calls to their Python implementations

## Usage

```bash
pip install anthropic
python agent.py
```

Remeber to set the `ANTHROPIC_API_KEY` environment variable before running.
