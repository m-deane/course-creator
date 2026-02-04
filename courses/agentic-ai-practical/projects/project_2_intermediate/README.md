# Project 2: Multi-Tool Research Agent

**Build an agent that can search the web, read files, and synthesize information.**

## What You'll Build

An agent that:
1. Takes a research question
2. Uses tools to gather information (web search, file reading, calculations)
3. Synthesizes findings into a report
4. Cites all sources

## Time: 4-6 hours

## Demo

```bash
$ python researcher.py "Compare Python vs Rust for CLI tools"

🔍 Searching for "Python CLI frameworks 2024"...
🔍 Searching for "Rust CLI frameworks performance"...
📄 Reading local notes: cli_comparison.md
🧮 Calculating benchmark differences...

📋 Research Report
==================

## Summary
Both Python and Rust are excellent for CLI tools, with different tradeoffs...

## Key Findings
1. **Development Speed**: Python is 2-3x faster to prototype (Source: web search)
2. **Runtime Performance**: Rust is 10-50x faster (Source: benchmarks)
3. **Distribution**: Rust compiles to single binary (Source: cli_comparison.md)

## Recommendation
For internal tools: Python. For distributed tools: Rust.

Sources:
- https://blog.example.com/python-cli-2024
- https://rust-lang.org/cli
- Local: cli_comparison.md
```

## What You'll Learn

- Multi-tool agent architecture
- Tool orchestration patterns
- Error handling in agents
- Report generation

## Tools to Implement

| Tool | Purpose |
|------|---------|
| `web_search` | Search the web (use DuckDuckGo API) |
| `read_file` | Read local files |
| `calculator` | Perform calculations |
| `write_report` | Save report to file |

## Getting Started

1. Copy the starter code
2. Implement each tool
3. Wire up the agent loop
4. Test with different research questions

## Starter Code

```python
"""
Multi-Tool Research Agent
Fill in the TODO sections.
"""

import anthropic
import json

client = anthropic.Anthropic()

# Tool definitions
TOOLS = [
    {
        "name": "web_search",
        "description": "Search the web for information. Returns top 3 results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "read_file",
        "description": "Read contents of a local file",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "calculator",
        "description": "Perform mathematical calculations",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
            },
            "required": ["expression"]
        }
    }
]


def web_search(query: str) -> str:
    """Search the web."""
    # TODO: Implement using DuckDuckGo or similar
    # Hint: Use duckduckgo-search package
    pass


def read_file(path: str) -> str:
    """Read a local file."""
    # TODO: Implement with error handling
    pass


def calculator(expression: str) -> str:
    """Calculate a math expression."""
    # TODO: Implement safely (don't use raw eval)
    pass


def run_tool(name: str, args: dict) -> str:
    """Execute a tool and return result."""
    handlers = {
        "web_search": web_search,
        "read_file": read_file,
        "calculator": calculator,
    }
    # TODO: Call the appropriate handler
    pass


def research(question: str) -> str:
    """Run the research agent."""
    messages = [{
        "role": "user",
        "content": f"""Research this question and provide a comprehensive answer:

{question}

Use the available tools to gather information. Search the web for current data,
read any relevant local files, and perform calculations if needed.

End with a structured report that includes:
1. Summary (2-3 sentences)
2. Key Findings (bulleted list with sources)
3. Recommendation (if applicable)
4. Sources (list all URLs and files used)"""
    }]

    # TODO: Implement the agent loop
    # 1. Call Claude with tools
    # 2. If tool_use, execute tool and continue
    # 3. If end_turn, return the response
    pass


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python researcher.py 'your research question'")
        return

    question = " ".join(sys.argv[1:])
    print(f"🔬 Researching: {question}\n")

    report = research(question)
    print("\n" + "=" * 50)
    print(report)


if __name__ == "__main__":
    main()
```

## Solution

Check [solution.py](solution.py) after attempting it yourself.

## Extend It

- Add a `summarize_url` tool to read web pages
- Implement caching to avoid repeated searches
- Add a `save_report` tool to write markdown files
- Build a web UI with Gradio

## Portfolio Tips

This project demonstrates:
- Agent architecture design
- External API integration
- Error handling patterns
- Report generation

Great for showing employers you can build production AI systems!
