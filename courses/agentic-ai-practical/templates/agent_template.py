"""
Tool-Using Agent Template - Copy and customize for your use case
Works with: Anthropic Claude API
Time to working: 5 minutes

Usage:
    python agent_template.py
"""

import anthropic
import logging
from typing import Callable, Any

# ============================================================
# CUSTOMIZE THESE
# ============================================================

SYSTEM_PROMPT = "You are a helpful assistant with access to tools."  # TODO: Customize
MODEL = "claude-sonnet-4-20250514"  # TODO: Change model if needed
MAX_TOKENS = 1024
MAX_TURNS = 10

# ============================================================
# COPY THIS ENTIRE BLOCK (production-ready)
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Agent:
    """Production-ready tool-using agent."""

    def __init__(self, system_prompt: str = SYSTEM_PROMPT):
        self.client = anthropic.Anthropic()
        self.system = system_prompt
        self.tools: list[dict] = []
        self.handlers: dict[str, Callable] = {}

    def tool(self, name: str, description: str, **parameters):
        """Decorator to register a tool.

        Example:
            @agent.tool("search", "Search the web", query={"type": "string"})
            def search(query: str) -> str:
                return f"Results for {query}"
        """
        def decorator(func: Callable) -> Callable:
            self.tools.append({
                "name": name,
                "description": description,
                "input_schema": {
                    "type": "object",
                    "properties": parameters,
                    "required": list(parameters.keys())
                }
            })
            self.handlers[name] = func
            return func
        return decorator

    def run(self, user_input: str) -> str:
        """Run the agent and return final response."""
        messages = [{"role": "user", "content": user_input}]

        for turn in range(MAX_TURNS):
            logger.info(f"Turn {turn + 1}/{MAX_TURNS}")

            response = self.client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=self.system,
                tools=self.tools if self.tools else None,
                messages=messages
            )

            # Check if done
            if response.stop_reason == "end_turn":
                for block in response.content:
                    if hasattr(block, 'text'):
                        return block.text
                return ""

            # Process tool calls
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    logger.info(f"Calling tool: {block.name}({block.input})")
                    try:
                        result = self.handlers[block.name](**block.input)
                    except Exception as e:
                        result = f"Error: {str(e)}"
                        logger.error(f"Tool error: {e}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result)
                    })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

        return "Max turns reached without final response"


# ============================================================
# EXAMPLE: Define your tools here
# ============================================================

agent = Agent()


@agent.tool("calculator", "Perform math calculations", expression={"type": "string", "description": "Math expression"})
def calculator(expression: str) -> str:
    """Calculate a math expression."""
    try:
        # WARNING: eval is dangerous with untrusted input. Use a safe math parser in production.
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@agent.tool("get_time", "Get current date and time")
def get_time() -> str:
    """Return current timestamp."""
    from datetime import datetime
    return datetime.now().isoformat()


# TODO: Add your own tools here!
# @agent.tool("your_tool", "Description", param1={"type": "string"})
# def your_tool(param1: str) -> str:
#     return "result"


# ============================================================
# RUN IT
# ============================================================

if __name__ == "__main__":
    # Interactive mode
    print("Agent ready. Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                break
            if not user_input:
                continue

            response = agent.run(user_input)
            print(f"Agent: {response}\n")

        except KeyboardInterrupt:
            break

    print("Goodbye!")
