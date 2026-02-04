"""
Common Patterns for AI Agents
Copy-paste these snippets into your projects.
Each recipe is self-contained and solves ONE specific problem.
"""

# ============================================================
# RECIPE 1: Simple API Call
# Problem: Make a basic LLM API call
# ============================================================

def simple_call(prompt: str) -> str:
    """Make a simple API call to Claude."""
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# Usage: result = simple_call("Explain Python decorators in one sentence")


# ============================================================
# RECIPE 2: Streaming Response
# Problem: Stream response tokens as they arrive
# ============================================================

def stream_response(prompt: str):
    """Stream response tokens for real-time output."""
    import anthropic
    client = anthropic.Anthropic()

    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    print()

# Usage: stream_response("Write a haiku about coding")


# ============================================================
# RECIPE 3: Retry with Exponential Backoff
# Problem: Handle rate limits and transient errors
# ============================================================

def call_with_retry(func, max_retries: int = 3):
    """Retry API call with exponential backoff."""
    import time
    delay = 1

    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if "rate" in str(e).lower() or "529" in str(e):
                print(f"Retry {attempt + 1}/{max_retries} in {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                raise
    raise Exception("Max retries exceeded")

# Usage: result = call_with_retry(lambda: simple_call("Hello"))


# ============================================================
# RECIPE 4: Structured JSON Output
# Problem: Get reliable JSON from the LLM
# ============================================================

def get_json(prompt: str, schema: dict) -> dict:
    """Extract structured JSON from LLM response."""
    import anthropic
    import json

    client = anthropic.Anthropic()
    system = f"""Return ONLY valid JSON matching this schema:
{json.dumps(schema, indent=2)}
No explanations, no markdown, just JSON."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": prompt}]
    )

    return json.loads(response.content[0].text)

# Usage:
# schema = {"name": "string", "age": "number", "skills": ["string"]}
# result = get_json("Extract info: John is 30, knows Python and SQL", schema)


# ============================================================
# RECIPE 5: Token Counter
# Problem: Estimate tokens before making a call
# ============================================================

def count_tokens(text: str) -> int:
    """Estimate token count (rough approximation)."""
    # Rule of thumb: ~4 characters per token for English
    return len(text) // 4

def count_tokens_accurate(text: str) -> int:
    """Count tokens using the API (more accurate, costs nothing)."""
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.count_tokens(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": text}]
    )
    return response.input_tokens

# Usage: tokens = count_tokens("Your text here")


# ============================================================
# RECIPE 6: Conversation Memory
# Problem: Maintain context across multiple turns
# ============================================================

class Conversation:
    """Simple conversation memory."""

    def __init__(self, system: str = "You are a helpful assistant."):
        import anthropic
        self.client = anthropic.Anthropic()
        self.system = system
        self.messages = []

    def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=self.system,
            messages=self.messages
        )
        assistant_msg = response.content[0].text
        self.messages.append({"role": "assistant", "content": assistant_msg})
        return assistant_msg

    def reset(self):
        self.messages = []

# Usage:
# conv = Conversation()
# print(conv.chat("My name is Alex"))
# print(conv.chat("What's my name?"))


# ============================================================
# RECIPE 7: Cost Calculator
# Problem: Estimate API costs
# ============================================================

PRICING = {
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},  # per million tokens
    "claude-3-5-haiku-20241022": {"input": 0.25, "output": 1.25},
}

def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Estimate cost in dollars."""
    prices = PRICING.get(model, PRICING["claude-sonnet-4-20250514"])
    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]
    return input_cost + output_cost

# Usage: cost = estimate_cost(1000, 500, "claude-sonnet-4-20250514")  # ~$0.01


# ============================================================
# RECIPE 8: Prompt Template
# Problem: Reusable prompts with variables
# ============================================================

def fill_template(template: str, **kwargs) -> str:
    """Fill a prompt template with variables."""
    return template.format(**kwargs)

EXTRACT_TEMPLATE = """Extract the following from the text:
- Names mentioned
- Dates mentioned
- Key actions

Text: {text}

Return as JSON with keys: names, dates, actions"""

# Usage:
# prompt = fill_template(EXTRACT_TEMPLATE, text="John met Mary on Jan 5 for lunch")
# result = get_json(prompt, {"names": [], "dates": [], "actions": []})


# ============================================================
# RECIPE 9: Parallel API Calls
# Problem: Make multiple calls concurrently
# ============================================================

def parallel_calls(prompts: list[str], max_workers: int = 5) -> list[str]:
    """Make multiple API calls in parallel."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(simple_call, prompts))
    return results

# Usage: results = parallel_calls(["Prompt 1", "Prompt 2", "Prompt 3"])


# ============================================================
# RECIPE 10: Error-Safe Wrapper
# Problem: Handle all possible API errors gracefully
# ============================================================

def safe_call(prompt: str, default: str = "Error occurred") -> str:
    """Make an API call with comprehensive error handling."""
    import anthropic

    try:
        return simple_call(prompt)
    except anthropic.RateLimitError:
        return "Rate limited. Try again later."
    except anthropic.APIConnectionError:
        return "Connection error. Check your internet."
    except anthropic.AuthenticationError:
        return "Invalid API key."
    except Exception as e:
        return f"{default}: {str(e)}"

# Usage: result = safe_call("Hello", default="Could not get response")
