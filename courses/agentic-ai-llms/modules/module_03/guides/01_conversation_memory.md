# Conversation Memory: Managing History and Context

> **Reading time:** ~12 min | **Module:** 3 — Memory & Context | **Prerequisites:** Module 2 — Tool Use

LLMs process each request independently—they don't remember previous conversations. Conversation memory systems maintain history, manage context window limits, and preserve continuity across interactions.

<div class="callout-insight">

**Insight:** Context windows are finite; conversations are not. The art of conversation memory is deciding what to keep, what to summarize, and what to discard while maintaining coherent, contextual responses.

</div>

---

## The Memory Problem

### Each API Call is Stateless


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>

```python

# Call 1
response = client.messages.create(
    messages=[{"role": "user", "content": "My name is Alice."}]
)

# Response: "Nice to meet you, Alice!"

# Call 2 - No memory of Call 1
response = client.messages.create(
    messages=[{"role": "user", "content": "What's my name?"}]
)

# Response: "I don't know your name. You haven't told me."
```

</div>
</div>

### Solution: Maintain Message History

```python

# Maintain conversation history
conversation = []

# Turn 1
conversation.append({"role": "user", "content": "My name is Alice."})
response = client.messages.create(messages=conversation)
conversation.append({"role": "assistant", "content": response.content[0].text})

# Turn 2 - Include full history
conversation.append({"role": "user", "content": "What's my name?"})
response = client.messages.create(messages=conversation)

# Response: "Your name is Alice."
```

---

## Memory Strategies

### 1. Buffer Memory (Full History)

Keep all messages in order:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>

```python
class BufferMemory:
    """Simple buffer that stores all messages."""

    def __init__(self):
        self.messages: list[dict] = []

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def get_messages(self) -> list[dict]:
        return self.messages.copy()

    def clear(self):
        self.messages = []
```

</div>
</div>

**Pros:** Complete context, simple
**Cons:** Unbounded growth, expensive for long conversations

### 2. Window Memory (Recent K Messages)

Keep only the last K exchanges:

```python
class WindowMemory:
    """Keep only the last k messages."""

    def __init__(self, k: int = 10):
        self.k = k
        self.messages: list[dict] = []

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        # Trim to window size (k pairs = 2k messages)
        if len(self.messages) > self.k * 2:
            self.messages = self.messages[-(self.k * 2):]

    def get_messages(self) -> list[dict]:
        return self.messages.copy()
```

**Pros:** Bounded size, predictable costs
**Cons:** Loses early context

### 3. Summary Memory

Periodically summarize older messages:

```python
class SummaryMemory:
    """Summarize old messages to maintain context efficiently."""

    def __init__(self, client, summary_threshold: int = 20):
        self.client = client
        self.summary_threshold = summary_threshold
        self.summary: str = ""
        self.recent_messages: list[dict] = []

    def add_message(self, role: str, content: str):
        self.recent_messages.append({"role": role, "content": content})

        if len(self.recent_messages) >= self.summary_threshold:
            self._update_summary()

    def _update_summary(self):
        """Summarize recent messages and update running summary."""
        conversation_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in self.recent_messages
        )

        prompt = f"""Summarize the following conversation, preserving key information,
decisions, and context needed for continuation.

Previous summary: {self.summary or 'None'}

New conversation:
{conversation_text}

Updated summary:"""

        response = self.client.messages.create(
            model="claude-3-haiku-20240307",  # Fast, cheap model for summaries
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        self.summary = response.content[0].text
        self.recent_messages = []

    def get_messages(self) -> list[dict]:
        """Get messages with summary as context."""
        messages = []
        if self.summary:
            messages.append({
                "role": "user",
                "content": f"[Previous conversation summary: {self.summary}]"
            })
        messages.extend(self.recent_messages)
        return messages
```

**Pros:** Maintains long-term context efficiently
**Cons:** Summarization loses details, adds latency

### 4. Token-Limited Memory

Keep messages up to a token budget:

```python
import anthropic


class TokenLimitedMemory:
    """Keep messages within a token budget."""

    def __init__(self, client, max_tokens: int = 50000):
        self.client = client
        self.max_tokens = max_tokens
        self.messages: list[dict] = []

    def _count_tokens(self, messages: list[dict]) -> int:
        """Count tokens in messages."""
        response = self.client.messages.count_tokens(
            model="claude-3-5-sonnet-20241022",
            messages=messages
        )
        return response.input_tokens

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self._trim_to_budget()

    def _trim_to_budget(self):
        """Remove oldest messages until under token budget."""
        while len(self.messages) > 1:
            current_tokens = self._count_tokens(self.messages)
            if current_tokens <= self.max_tokens:
                break
            # Remove oldest message pair
            self.messages.pop(0)
            if self.messages and self.messages[0]["role"] == "assistant":
                self.messages.pop(0)

    def get_messages(self) -> list[dict]:
        return self.messages.copy()
```

**Pros:** Precise budget control
**Cons:** Token counting adds latency

---

## Hybrid Memory Systems

### Combining Short and Long-term Memory


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>

```python
class HybridMemory:
    """Combine buffer for recent, summary for old, vector for retrieval."""

    def __init__(self, client, vector_store):
        self.client = client
        self.vector_store = vector_store
        self.summary = ""
        self.buffer: list[dict] = []
        self.buffer_size = 10

    def add_message(self, role: str, content: str):
        message = {"role": role, "content": content}
        self.buffer.append(message)

        # Store in vector DB for retrieval
        self.vector_store.add(
            documents=[content],
            metadatas=[{"role": role, "timestamp": time.time()}],
            ids=[str(uuid.uuid4())]
        )

        # Summarize when buffer full
        if len(self.buffer) >= self.buffer_size:
            self._summarize_buffer()

    def _summarize_buffer(self):
        """Summarize oldest messages in buffer."""
        to_summarize = self.buffer[:self.buffer_size // 2]
        self.buffer = self.buffer[self.buffer_size // 2:]

        conversation = "\n".join(f"{m['role']}: {m['content']}" for m in to_summarize)

        response = self.client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": f"Briefly summarize:\n\n{conversation}"
            }]
        )

        new_summary = response.content[0].text
        self.summary = f"{self.summary}\n{new_summary}".strip()

    def get_context(self, query: str, n_retrieved: int = 3) -> list[dict]:
        """Get full context for a query."""
        messages = []

        # Add summary of old conversations
        if self.summary:
            messages.append({
                "role": "user",
                "content": f"[Context from earlier: {self.summary}]"
            })

        # Add relevant retrieved messages
        results = self.vector_store.query(query_texts=[query], n_results=n_retrieved)
        if results["documents"][0]:
            retrieved = "\n".join(results["documents"][0])
            messages.append({
                "role": "user",
                "content": f"[Related previous information: {retrieved}]"
            })

        # Add recent buffer
        messages.extend(self.buffer)

        return messages
```

</div>
</div>

---

## Context Window Management

### Estimating Token Usage


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>

```python
def estimate_tokens(text: str) -> int:
    """Rough estimate: ~4 characters per token for English."""
    return len(text) // 4


def estimate_conversation_tokens(messages: list[dict]) -> int:
    """Estimate tokens in a conversation."""
    total = 0
    for m in messages:
        total += estimate_tokens(m["content"])
        total += 4  # Role tokens overhead
    return total
```

</div>
</div>

### Dynamic Context Allocation

```python
class DynamicContextManager:
    """Dynamically allocate context budget across sources."""

    def __init__(self, max_tokens: int = 100000):
        self.max_tokens = max_tokens
        # Reserve tokens for response
        self.response_budget = 4000
        self.available = max_tokens - self.response_budget

    def allocate(
        self,
        system_prompt: str,
        retrieved_docs: list[str],
        conversation: list[dict],
        min_conversation_turns: int = 4
    ) -> dict:
        """Allocate tokens across different context sources."""

        system_tokens = estimate_tokens(system_prompt)
        remaining = self.available - system_tokens

        # Reserve minimum for recent conversation
        min_conv_tokens = min_conversation_turns * 500
        remaining -= min_conv_tokens

        # Allocate to retrieved docs
        doc_budget = min(remaining * 0.6, 20000)  # Up to 60% or 20K
        selected_docs = self._select_documents(retrieved_docs, int(doc_budget))

        # Remaining to conversation history
        doc_actual = sum(estimate_tokens(d) for d in selected_docs)
        conv_budget = remaining - doc_actual + min_conv_tokens
        selected_conv = self._select_messages(conversation, int(conv_budget))

        return {
            "system": system_prompt,
            "documents": selected_docs,
            "conversation": selected_conv
        }

    def _select_documents(self, docs: list[str], budget: int) -> list[str]:
        """Select documents within token budget."""
        selected = []
        used = 0
        for doc in docs:
            doc_tokens = estimate_tokens(doc)
            if used + doc_tokens <= budget:
                selected.append(doc)
                used += doc_tokens
        return selected

    def _select_messages(self, messages: list[dict], budget: int) -> list[dict]:
        """Select most recent messages within budget."""
        selected = []
        used = 0
        for m in reversed(messages):
            m_tokens = estimate_tokens(m["content"]) + 4
            if used + m_tokens <= budget:
                selected.insert(0, m)
                used += m_tokens
            else:
                break
        return selected
```

---

## Conversation Patterns

### System Prompt + History


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>

```python
def build_conversation(
    system_prompt: str,
    memory: BufferMemory,
    current_query: str
) -> tuple[str, list[dict]]:
    """Build complete conversation context."""

    messages = memory.get_messages()
    messages.append({"role": "user", "content": current_query})

    return system_prompt, messages


# Usage
system, messages = build_conversation(
    system_prompt="You are a helpful assistant.",
    memory=conversation_memory,
    current_query="What did we discuss earlier?"
)

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    system=system,
    messages=messages
)
```


### Injecting Context

```python
def inject_context(messages: list[dict], context: str, position: str = "start") -> list[dict]:
    """Inject context into conversation."""

    context_message = {
        "role": "user",
        "content": f"[Reference information: {context}]"
    }

    result = messages.copy()
    if position == "start":
        result.insert(0, context_message)
    elif position == "before_last":
        result.insert(-1, context_message)

    return result
```

---

## Persistence

### Saving and Loading Memory


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>

```python
import json
from pathlib import Path


class PersistentMemory(BufferMemory):
    """Memory that persists to disk."""

    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = Path(filepath)
        self._load()

    def _load(self):
        """Load messages from disk."""
        if self.filepath.exists():
            with open(self.filepath, 'r') as f:
                data = json.load(f)
                self.messages = data.get("messages", [])

    def _save(self):
        """Save messages to disk."""
        with open(self.filepath, 'w') as f:
            json.dump({"messages": self.messages}, f)

    def add_user_message(self, content: str):
        super().add_user_message(content)
        self._save()

    def add_assistant_message(self, content: str):
        super().add_assistant_message(content)
        self._save()
```


### Session-Based Memory

```python
class SessionManager:
    """Manage multiple conversation sessions."""

    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

    def create_session(self) -> str:
        """Create a new session."""
        session_id = str(uuid.uuid4())[:8]
        return session_id

    def get_memory(self, session_id: str) -> PersistentMemory:
        """Get or create memory for a session."""
        filepath = self.storage_dir / f"{session_id}.json"
        return PersistentMemory(str(filepath))

    def list_sessions(self) -> list[str]:
        """List all session IDs."""
        return [f.stem for f in self.storage_dir.glob("*.json")]

    def delete_session(self, session_id: str):
        """Delete a session."""
        filepath = self.storage_dir / f"{session_id}.json"
        if filepath.exists():
            filepath.unlink()
```

---

## Best Practices

1. **Start Simple**: Buffer memory is often sufficient
2. **Monitor Token Usage**: Log token counts to understand costs
3. **Use System Prompts for Static Context**: Don't repeat in every message
4. **Prioritize Recent Context**: Recent messages usually matter most
5. **Summarize Strategically**: Use fast models for summaries
6. **Test Memory Boundaries**: Verify behavior when context is truncated

<div class="callout-key">

**Key Concept Summary:** This guide covered the core concepts. Review the companion slides for visual summaries and the hands-on notebook for practice implementations.


---

*Conversation memory transforms stateless API calls into continuous interactions. Choose the right strategy for your use case and iterate based on real usage patterns.*


## Practice Questions

1. Explain in your own words how the concepts in this guide relate to building production agents.
2. What are the key tradeoffs you need to consider when applying these techniques?
3. Describe a scenario where the approach from this guide would be the wrong choice, and what you would use instead.

---

**Next Steps:**

<a class="link-card" href="./01_conversation_memory_slides.md">
  <div class="link-card-title">Conversation Memory Patterns — Companion Slides</div>
  <div class="link-card-description">Visual slide deck with diagrams, speaker notes, and key takeaways.</div>
</a>

<a class="link-card" href="../notebooks/01_memory_patterns.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with working code and guided exercises.</div>
</a>
