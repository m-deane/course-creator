# Conversation Memory

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONVERSATION MEMORY                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Turn 1:                                                       │
│   ┌─────────────────────────────────────────────────────┐       │
│   │ messages = [                                         │       │
│   │   {"role": "user", "content": "I'm Alex"}            │       │
│   │ ]                                                    │       │
│   └─────────────────────────────────────────────────────┘       │
│                          │                                      │
│                          ▼                                      │
│   ┌─────────────────────────────────────────────────────┐       │
│   │ messages = [                                         │       │
│   │   {"role": "user", "content": "I'm Alex"},           │       │
│   │   {"role": "assistant", "content": "Hi Alex!"}       │  ←ADD │
│   │ ]                                                    │       │
│   └─────────────────────────────────────────────────────┘       │
│                                                                 │
│   Turn 2:                                                       │
│   ┌─────────────────────────────────────────────────────┐       │
│   │ messages = [                                         │       │
│   │   {"role": "user", "content": "I'm Alex"},           │       │
│   │   {"role": "assistant", "content": "Hi Alex!"},      │       │
│   │   {"role": "user", "content": "What's my name?"}     │  ←ADD │
│   │ ]                                                    │       │
│   └─────────────────────────────────────────────────────┘       │
│                          │                                      │
│                          ▼                                      │
│   LLM sees ALL messages → knows name is Alex                    │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ TL;DR: Memory = just keep adding messages to the list           │
├─────────────────────────────────────────────────────────────────┤
│ Code:                                                           │
│   messages.append({"role": "user", "content": user_input})      │
│   response = client.messages.create(messages=messages)          │
│   messages.append({"role": "assistant", "content": response})   │
├─────────────────────────────────────────────────────────────────┤
│ Pitfall: Context window fills up! Trim old messages:           │
│          messages = [system] + messages[-20:]  # Keep last 20   │
└─────────────────────────────────────────────────────────────────┘
```

## Memory Strategies

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  SIMPLE: Keep all messages                                     │
│  messages = [...all messages...]                               │
│  Problem: Hits context limit                                   │
│                                                                │
│  SLIDING WINDOW: Keep last N                                   │
│  messages = messages[-20:]                                     │
│  Problem: Forgets early context                                │
│                                                                │
│  SUMMARIZE: Compress old messages                              │
│  summary = llm("Summarize: " + old_messages)                   │
│  messages = [{"role": "system", "content": summary}] + recent  │
│  Problem: Loses details                                        │
│                                                                │
│  RAG: Store in vector DB, retrieve relevant                    │
│  Best for: Long-term memory across sessions                    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## Context Window Limits

| Model | Context Window | ~Messages |
|-------|---------------|-----------|
| Claude Sonnet | 200K tokens | ~500 turns |
| GPT-4 | 128K tokens | ~300 turns |
| GPT-3.5 | 16K tokens | ~40 turns |

## Quick Implementation

```python
class Memory:
    def __init__(self, max_messages=50):
        self.messages = []
        self.max = max_messages

    def add(self, role, content):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max:
            self.messages = self.messages[-self.max:]

    def get(self):
        return self.messages
```

→ Full example: [../../quick-starts/00_your_first_agent.ipynb](../../quick-starts/00_your_first_agent.ipynb)
