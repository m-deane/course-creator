# The Closed Loop: The Mental Model for Modern AI Engineering

## In Brief

The closed loop is the core mental model for building production LLM systems. It describes how an AI system receives goals, builds context, generates plans, takes actions, observes results, updates memory, and evaluates progress—repeatedly until the goal is achieved.

## Key Insight

**A chatbot answers questions. A system achieves goals.**

The difference is the loop: generating → observing → learning → improving → generating again.

## Visual Explanation

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          THE CLOSED LOOP                                 │
└──────────────────────────────────────────────────────────────────────────┘

                              ┌─────────┐
                              │  GOAL   │
                              │ (input) │
                              └────┬────┘
                                   │
                   ┌───────────────┼───────────────┐
                   │               ▼               │
                   │        ┌───────────┐          │
                   │        │  CONTEXT  │          │
                   │        │  BUILDER  │          │
                   │        └─────┬─────┘          │
                   │              │                │
                   │    ┌─────────┴─────────┐      │
                   │    │                   │      │
                   │    ▼                   ▼      │
              ┌────┴────┐             ┌─────┴────┐ │
              │ MEMORY  │             │ RETRIEVAL│ │
              │ (state) │             │  (docs)  │ │
              └────┬────┘             └─────┬────┘ │
                   │                        │      │
                   └─────────┬──────────────┘      │
                             ▼                     │
                      ┌────────────┐               │
                      │   PLAN /   │               │
                      │  GENERATE  │               │
                      └──────┬─────┘               │
                             │                     │
              ┌──────────────┼──────────────┐      │
              │              │              │      │
              ▼              ▼              ▼      │
        ┌─────────┐    ┌─────────┐    ┌─────────┐  │
        │  TEXT   │    │  TOOL   │    │  CODE   │  │
        │ OUTPUT  │    │  CALL   │    │  EXEC   │  │
        └────┬────┘    └────┬────┘    └────┬────┘  │
              │              │              │      │
              └──────────────┼──────────────┘      │
                             ▼                     │
                      ┌────────────┐               │
                      │  OBSERVE   │               │
                      │  RESULTS   │               │
                      └──────┬─────┘               │
                             │                     │
                             ▼                     │
                      ┌────────────┐               │
                      │  EVALUATE  │               │
                      │            │               │
                      └──────┬─────┘               │
                             │                     │
              ┌──────────────┼──────────────┐      │
              │              │              │      │
              ▼              ▼              ▼      │
        ┌─────────┐    ┌─────────┐    ┌─────────┐  │
        │ SUCCESS │    │ UPDATE  │    │  RETRY  │──┘
        │ (done)  │    │ MEMORY  │    │  (loop) │
        └─────────┘    └────┬────┘    └─────────┘
                            │
                            └──────► Store for future interactions
```

## The Seven Stages

### Stage 1: Goal Interpretation

```
Input:  "Book me a table for 4 at an Italian restaurant tomorrow at 7pm"

What the system must understand:
- Task type: Reservation booking
- Constraints: 4 people, Italian cuisine, tomorrow, 7pm
- Success criteria: Confirmed reservation
- Implicit: User's location, preferences, budget
```

**Key capability:** Parse natural language into structured intent.

### Stage 2: Context Building

```python
def build_context(goal, memory, retriever):
    """Assemble everything the model needs to know."""

    context = {
        # From memory
        "user_preferences": memory.get("user_preferences"),
        "past_bookings": memory.get("recent_restaurants"),
        "conversation_history": memory.get("current_session"),

        # From retrieval
        "relevant_restaurants": retriever.search(
            "Italian restaurants near user",
            filters={"rating": ">4", "accepts_reservations": True}
        ),

        # From tools
        "availability": check_availability(
            restaurants=relevant_restaurants,
            date="tomorrow",
            time="7pm",
            party_size=4
        )
    }

    return context
```

**Key capability:** Combine memory, retrieval, and real-time data.

### Stage 3: Plan / Generate

The model uses context to decide what to do:

```
Option A: Enough information → Generate response
Option B: Need more info → Ask clarifying question
Option C: Need external data → Plan tool calls
Option D: Complex task → Break into subtasks
```

**ReAct Pattern Example:**
```
Thought: I need to find Italian restaurants with availability tomorrow at 7pm.
Action: search_restaurants(cuisine="Italian", date="2024-12-21", time="19:00", party=4)
Observation: Found 3 restaurants with availability: Bella Italia, Trattoria Roma, Casa Napoli
Thought: I should check the user's past preferences.
Action: get_user_preferences(category="restaurants")
Observation: User prefers quiet ambiance, has visited Trattoria Roma before (rated 5 stars)
Thought: Trattoria Roma matches preferences and has availability. I'll recommend it.
Action: make_reservation(restaurant="Trattoria Roma", date="2024-12-21", time="19:00", party=4)
Observation: Reservation confirmed. Confirmation #TR-78234
Thought: Task complete. I'll summarize for the user.
```

### Stage 4: Act (Execute Tools)

```python
class ToolExecutor:
    def execute(self, action: ToolCall) -> ToolResult:
        # Validate the call
        if not self.is_valid(action):
            return ToolResult(error="Invalid parameters")

        # Execute with timeout and retry
        try:
            result = self.tools[action.name].run(
                **action.parameters,
                timeout=30
            )
            return ToolResult(success=True, data=result)

        except TimeoutError:
            return ToolResult(error="Tool timed out", retry=True)

        except ToolError as e:
            return ToolResult(error=str(e), retry=e.is_retryable)
```

**Key capability:** Reliable tool execution with error handling.

### Stage 5: Observe Results

```python
def observe(action_result, expected_outcome):
    """Process the result of an action."""

    observation = {
        "success": action_result.success,
        "data": action_result.data,
        "matches_expectation": validate(action_result, expected_outcome),
        "side_effects": detect_side_effects(action_result),
        "next_steps": infer_next_steps(action_result)
    }

    return observation
```

**Key capability:** Interpret results and detect anomalies.

### Stage 6: Evaluate

```python
def evaluate(goal, observations, constraints):
    """Determine if we've succeeded and what to do next."""

    # Check goal completion
    if goal_achieved(goal, observations):
        return Decision(status="complete", confidence=0.95)

    # Check for blockers
    if unrecoverable_error(observations):
        return Decision(status="failed", reason=observations.error)

    # Check for progress
    if making_progress(observations):
        return Decision(status="continue", next_action=plan_next_step())

    # Stuck - need different approach
    return Decision(status="retry", strategy="alternative_approach")
```

**Key capability:** Judge success, detect failure, decide next move.

### Stage 7: Update Memory

```python
def update_memory(memory, interaction):
    """Store useful information for future interactions."""

    # Short-term: Current conversation
    memory.conversation.append(interaction)

    # Working memory: Task-relevant state
    if interaction.has_useful_facts:
        memory.working.update(interaction.extracted_facts)

    # Long-term: Persistent knowledge
    if interaction.is_significant:
        memory.long_term.store(
            content=interaction.summary,
            embedding=embed(interaction),
            metadata={"timestamp": now(), "type": interaction.type}
        )

    # Decay: Remove stale information
    memory.decay_old_entries(threshold=0.3)
```

**Key capability:** Selective storage and retrieval.

## Loop Characteristics

### Loops Can Be Nested

```
Outer loop: Complete user's project (hours/days)
  └── Inner loop: Complete current task (minutes)
        └── Micro loop: Execute tool call (seconds)
```

### Loops Can Run in Parallel

```
Main agent: Coordinate overall task
  ├── Research agent: Gather information
  ├── Execution agent: Take actions
  └── Verification agent: Check results
```

### Loops Must Be Bounded

```python
MAX_ITERATIONS = 10
TIMEOUT_SECONDS = 300

for iteration in range(MAX_ITERATIONS):
    if time_elapsed > TIMEOUT_SECONDS:
        return graceful_failure("Timeout reached")

    result = run_one_iteration()

    if result.is_complete:
        return result

    if result.is_stuck:
        try_alternative_approach()

return graceful_failure("Max iterations reached")
```

## The Closed-Loop Advantage

| Open Loop (Chatbot) | Closed Loop (System) |
|---------------------|----------------------|
| One-shot generation | Iterative refinement |
| Hopes for correctness | Verifies results |
| Forgets immediately | Learns from interactions |
| Fails silently | Detects and recovers |
| Static behavior | Improves over time |

## Common Pitfalls

### Pitfall 1: Infinite Loops
```
Problem: Agent keeps trying the same failing approach.
Solution: Track attempted strategies, force alternatives after N failures.
```

### Pitfall 2: Goal Drift
```
Problem: Agent solves a different problem than requested.
Solution: Periodically re-check alignment with original goal.
```

### Pitfall 3: Memory Bloat
```
Problem: Storing everything fills context and slows retrieval.
Solution: Selective storage, summarization, decay policies.
```

## Implementation Skeleton

```python
class ClosedLoopAgent:
    def __init__(self):
        self.model = LLM()
        self.memory = MemoryManager()
        self.tools = ToolRegistry()
        self.evaluator = Evaluator()

    def run(self, goal: str, max_iterations: int = 10) -> Result:
        for i in range(max_iterations):
            # Build context
            context = self.build_context(goal)

            # Generate plan/action
            action = self.model.generate(goal, context)

            # Execute
            if action.type == "tool_call":
                result = self.tools.execute(action)
            else:
                result = action.text

            # Observe and evaluate
            observation = self.observe(result)
            evaluation = self.evaluator.evaluate(goal, observation)

            # Update memory
            self.memory.update(goal, action, observation)

            # Check completion
            if evaluation.is_complete:
                return Result(success=True, output=result)

            if evaluation.is_failed:
                return Result(success=False, error=evaluation.reason)

        return Result(success=False, error="Max iterations reached")
```

## Connections

- **Builds on:** Understanding that LLMs are systems, not just models
- **Leads to:** Memory systems (Module 03), Tool use (Module 04), Evaluation (Module 07)

## Practice Problems

1. **Trace a loop:** Given the goal "Find the weather in Tokyo and send it to my email," trace through all 7 stages of the loop.

2. **Design evaluation:** What criteria would you use to evaluate if an agent successfully "summarized a research paper"?

3. **Handle failure:** An agent is trying to book a flight but the API keeps timing out. Design the retry and escalation logic.

## Further Reading

- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)
- Module 04 for detailed tool use patterns
