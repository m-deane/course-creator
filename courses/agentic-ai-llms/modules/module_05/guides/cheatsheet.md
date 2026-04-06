# Multi-Agent Systems Cheatsheet

> **Reading time:** ~5 min | **Module:** 5 — Multi-Agent Systems | **Prerequisites:** Module 5 guides

## Key Concepts

| Concept | Definition |
|---------|-----------|
| **Orchestrator** | Coordinator agent that assigns tasks and synthesizes results from specialized agents |
| **Supervisor Pattern** | One agent directs and manages other agents hierarchically |
| **Peer-to-Peer** | Agents collaborate as equals without central coordination |
| **Specialization** | Assigning specific roles and expertise domains to different agents |
| **Message Passing** | Communication mechanism where agents send structured messages to each other |
| **Shared Memory** | Common knowledge base or state that multiple agents can read and write |
| **Blackboard System** | Collaborative workspace where agents post partial solutions |
| **Consensus** | Agreement reached by multiple agents through voting or deliberation |
| **Agent Protocol** | Standardized format and rules for agent-to-agent communication |

## Common Patterns

### Basic Supervisor Pattern

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>
<div class="code-body">

```python
class Supervisor:
    def __init__(self, agents):
        self.agents = agents  # {role: agent}
        self.history = []

    def process_request(self, user_request):
        # Decide which agents to involve
        plan = self.create_plan(user_request)

        results = {}
        for step in plan:
            # Delegate to specialized agent
            agent = self.agents[step["role"]]
            context = self.build_context(step, results)
            result = agent.execute(step["task"], context)
            results[step["role"]] = result

        # Synthesize final response
        return self.synthesize(user_request, results)

    def create_plan(self, request):
        prompt = f"""
        Request: {request}
        Available agents: {list(self.agents.keys())}

        Create a step-by-step plan. Return JSON:
        [{{"role": "researcher", "task": "..."}}]
        """
        return llm.generate(prompt)
```

</div>
</div>

### Peer-to-Peer Collaboration

```python
class CollaborativeAgent:
    def __init__(self, name, expertise, peers):
        self.name = name
        self.expertise = expertise
        self.peers = peers
        self.inbox = []

    def process_task(self, task):
        # Try to solve alone
        if self.can_handle(task):
            return self.solve(task)

        # Ask peers for help
        responses = []
        for peer in self.peers:
            message = {
                "from": self.name,
                "request": task,
                "context": self.gather_context()
            }
            response = peer.receive_message(message)
            responses.append(response)

        # Synthesize peer responses
        return self.combine_perspectives(responses)

    def receive_message(self, message):
        self.inbox.append(message)

        if self.can_help(message["request"]):
            return {
                "from": self.name,
                "contribution": self.solve(message["request"]),
                "confidence": self.assess_confidence()
            }
        return None
```

### Shared Memory / Blackboard

```python
class Blackboard:
    def __init__(self):
        self.memory = {}
        self.lock = threading.Lock()

    def write(self, key, value, agent_id):
        with self.lock:
            if key not in self.memory:
                self.memory[key] = []
            self.memory[key].append({
                "value": value,
                "agent": agent_id,
                "timestamp": time.time()
            })

    def read(self, key):
        with self.lock:
            return self.memory.get(key, [])

class BlackboardAgent:
    def __init__(self, agent_id, blackboard):
        self.id = agent_id
        self.blackboard = blackboard

    def contribute(self, task):
        # Read current state
        current = self.blackboard.read("task_state")

        # Add contribution
        my_contribution = self.process(task, current)
        self.blackboard.write("task_state", my_contribution, self.id)

        # Check if complete
        all_contributions = self.blackboard.read("task_state")
        if self.is_complete(all_contributions):
            final = self.synthesize(all_contributions)
            self.blackboard.write("final_result", final, self.id)
            return final

        return "In progress"
```

### Consensus Building

```python
def debate_consensus(agents, question, rounds=3):
    proposals = []

    # Initial proposals
    for agent in agents:
        proposal = agent.propose(question)
        proposals.append({
            "agent": agent.name,
            "proposal": proposal,
            "votes": 0
        })

    # Debate rounds
    for round_num in range(rounds):
        # Each agent critiques others' proposals
        for agent in agents:
            critiques = agent.critique(proposals)

            # Update proposals based on critique
            proposals = update_proposals(proposals, critiques)

    # Final voting
    for agent in agents:
        vote = agent.vote(proposals)
        proposals[vote]["votes"] += 1

    # Return consensus
    winner = max(proposals, key=lambda p: p["votes"])
    return winner["proposal"]
```

### Agent Team with Role Specialization

```python
class AgentTeam:
    def __init__(self):
        self.agents = {
            "researcher": ResearchAgent(
                role="researcher",
                tools=["web_search", "arxiv", "wikipedia"]
            ),
            "coder": CodingAgent(
                role="coder",
                tools=["python_repl", "code_search"]
            ),
            "tester": TestingAgent(
                role="tester",
                tools=["pytest", "coverage"]
            ),
            "reviewer": ReviewAgent(
                role="reviewer",
                tools=["linter", "security_scan"]
            )
        }

    def build_feature(self, requirement):
        # Research phase
        research = self.agents["researcher"].investigate(requirement)

        # Implementation phase
        code = self.agents["coder"].implement(requirement, research)

        # Testing phase
        tests = self.agents["tester"].create_tests(code)
        test_results = self.agents["tester"].run_tests(tests)

        # Review phase
        if not test_results["passed"]:
            code = self.agents["coder"].fix(code, test_results)

        review = self.agents["reviewer"].review(code, tests)

        return {
            "code": code,
            "tests": tests,
            "research": research,
            "review": review
        }
```

## Gotchas

### Problem: Agents talking in circles without progress
**Symptom:** Peer-to-peer agents endlessly debate without reaching conclusion
**Solution:**
- Set maximum interaction rounds
- Use a supervisor to detect stalemates
- Implement tie-breaking mechanisms
- Track progress metrics (new information per round)

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>
<div class="code-body">

```python
# Bad: Unlimited debate
while not consensus:
    for agent in agents:
        agent.respond_to_others()

# Good: Limited rounds with progress tracking
max_rounds = 5
for round in range(max_rounds):
    new_info = False
    for agent in agents:
        response = agent.respond_to_others()
        if is_novel(response):
            new_info = True

    if not new_info:
        break  # Converged or stuck
```

</div>
</div>

### Problem: Race conditions in shared memory
**Symptom:** Agents overwrite each other's updates or read stale data
**Solution:**
- Use proper locking mechanisms
- Implement optimistic concurrency control
- Use append-only logs instead of mutable state
- Add timestamps and version numbers

### Problem: Orchestrator becomes bottleneck
**Symptom:** All agents wait for orchestrator decisions, high latency
**Solution:**
- Allow agents to work independently when possible
- Use async/parallel execution for independent subtasks
- Implement hierarchical orchestration (multiple supervisors)
- Cache common orchestrator decisions

### Problem: Agent specialization too narrow
**Symptom:** Tasks fall through cracks, no agent can handle them
**Solution:**
- Have a "generalist" fallback agent
- Allow agents to expand capabilities dynamically
- Implement task routing based on confidence scores
- Use meta-agent to assign tasks to appropriate specialists

### Problem: Consensus takes too long
**Symptom:** Multiple rounds of debate significantly increase latency and cost
**Solution:**
- Use voting instead of full deliberation for simple decisions
- Implement early stopping when strong majority reached
- Use weighted voting (expert agents have more influence)
- Cache consensus for repeated questions

### Problem: Message passing overhead
**Symptom:** Agents spend more time communicating than working
**Solution:**
- Batch messages when possible
- Use async communication (don't wait for immediate reply)
- Implement message priorities
- Reduce protocol verbosity

```python
# Bad: Synchronous one-by-one messaging
for peer in peers:
    response = peer.send_and_wait(message)
    process(response)

# Good: Async batch messaging
futures = [peer.send_async(message) for peer in peers]
responses = await asyncio.gather(*futures)
```

### Problem: Context explosion in agent conversations
**Symptom:** Agent prompts grow unbounded with full conversation history
**Solution:**
- Summarize old messages
- Use semantic compression
- Keep only relevant context per agent
- Implement "working memory" vs "long-term memory"

## Quick Decision Guide

**When to use Supervisor pattern?**
- Clear task decomposition possible
- Agents have distinct specializations
- Need centralized control and monitoring

**When to use Peer-to-Peer?**
- Problem requires diverse perspectives
- No clear hierarchy in expertise
- Need creative collaboration and debate

**When to use Shared Memory?**
- Agents build on each other's work
- Need to avoid duplicate effort
- Complex state needs coordination

**When to use Market-Based?**
- Resource allocation is key concern
- Agents have varying costs/capabilities
- Need to optimize for efficiency

**When to NOT use multi-agent?**
- Task is simple enough for single agent
- Coordination overhead exceeds benefits
- Real-time response required (latency sensitive)
- Budget is limited (multiple LLM calls are expensive)
