# Multi-Agent Patterns: Orchestration Architectures

> **Reading time:** ~12 min | **Module:** 5 — Multi-Agent Systems | **Prerequisites:** Module 4 — Agentic Patterns

Multi-agent systems divide complex tasks among specialized agents. Different orchestration patterns suit different problems—supervisor patterns for delegation, peer-to-peer for collaboration, hierarchical for scale.

<div class="callout-insight">

**Insight:** The right architecture matches problem structure. Supervisor patterns work for clear delegation. Peer networks suit collaborative exploration. Choose based on your task decomposition, not technology preferences.

</div>

---

## Pattern 1: Supervisor (Hub and Spoke)

One orchestrator delegates to specialized workers:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>
<div class="code-body">

```python
import anthropic
from dataclasses import dataclass
from typing import Callable


@dataclass
class WorkerAgent:
    name: str
    description: str
    handler: Callable


class SupervisorAgent:
    """Orchestrator that delegates to specialized workers."""

    def __init__(self, workers: list[WorkerAgent]):
        self.client = anthropic.Anthropic()
        self.workers = {w.name: w for w in workers}

    def run(self, task: str) -> str:
        """Process task by delegating to appropriate workers."""

        # Decide which worker(s) to use
        delegation = self._decide_delegation(task)

        # Execute delegated tasks
        results = {}
        for worker_name, subtask in delegation.items():
            worker = self.workers.get(worker_name)
            if worker:
                results[worker_name] = worker.handler(subtask)

        # Synthesize final response
        return self._synthesize(task, results)

    def _decide_delegation(self, task: str) -> dict[str, str]:
        """Decide which workers to use and what to delegate."""

        worker_list = "\n".join(
            f"- {w.name}: {w.description}"
            for w in self.workers.values()
        )

        prompt = f"""You are a task coordinator with these specialized workers:
{worker_list}

Given this task:
{task}

Decide which worker(s) should handle parts of this task.
Return as JSON:
{{
    "worker_name": "specific subtask for this worker",
    ...
}}

Only include workers that are needed. Be specific about each subtask."""

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        import json
        return json.loads(response.content[0].text)

    def _synthesize(self, original_task: str, results: dict) -> str:
        """Combine worker results into final response."""

        results_text = "\n".join(
            f"**{name}**: {result}" for name, result in results.items()
        )

        prompt = f"""Original task: {original_task}

Results from specialized agents:
{results_text}

Synthesize these results into a coherent final response."""

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text


# Usage
def research_handler(query: str) -> str:
    # Research agent logic
    return f"Research findings on: {query}"

def code_handler(spec: str) -> str:
    # Code agent logic
    return f"Code implementation for: {spec}"

def review_handler(code: str) -> str:
    # Review agent logic
    return f"Review of: {code}"


supervisor = SupervisorAgent([
    WorkerAgent("researcher", "Searches and analyzes information", research_handler),
    WorkerAgent("developer", "Writes and implements code", code_handler),
    WorkerAgent("reviewer", "Reviews code and provides feedback", review_handler),
])

result = supervisor.run("Research Python async patterns and implement an example")
```

</div>
</div>

---

## Pattern 2: Peer-to-Peer (Collaborative)

Agents work together without central control:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>
<div class="code-body">

```python
import asyncio
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Message:
    sender: str
    content: str
    message_type: str = "general"  # general, request, response


@dataclass
class PeerAgent:
    name: str
    expertise: str
    client: anthropic.Anthropic = field(default_factory=anthropic.Anthropic)
    mailbox: list[Message] = field(default_factory=list)

    def receive(self, message: Message):
        """Receive a message from another agent."""
        self.mailbox.append(message)

    def process_messages(self) -> list[Message]:
        """Process all messages and generate responses."""
        responses = []

        for msg in self.mailbox:
            if msg.message_type == "request":
                response = self._handle_request(msg)
                responses.append(Message(
                    sender=self.name,
                    content=response,
                    message_type="response"
                ))

        self.mailbox.clear()
        return responses

    def _handle_request(self, request: Message) -> str:
        """Handle a request message."""
        prompt = f"""You are {self.name}, an expert in {self.expertise}.

{request.sender} sent you this request:
{request.content}

Provide your expert input."""

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text


class PeerNetwork:
    """Coordinate peer-to-peer agent collaboration."""

    def __init__(self, agents: list[PeerAgent]):
        self.agents = {a.name: a for a in agents}

    def broadcast(self, sender: str, content: str, message_type: str = "general"):
        """Send message to all agents except sender."""
        msg = Message(sender=sender, content=content, message_type=message_type)
        for name, agent in self.agents.items():
            if name != sender:
                agent.receive(msg)

    def send(self, sender: str, recipient: str, content: str, message_type: str = "general"):
        """Send message to specific agent."""
        msg = Message(sender=sender, content=content, message_type=message_type)
        self.agents[recipient].receive(msg)

    async def run_collaboration(self, task: str, max_rounds: int = 5) -> dict:
        """Run a collaborative session."""

        # Broadcast initial task
        self.broadcast("coordinator", f"Task: {task}", "request")

        all_responses = []

        for round_num in range(max_rounds):
            round_responses = []

            # Each agent processes messages
            for agent in self.agents.values():
                responses = agent.process_messages()
                round_responses.extend(responses)

                # Share responses with network
                for resp in responses:
                    self.broadcast(resp.sender, resp.content)

            all_responses.extend(round_responses)

            # Check if consensus reached
            if self._check_consensus(round_responses):
                break

        return {
            "rounds": round_num + 1,
            "responses": all_responses
        }

    def _check_consensus(self, responses: list[Message]) -> bool:
        """Check if agents have reached consensus."""
        # Simple heuristic: no new substantive responses
        return len(responses) == 0
```

</div>
</div>

---

## Pattern 3: Hierarchical

Nested management for complex organizational tasks:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>
<div class="code-body">

```python
@dataclass
class HierarchicalAgent:
    name: str
    role: str
    subordinates: list["HierarchicalAgent"] = field(default_factory=list)
    supervisor: Optional["HierarchicalAgent"] = None

    def assign_task(self, task: str) -> dict:
        """Assign task to self or delegate to subordinates."""

        if not self.subordinates:
            # Leaf node - execute directly
            return {"executor": self.name, "result": self._execute(task)}

        # Decompose and delegate
        subtasks = self._decompose(task)

        results = {}
        for subtask, subordinate in zip(subtasks, self.subordinates):
            results[subordinate.name] = subordinate.assign_task(subtask)

        # Aggregate results
        return {
            "coordinator": self.name,
            "delegated_to": list(results.keys()),
            "results": results
        }

    def _decompose(self, task: str) -> list[str]:
        """Break task into subtasks for subordinates."""
        client = anthropic.Anthropic()

        prompt = f"""As {self.role}, break this task into {len(self.subordinates)} subtasks:
Task: {task}

Subordinates:
{chr(10).join(f"- {s.name} ({s.role})" for s in self.subordinates)}

Return one subtask per line, matched to each subordinate."""

        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text.strip().split('\n')

    def _execute(self, task: str) -> str:
        """Execute task directly."""
        client = anthropic.Anthropic()

        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            messages=[{"role": "user", "content": f"As {self.role}, complete: {task}"}]
        )

        return response.content[0].text


# Build hierarchy
ceo = HierarchicalAgent("CEO", "Chief Executive")
cto = HierarchicalAgent("CTO", "Technical Lead")
cmo = HierarchicalAgent("CMO", "Marketing Lead")

dev1 = HierarchicalAgent("Dev1", "Backend Developer")
dev2 = HierarchicalAgent("Dev2", "Frontend Developer")
mktr = HierarchicalAgent("Marketer", "Content Creator")

cto.subordinates = [dev1, dev2]
cmo.subordinates = [mktr]
ceo.subordinates = [cto, cmo]

# Execute hierarchically
result = ceo.assign_task("Launch a new product feature")
```

</div>
</div>

---

## Pattern 4: Pipeline (Sequential)

Agents form a processing chain:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>
<div class="code-body">

```python
class PipelineStage:
    """Single stage in an agent pipeline."""

    def __init__(self, name: str, system_prompt: str, next_stage: Optional["PipelineStage"] = None):
        self.name = name
        self.system_prompt = system_prompt
        self.next_stage = next_stage
        self.client = anthropic.Anthropic()

    def process(self, input_data: dict) -> dict:
        """Process input and pass to next stage."""

        # Process at this stage
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            system=self.system_prompt,
            messages=[{"role": "user", "content": str(input_data)}]
        )

        output = {
            "stage": self.name,
            "input": input_data,
            "output": response.content[0].text
        }

        # Pass to next stage if exists
        if self.next_stage:
            output["next"] = self.next_stage.process({
                "previous_stage": self.name,
                "data": output["output"]
            })

        return output


# Build pipeline
reviewer = PipelineStage(
    "Reviewer",
    "Review the content for errors and suggest improvements."
)

editor = PipelineStage(
    "Editor",
    "Edit the content based on review feedback.",
    next_stage=reviewer
)

writer = PipelineStage(
    "Writer",
    "Write content based on the requirements.",
    next_stage=editor
)

# Process through pipeline
result = writer.process({"topic": "Introduction to AI Agents"})
```

</div>
</div>

---

## Pattern 5: Debate and Consensus

Agents argue positions and reach agreement:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>
<div class="code-body">

```python
class DebateAgent:
    """Agent that takes a position and argues for it."""

    def __init__(self, name: str, position: str):
        self.name = name
        self.position = position
        self.client = anthropic.Anthropic()

    def argue(self, topic: str, opposing_arguments: list[str] = None) -> str:
        """Generate argument for position."""

        context = ""
        if opposing_arguments:
            context = f"\n\nOpposing arguments to address:\n"
            context += "\n".join(f"- {arg}" for arg in opposing_arguments)

        prompt = f"""You are arguing for: {self.position}

Topic: {topic}
{context}

Make your best argument. Be specific and persuasive."""

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text


class DebateModerator:
    """Moderator that runs debates and synthesizes conclusions."""

    def __init__(self, debaters: list[DebateAgent]):
        self.debaters = debaters
        self.client = anthropic.Anthropic()

    def run_debate(self, topic: str, rounds: int = 3) -> dict:
        """Run a multi-round debate."""

        arguments = {d.name: [] for d in self.debaters}

        for round_num in range(rounds):
            for debater in self.debaters:
                # Get opposing arguments from previous round
                opposing = []
                for other in self.debaters:
                    if other.name != debater.name and arguments[other.name]:
                        opposing.append(arguments[other.name][-1])

                argument = debater.argue(topic, opposing if round_num > 0 else None)
                arguments[debater.name].append(argument)

        # Synthesize conclusion
        conclusion = self._synthesize(topic, arguments)

        return {
            "topic": topic,
            "rounds": arguments,
            "conclusion": conclusion
        }

    def _synthesize(self, topic: str, arguments: dict) -> str:
        """Synthesize a balanced conclusion."""

        all_args = ""
        for name, args in arguments.items():
            all_args += f"\n\n{name}'s arguments:\n"
            all_args += "\n---\n".join(args)

        prompt = f"""Topic: {topic}

Arguments presented:
{all_args}

As a neutral moderator, synthesize a balanced conclusion that:
1. Acknowledges valid points from all sides
2. Identifies areas of agreement
3. Recommends a nuanced position"""

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text


# Usage
optimist = DebateAgent("Optimist", "AI will primarily benefit humanity")
pessimist = DebateAgent("Pessimist", "AI poses significant risks")
pragmatist = DebateAgent("Pragmatist", "AI outcomes depend on implementation")

moderator = DebateModerator([optimist, pessimist, pragmatist])
debate_result = moderator.run_debate("The impact of AI on employment")
```

</div>
</div>

---

## Choosing the Right Pattern

| Scenario | Recommended Pattern |
|----------|-------------------|
| Clear task delegation | Supervisor |
| Creative collaboration | Peer-to-Peer |
| Large-scale organization | Hierarchical |
| Sequential processing | Pipeline |
| Decision-making | Debate/Consensus |
| Dynamic workloads | Market-Based |

<div class="callout-key">

**Key Concept Summary:** This guide covered the core concepts. Review the companion slides for visual summaries and the hands-on notebook for practice implementations.

</div>

---

*Multi-agent systems multiply capabilities. Choose patterns that match your problem structure and let agents collaborate effectively.*


## Practice Questions

1. Explain in your own words how the concepts in this guide relate to building production agents.
2. What are the key tradeoffs you need to consider when applying these techniques?
3. Describe a scenario where the approach from this guide would be the wrong choice, and what you would use instead.

---

**Next Steps:**

<a class="link-card" href="./01_multi_agent_patterns_slides.md">
  <div class="link-card-title">Multi-Agent Architecture Patterns — Companion Slides</div>
  <div class="link-card-description">Visual slide deck with diagrams, speaker notes, and key takeaways.</div>
</a>

<a class="link-card" href="../notebooks/01_orchestrator_agents.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with working code and guided exercises.</div>
</a>
