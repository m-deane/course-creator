# Agent Communication: Message Passing and Protocols

> **Reading time:** ~10 min | **Module:** 5 — Multi-Agent Systems | **Prerequisites:** Module 5 — Multi-Agent Patterns

Multi-agent systems require structured communication. This guide covers message formats, communication protocols, and patterns for agents to share information, delegate tasks, and coordinate actions.

<div class="callout-insight">

**Insight:** Clear protocols prevent chaos. Without structured communication, agents talk past each other, duplicate work, or create deadlocks. Define message formats and interaction patterns explicitly.

</div>

---

## Message Structures

### Basic Message Format


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any
from enum import Enum


class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    INFORM = "inform"
    QUERY = "query"
    DELEGATE = "delegate"
    RESULT = "result"
    ERROR = "error"


@dataclass
class AgentMessage:
    sender: str
    recipient: str
    message_type: MessageType
    content: Any
    conversation_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    in_reply_to: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "type": self.message_type.value,
            "content": self.content,
            "conversation_id": self.conversation_id,
            "timestamp": self.timestamp.isoformat(),
            "in_reply_to": self.in_reply_to,
            "metadata": self.metadata
        }
```

</div>
</div>

### Typed Message Content

```python
@dataclass
class TaskRequest:
    task_description: str
    priority: int = 1
    deadline: Optional[datetime] = None
    context: dict = field(default_factory=dict)


@dataclass
class TaskResult:
    success: bool
    result: Any
    execution_time: float
    errors: list = field(default_factory=list)


@dataclass
class QueryMessage:
    question: str
    expected_format: Optional[str] = None
    constraints: dict = field(default_factory=dict)
```

---

## Communication Patterns

### Request-Response


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>

```python
class RequestResponseProtocol:
    """Simple request-response communication."""

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.pending_requests: dict[str, asyncio.Future] = {}

    async def send_request(
        self,
        sender: str,
        recipient: str,
        content: Any,
        router
    ) -> Any:
        """Send request and wait for response."""

        message = AgentMessage(
            sender=sender,
            recipient=recipient,
            message_type=MessageType.REQUEST,
            content=content,
            conversation_id=str(uuid.uuid4())
        )

        future = asyncio.Future()
        self.pending_requests[message.conversation_id] = future

        await router.send(message)

        try:
            response = await asyncio.wait_for(future, self.timeout)
            return response
        except asyncio.TimeoutError:
            del self.pending_requests[message.conversation_id]
            raise TimeoutError(f"No response from {recipient}")

    def handle_response(self, message: AgentMessage):
        """Handle incoming response."""
        if message.in_reply_to in self.pending_requests:
            future = self.pending_requests.pop(message.in_reply_to)
            future.set_result(message.content)
```

</div>
</div>

### Publish-Subscribe

```python
class PubSubBroker:
    """Publish-subscribe message broker."""

    def __init__(self):
        self.subscriptions: dict[str, list[str]] = {}  # topic -> [agents]
        self.agent_queues: dict[str, asyncio.Queue] = {}

    def subscribe(self, agent_id: str, topic: str):
        """Subscribe agent to topic."""
        if topic not in self.subscriptions:
            self.subscriptions[topic] = []
        if agent_id not in self.subscriptions[topic]:
            self.subscriptions[topic].append(agent_id)

        if agent_id not in self.agent_queues:
            self.agent_queues[agent_id] = asyncio.Queue()

    def unsubscribe(self, agent_id: str, topic: str):
        """Unsubscribe agent from topic."""
        if topic in self.subscriptions:
            self.subscriptions[topic] = [
                a for a in self.subscriptions[topic] if a != agent_id
            ]

    async def publish(self, topic: str, message: AgentMessage):
        """Publish message to all subscribers."""
        subscribers = self.subscriptions.get(topic, [])
        for agent_id in subscribers:
            if agent_id in self.agent_queues:
                await self.agent_queues[agent_id].put(message)

    async def receive(self, agent_id: str) -> AgentMessage:
        """Receive next message for agent."""
        if agent_id not in self.agent_queues:
            self.agent_queues[agent_id] = asyncio.Queue()
        return await self.agent_queues[agent_id].get()
```

### Blackboard (Shared Memory)

```python
class Blackboard:
    """Shared workspace for agent collaboration."""

    def __init__(self):
        self.data: dict[str, Any] = {}
        self.history: list[dict] = []
        self.watchers: dict[str, list[Callable]] = {}
        self.lock = asyncio.Lock()

    async def write(self, key: str, value: Any, author: str):
        """Write to blackboard."""
        async with self.lock:
            old_value = self.data.get(key)
            self.data[key] = value
            self.history.append({
                "action": "write",
                "key": key,
                "value": value,
                "author": author,
                "timestamp": datetime.utcnow()
            })

        # Notify watchers
        await self._notify_watchers(key, old_value, value)

    async def read(self, key: str) -> Any:
        """Read from blackboard."""
        return self.data.get(key)

    async def read_pattern(self, pattern: str) -> dict:
        """Read all keys matching pattern."""
        import re
        regex = re.compile(pattern)
        return {k: v for k, v in self.data.items() if regex.match(k)}

    def watch(self, key: str, callback: Callable):
        """Watch for changes to a key."""
        if key not in self.watchers:
            self.watchers[key] = []
        self.watchers[key].append(callback)

    async def _notify_watchers(self, key: str, old_value: Any, new_value: Any):
        """Notify watchers of changes."""
        for callback in self.watchers.get(key, []):
            await callback(key, old_value, new_value)
```

---

## Coordination Protocols

### Contract Net Protocol

Agents bid on tasks:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>

```python
class ContractNetManager:
    """Manager for Contract Net Protocol."""

    def __init__(self, agents: list[str], router):
        self.agents = agents
        self.router = router

    async def announce_task(self, task: TaskRequest) -> str:
        """Announce task and collect bids."""

        # Send call for proposals
        conversation_id = str(uuid.uuid4())
        bids = []

        for agent_id in self.agents:
            message = AgentMessage(
                sender="manager",
                recipient=agent_id,
                message_type=MessageType.REQUEST,
                content={"type": "call_for_proposals", "task": task},
                conversation_id=conversation_id
            )
            await self.router.send(message)

        # Collect bids (with timeout)
        deadline = datetime.utcnow() + timedelta(seconds=10)
        while datetime.utcnow() < deadline:
            try:
                response = await asyncio.wait_for(
                    self.router.receive("manager"),
                    timeout=1.0
                )
                if response.conversation_id == conversation_id:
                    bids.append({
                        "agent": response.sender,
                        "bid": response.content
                    })
            except asyncio.TimeoutError:
                continue

        # Select winner
        if not bids:
            raise ValueError("No bids received")

        winner = self._select_winner(bids)

        # Award contract
        await self._award_contract(winner, task, conversation_id)

        return winner["agent"]

    def _select_winner(self, bids: list[dict]) -> dict:
        """Select best bid (lowest cost, highest capability)."""
        return min(bids, key=lambda b: b["bid"].get("cost", float("inf")))

    async def _award_contract(self, winner: dict, task: TaskRequest, conv_id: str):
        """Award contract to winner, reject others."""
        # Award winner
        await self.router.send(AgentMessage(
            sender="manager",
            recipient=winner["agent"],
            message_type=MessageType.DELEGATE,
            content={"type": "award", "task": task},
            conversation_id=conv_id
        ))
```

</div>
</div>

### Consensus Protocol

Agents reach agreement:

```python
class ConsensusProtocol:
    """Simple voting-based consensus."""

    def __init__(self, agents: list[str], router):
        self.agents = agents
        self.router = router
        self.quorum = len(agents) // 2 + 1

    async def propose_and_vote(self, proposal: Any) -> dict:
        """Propose something and collect votes."""

        conversation_id = str(uuid.uuid4())
        votes = {"for": [], "against": []}

        # Send proposal to all agents
        for agent_id in self.agents:
            await self.router.send(AgentMessage(
                sender="coordinator",
                recipient=agent_id,
                message_type=MessageType.QUERY,
                content={"type": "vote_request", "proposal": proposal},
                conversation_id=conversation_id
            ))

        # Collect votes
        deadline = datetime.utcnow() + timedelta(seconds=15)
        while datetime.utcnow() < deadline and len(votes["for"]) + len(votes["against"]) < len(self.agents):
            try:
                response = await asyncio.wait_for(
                    self.router.receive("coordinator"),
                    timeout=1.0
                )
                if response.conversation_id == conversation_id:
                    vote = response.content.get("vote")
                    if vote == "for":
                        votes["for"].append(response.sender)
                    else:
                        votes["against"].append(response.sender)
            except asyncio.TimeoutError:
                continue

        # Determine outcome
        passed = len(votes["for"]) >= self.quorum

        return {
            "proposal": proposal,
            "passed": passed,
            "votes": votes,
            "quorum": self.quorum
        }
```

---

## Message Routing


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>

```python
class MessageRouter:
    """Central message router for multi-agent system."""

    def __init__(self):
        self.agents: dict[str, asyncio.Queue] = {}
        self.message_log: list[AgentMessage] = []

    def register_agent(self, agent_id: str):
        """Register an agent with the router."""
        self.agents[agent_id] = asyncio.Queue()

    async def send(self, message: AgentMessage):
        """Route message to recipient."""
        self.message_log.append(message)

        if message.recipient == "broadcast":
            for agent_id, queue in self.agents.items():
                if agent_id != message.sender:
                    await queue.put(message)
        elif message.recipient in self.agents:
            await self.agents[message.recipient].put(message)
        else:
            raise ValueError(f"Unknown recipient: {message.recipient}")

    async def receive(self, agent_id: str) -> AgentMessage:
        """Receive next message for agent."""
        return await self.agents[agent_id].get()

    def get_conversation(self, conversation_id: str) -> list[AgentMessage]:
        """Get all messages in a conversation."""
        return [m for m in self.message_log if m.conversation_id == conversation_id]
```


<div class="callout-key">

**Key Concept Summary:** This guide covered the core concepts. Review the companion slides for visual summaries and the hands-on notebook for practice implementations.


---

*Effective communication transforms independent agents into collaborative systems. Design your protocols as carefully as your individual agents.*


## Practice Questions

1. Explain in your own words how the concepts in this guide relate to building production agents.
2. What are the key tradeoffs you need to consider when applying these techniques?
3. Describe a scenario where the approach from this guide would be the wrong choice, and what you would use instead.

---

**Next Steps:**

<a class="link-card" href="./02_agent_communication_slides.md">
  <div class="link-card-title">Agent Communication Protocols — Companion Slides</div>
  <div class="link-card-description">Visual slide deck with diagrams, speaker notes, and key takeaways.</div>
</a>

<a class="link-card" href="../notebooks/01_orchestrator_agents.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with working code and guided exercises.</div>
</a>
