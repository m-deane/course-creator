# Quiz: Module 5 - Multi-Agent Systems

**Estimated Time:** 20 minutes
**Total Points:** 100
**Passing Score:** 70%

---

## Instructions

This quiz evaluates your understanding of multi-agent architectures, orchestration patterns, agent communication, and specialization strategies. Provide thoughtful, practical answers.

---

## Part A: Multi-Agent Patterns (35 points)

### Question 1 (10 points)
Describe four different multi-agent patterns and when each is most appropriate. Include: Supervisor, Peer-to-Peer, Hierarchical, and one other.

**Your Answer:**

---

### Question 2 (8 points)
What is the primary advantage of using multiple specialized agents instead of a single general-purpose agent?

A) It's always faster
B) It uses fewer tokens
C) Each agent can be optimized for its specific domain, improving overall performance
D) It's easier to implement

**Your Answer:**

---

### Question 3 (12 points)
Design a multi-agent architecture for a software development team assistant. Identify at least 3 specialized agents, their roles, and how they should coordinate.

**Your Answer:**

---

### Question 4 (5 points)
**True or False:** In a peer-to-peer multi-agent system, agents can communicate directly without a central coordinator.

**Your Answer:**

---

## Part B: Agent Communication (30 points)

### Question 5 (10 points)
Compare three different communication patterns for multi-agent systems:
- Direct messaging
- Shared memory
- Broadcast

For each, provide one advantage and one disadvantage.

**Your Answer:**

---

### Question 6 (8 points)
What information should be included in a message passed between agents to ensure effective coordination? List at least 4 key components.

**Your Answer:**

---

### Question 7 (12 points)
You have a research agent that finds information and a writing agent that creates content. Design a message protocol for them to collaborate on writing a blog post. Show the message flow with at least 3 messages.

**Your Answer:**

---

## Part C: Specialization (25 points)

### Question 8 (10 points)
Explain role-based agent design. What are three ways to implement specialization in an agent?

**Your Answer:**

---

### Question 9 (8 points)
You're building a multi-agent system for customer support. Which specialization strategy is most appropriate?

A) One generalist agent handles everything
B) Agents specialized by function (triage, technical support, billing, escalation)
C) Agents specialized by customer tier (free, paid, enterprise)
D) Randomly distribute queries across identical agents

**Your Answer:**

---

### Question 10 (7 points)
What is the trade-off between agent specialization depth and system flexibility?

**Your Answer:**

---

## Part D: Coordination & Consensus (10 points)

### Question 11 (5 points)
In a multi-agent debate system, what is the purpose of having agents argue different positions before reaching a consensus?

**Your Answer:**

---

### Question 12 (5 points)
How can you prevent deadlock in a multi-agent system where agents depend on each other's outputs?

**Your Answer:**

---

## Part E: Orchestration (Bonus - 10 points)

### Question 13 (10 points)
Design an orchestration flow for this scenario:

**Task:** Generate a technical article with code examples, diagrams, and peer review

**Available Agents:**
- Research Agent (finds sources)
- Code Agent (writes/tests code)
- Writing Agent (creates content)
- Review Agent (provides critique)
- Diagram Agent (creates visualizations)

Show the orchestration sequence, including any parallel steps and dependencies.

**Your Answer:**

---

# Answer Key

## Part A: Multi-Agent Patterns

**Question 1:** (10 points)
*Sample Answer:*

**Supervisor Pattern:** One orchestrator agent delegates to specialized worker agents. Best for: Clear task delegation with central coordination (e.g., project management).

**Peer-to-Peer:** Agents collaborate as equals without hierarchy. Best for: Collaborative problem-solving where no single agent should dominate (e.g., brainstorming, debate).

**Hierarchical:** Nested management layers with supervisors managing supervisors. Best for: Complex organizations requiring multiple levels of coordination (e.g., enterprise automation).

**Market-Based:** Agents bid on tasks based on capability/availability. Best for: Resource optimization and load balancing across dynamic agent pools.

*Grading:* 2.5 points per pattern (description + use case)

**Question 2:** Answer: C (8 points)
*Explanation:* Specialized agents can have tailored system prompts, tools, and even different models optimized for their domain, resulting in better overall performance than a jack-of-all-trades agent.

**Question 3:** (12 points)
*Sample Answer:*

**Agents:**
1. **Planning Agent** - Breaks down feature requests into technical tasks, creates specifications
2. **Code Agent** - Implements features, writes code following best practices
3. **Testing Agent** - Writes and runs tests, identifies bugs
4. **Review Agent** - Reviews code for quality, security, and maintainability

**Coordination:**
- Supervisor orchestrator receives user request
- Planning Agent creates spec → sends to Code Agent
- Code Agent implements → sends code to Testing Agent
- Testing Agent runs tests → if bugs found, sends back to Code Agent
- Once tests pass → Review Agent provides feedback
- Iterate until Review approves → return to user

*Grading:*
- 3+ specialized agents with clear roles (6 pts)
- Logical coordination flow (4 pts)
- Dependencies and iteration (2 pts)

**Question 4:** True (5 points)
*Explanation:* Peer-to-peer systems allow direct agent-to-agent communication without requiring a central coordinator, though this increases complexity in managing coordination.

## Part B: Agent Communication

**Question 5:** (10 points)
*Sample Answer:*

**Direct Messaging:**
- Advantage: Explicit, traceable communication flow
- Disadvantage: Requires agents to know about each other, tight coupling

**Shared Memory:**
- Advantage: Decouples agents, any agent can access shared state
- Disadvantage: Race conditions, harder to debug, requires synchronization

**Broadcast:**
- Advantage: One-to-many communication, easy to add new listeners
- Disadvantage: All agents receive all messages, potential noise/irrelevant info

*Grading:* 3.3 points per pattern (both advantage and disadvantage needed)

**Question 6:** (8 points)
*Sample Answer:*
1. **Sender ID** - Which agent sent the message
2. **Recipient ID** - Target agent (or "broadcast")
3. **Message Type** - Purpose (request, response, notification, etc.)
4. **Payload** - Actual content/data
5. **Context/Session ID** - Link to broader task/conversation
6. **Timestamp** - When message was sent
7. **Priority** - Urgency level (optional)

*Grading:* 2 points each for 4 valid components

**Question 7:** (12 points)
*Sample Answer:*

```
Message 1: Orchestrator → Research Agent
{
  "from": "orchestrator",
  "to": "research_agent",
  "type": "task_request",
  "task": "research_topic",
  "payload": {
    "topic": "Introduction to Transformer Architecture",
    "required_sources": 5,
    "focus": "beginner-friendly explanations"
  }
}

Message 2: Research Agent → Writing Agent
{
  "from": "research_agent",
  "to": "writing_agent",
  "type": "research_complete",
  "payload": {
    "sources": [...],
    "key_points": [...],
    "recommended_structure": [...]
  }
}

Message 3: Writing Agent → Orchestrator
{
  "from": "writing_agent",
  "to": "orchestrator",
  "type": "draft_complete",
  "payload": {
    "content": "...",
    "word_count": 1500,
    "sources_cited": [...]
  }
}
```

*Grading:*
- 3+ messages shown (4 pts)
- Proper message structure (4 pts)
- Logical collaboration flow (4 pts)

## Part C: Specialization

**Question 8:** (10 points)
*Sample Answer:* Role-based agent design assigns specific responsibilities and expertise to each agent. Implementation methods:

1. **System Prompt Specialization** - Tailor system prompt to define expertise, constraints, and behavior for specific role
2. **Tool Specialization** - Provide role-specific tools (e.g., data agent gets database tools, code agent gets compiler)
3. **Model Selection** - Use different models for different roles (e.g., faster model for simple tasks, advanced model for reasoning)
4. **Knowledge Base Specialization** - Each agent has access to domain-specific RAG/knowledge

*Grading:* 2.5 points for explaining concept, 2.5 points each for 3 methods

**Question 9:** Answer: B (8 points)
*Explanation:* Functional specialization (triage, technical, billing) allows each agent to develop deep expertise in its domain, use specialized tools, and provide better service than a generalist trying to handle all scenarios.

**Question 10:** (7 points)
*Sample Answer:* Deep specialization improves performance in specific domains but reduces flexibility. Highly specialized agents may struggle with edge cases outside their narrow expertise, requiring more agents or fallback mechanisms. More generalist agents are flexible but may lack depth. The optimal balance depends on task diversity and whether deep expertise or broad capability is more valuable.

## Part D: Coordination & Consensus

**Question 11:** (5 points)
*Sample Answer:* Multi-agent debate exposes different perspectives and reasoning paths, helping identify flaws in individual agent's logic. By having agents challenge each other, the system can catch errors, explore alternatives, and reach more robust conclusions than any single agent. This implements a form of "ensemble reasoning."

**Question 12:** (5 points)
*Sample Answer:* Strategies to prevent deadlock:
- Implement timeouts for agent responses
- Define clear dependency graphs and detect cycles
- Use asynchronous communication with timeouts
- Implement a supervisor that can break deadlocks
- Set maximum iteration counts

*Any valid strategy worth full points*

## Part E: Orchestration

**Question 13:** (10 points)
*Sample Answer:*

```
Orchestration Flow:

Phase 1 (Parallel):
├─ Research Agent → finds sources and references
└─ Code Agent → prepares code examples (if topic is known)

Phase 2 (Sequential):
Research Agent output → Writing Agent → creates article draft

Phase 3 (Parallel):
Writing Agent output →
├─ Diagram Agent → creates visualizations
└─ Code Agent → refines/tests code examples (if not done in Phase 1)

Phase 4 (Sequential):
Combined output → Review Agent → provides critique

Phase 5 (Conditional):
If Review requires changes:
  → Route back to appropriate agent (Writing/Code/Diagram)
  → Iterate until Review approves
Else:
  → Return final article
```

*Grading:*
- Identifies parallel opportunities (3 pts)
- Shows dependencies (3 pts)
- Includes review/iteration loop (2 pts)
- Clear, logical sequence (2 pts)

---

## Scoring Guide

- **90-100 points:** Excellent - Strong understanding of multi-agent systems
- **80-89 points:** Good - Solid grasp with minor gaps
- **70-79 points:** Passing - Core concepts understood, practice design
- **Below 70:** Review module and design multi-agent system

**Key Topics to Review if Struggling:**
- Multi-agent patterns (supervisor, peer-to-peer, hierarchical)
- Communication strategies and trade-offs
- Message protocol design
- Specialization through system prompts and tools
- Orchestration and dependency management
- Consensus and debate mechanisms
- Preventing deadlock and handling failures
