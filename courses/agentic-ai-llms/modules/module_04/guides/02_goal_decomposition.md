# Goal Decomposition: Breaking Complex Tasks into Steps

> **Reading time:** ~10 min | **Module:** 4 — Agentic Patterns | **Prerequisites:** Module 4 — ReAct Pattern

Complex goals require breaking down into manageable subtasks. Goal decomposition transforms an ambitious objective like "build a data pipeline" into a sequence of concrete, achievable steps that an agent can execute systematically.

<div class="callout-insight">

**Insight:** Complex tasks fail when attempted atomically. Decomposition makes the implicit explicit—surfacing dependencies, enabling parallel execution, and creating natural checkpoints for validation and course correction.

</div>

---

## Decomposition Strategies

### 1. Top-Down Decomposition

Start with the goal, recursively break into subtasks:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>
<div class="code-body">

```python
import anthropic

client = anthropic.Anthropic()


def decompose_goal(goal: str, max_depth: int = 3, current_depth: int = 0) -> dict:
    """Recursively decompose a goal into subtasks."""

    prompt = f"""Break down this goal into 3-5 concrete subtasks.
Each subtask should be:
- Specific and actionable
- Completable independently
- Ordered by dependency (prerequisites first)

Goal: {goal}

Respond in JSON format:
{{
    "subtasks": [
        {{"id": 1, "description": "...", "depends_on": []}},
        {{"id": 2, "description": "...", "depends_on": [1]}},
        ...
    ]
}}"""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    import json
    result = json.loads(response.content[0].text)

    # Recursively decompose if subtasks are still complex
    if current_depth < max_depth:
        for subtask in result["subtasks"]:
            if needs_decomposition(subtask["description"]):
                subtask["subtasks"] = decompose_goal(
                    subtask["description"],
                    max_depth,
                    current_depth + 1
                )["subtasks"]

    return result


def needs_decomposition(task: str) -> bool:
    """Heuristic: task needs decomposition if it's complex."""
    complexity_indicators = [
        "implement", "build", "create", "develop",
        "analyze", "optimize", "integrate", "migrate"
    ]
    return any(ind in task.lower() for ind in complexity_indicators) and len(task) > 50
```

</div>
</div>

### 2. Template-Based Decomposition

Use domain-specific templates:

```python
DECOMPOSITION_TEMPLATES = {
    "data_pipeline": [
        "Define data sources and schemas",
        "Set up ingestion connectors",
        "Implement transformation logic",
        "Create data validation rules",
        "Set up destination/storage",
        "Implement error handling",
        "Add monitoring and alerting",
        "Write documentation"
    ],
    "api_endpoint": [
        "Define request/response schemas",
        "Implement business logic",
        "Add input validation",
        "Implement error handling",
        "Add authentication/authorization",
        "Write tests",
        "Add API documentation"
    ],
    "bug_fix": [
        "Reproduce the bug",
        "Identify root cause",
        "Implement fix",
        "Write regression test",
        "Verify fix in staging",
        "Document the change"
    ]
}


def decompose_with_template(goal: str) -> list[str]:
    """Match goal to template and customize."""

    # Determine template
    prompt = f"""Classify this task into one of these categories:
- data_pipeline
- api_endpoint
- bug_fix
- other

Task: {goal}

Category:"""

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}]
    )

    category = response.content[0].text.strip().lower()

    if category in DECOMPOSITION_TEMPLATES:
        template = DECOMPOSITION_TEMPLATES[category]
        # Customize template for specific goal
        return customize_template(template, goal)
    else:
        # Fall back to dynamic decomposition
        return decompose_goal(goal)["subtasks"]


def customize_template(template: list[str], goal: str) -> list[str]:
    """Customize template steps for the specific goal."""

    prompt = f"""Given this goal:
{goal}

Customize these generic steps to be specific to the goal:
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(template))}

Return the customized steps, one per line:"""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text.strip().split('\n')
```

### 3. Constraint-Aware Decomposition

Consider resources, time, and dependencies:

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class Task:
    id: str
    description: str
    estimated_time: Optional[int] = None  # minutes
    requires_tools: list[str] = None
    depends_on: list[str] = None
    priority: int = 1


def decompose_with_constraints(
    goal: str,
    available_tools: list[str],
    time_budget: int  # minutes
) -> list[Task]:
    """Decompose goal considering available resources."""

    prompt = f"""Break down this goal into tasks, considering:
- Available tools: {', '.join(available_tools)}
- Total time budget: {time_budget} minutes

Goal: {goal}

For each task, specify:
- Description
- Estimated time (minutes)
- Required tools (from available list)
- Dependencies (task IDs this depends on)
- Priority (1=highest, 5=lowest)

Return as JSON array:
[
    {{
        "id": "task_1",
        "description": "...",
        "estimated_time": 30,
        "requires_tools": ["tool1"],
        "depends_on": [],
        "priority": 1
    }},
    ...
]"""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )

    import json
    tasks_data = json.loads(response.content[0].text)

    # Validate time budget
    total_time = sum(t["estimated_time"] for t in tasks_data)
    if total_time > time_budget:
        # Prioritize and potentially drop low-priority tasks
        tasks_data.sort(key=lambda t: t["priority"])
        filtered = []
        running_time = 0
        for t in tasks_data:
            if running_time + t["estimated_time"] <= time_budget:
                filtered.append(t)
                running_time += t["estimated_time"]
        tasks_data = filtered

    return [Task(**t) for t in tasks_data]
```

---

## Execution Strategies

### Sequential Execution

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>
<div class="code-body">

```python
class SequentialExecutor:
    """Execute tasks in dependency order."""

    def __init__(self, agent):
        self.agent = agent
        self.completed = {}

    def execute(self, tasks: list[Task]) -> dict:
        """Execute tasks sequentially, respecting dependencies."""

        # Topological sort
        ordered = self._topological_sort(tasks)

        results = {}
        for task in ordered:
            # Check dependencies
            for dep_id in task.depends_on or []:
                if dep_id not in self.completed:
                    raise ValueError(f"Dependency {dep_id} not completed")

            # Provide context from dependencies
            context = {
                dep_id: self.completed[dep_id]
                for dep_id in (task.depends_on or [])
            }

            # Execute
            result = self.agent.run(
                f"Complete this task: {task.description}\n\n"
                f"Context from previous tasks: {context}"
            )

            results[task.id] = result
            self.completed[task.id] = result

        return results

    def _topological_sort(self, tasks: list[Task]) -> list[Task]:
        """Sort tasks by dependencies."""
        task_map = {t.id: t for t in tasks}
        visited = set()
        ordered = []

        def visit(task_id):
            if task_id in visited:
                return
            visited.add(task_id)
            task = task_map[task_id]
            for dep in task.depends_on or []:
                visit(dep)
            ordered.append(task)

        for task in tasks:
            visit(task.id)

        return ordered
```

</div>
</div>

### Parallel Execution

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor


class ParallelExecutor:
    """Execute independent tasks in parallel."""

    def __init__(self, agent, max_workers: int = 4):
        self.agent = agent
        self.max_workers = max_workers
        self.completed = {}
        self.lock = asyncio.Lock()

    async def execute(self, tasks: list[Task]) -> dict:
        """Execute tasks with maximum parallelism."""

        remaining = {t.id: t for t in tasks}
        results = {}

        while remaining:
            # Find ready tasks (all dependencies met)
            ready = [
                t for t in remaining.values()
                if all(d in self.completed for d in (t.depends_on or []))
            ]

            if not ready:
                if remaining:
                    raise ValueError("Circular dependency detected")
                break

            # Execute ready tasks in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                loop = asyncio.get_event_loop()
                futures = [
                    loop.run_in_executor(executor, self._execute_task, task)
                    for task in ready
                ]
                task_results = await asyncio.gather(*futures)

            # Update completed
            for task, result in zip(ready, task_results):
                results[task.id] = result
                async with self.lock:
                    self.completed[task.id] = result
                del remaining[task.id]

        return results

    def _execute_task(self, task: Task) -> str:
        context = {
            dep_id: self.completed[dep_id]
            for dep_id in (task.depends_on or [])
        }
        return self.agent.run(
            f"Complete this task: {task.description}\n\nContext: {context}"
        )
```

---

## Plan-and-Execute Pattern

### Create Plan, Then Execute

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>
<div class="code-body">

```python
class PlanAndExecuteAgent:
    """Agent that creates a full plan before execution."""

    def __init__(self, planner_model: str, executor_model: str):
        self.client = anthropic.Anthropic()
        self.planner_model = planner_model
        self.executor_model = executor_model

    def run(self, goal: str) -> str:
        # Phase 1: Planning
        plan = self._create_plan(goal)

        # Phase 2: Execution
        results = []
        for step in plan:
            result = self._execute_step(step, results)
            results.append({"step": step, "result": result})

        # Phase 3: Synthesize final answer
        return self._synthesize(goal, results)

    def _create_plan(self, goal: str) -> list[str]:
        """Create a complete plan for the goal."""

        prompt = f"""Create a step-by-step plan to achieve this goal:
{goal}

Requirements:
- Each step should be concrete and actionable
- Steps should be in execution order
- Include validation/verification steps
- Be comprehensive but not excessive

Return steps as a numbered list."""

        response = self.client.messages.create(
            model=self.planner_model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse numbered list
        import re
        steps = re.findall(r'\d+\.\s*(.+)', response.content[0].text)
        return steps

    def _execute_step(self, step: str, previous_results: list) -> str:
        """Execute a single step with context from previous steps."""

        context = ""
        if previous_results:
            context = "Previous steps completed:\n"
            context += "\n".join(
                f"- {r['step']}: {r['result'][:100]}..."
                for r in previous_results[-3:]  # Last 3 for context
            )

        prompt = f"""{context}

Current step to execute: {step}

Complete this step and report the result."""

        response = self.client.messages.create(
            model=self.executor_model,
            max_tokens=1000,
            tools=[...],  # Relevant tools
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def _synthesize(self, goal: str, results: list) -> str:
        """Synthesize final answer from all step results."""

        prompt = f"""Goal: {goal}

Steps completed:
{chr(10).join(f"{i+1}. {r['step']}: {r['result']}" for i, r in enumerate(results))}

Provide a final summary of what was accomplished."""

        response = self.client.messages.create(
            model=self.planner_model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text
```

</div>
</div>

---

## Adaptive Replanning

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>
<div class="code-body">

```python
class AdaptivePlanAgent:
    """Agent that replans when execution diverges from expectations."""

    def __init__(self, client, tools):
        self.client = client
        self.tools = tools

    def run(self, goal: str) -> str:
        plan = self._create_plan(goal)
        results = []

        step_idx = 0
        replan_count = 0
        max_replans = 3

        while step_idx < len(plan) and replan_count < max_replans:
            step = plan[step_idx]

            # Execute step
            result = self._execute_step(step, results)
            results.append({"step": step, "result": result})

            # Evaluate if we're on track
            evaluation = self._evaluate_progress(goal, plan, results)

            if evaluation["on_track"]:
                step_idx += 1
            else:
                # Replan from current state
                remaining_goal = self._extract_remaining_goal(goal, results)
                plan = self._create_plan(remaining_goal)
                step_idx = 0
                replan_count += 1

        return self._synthesize(goal, results)

    def _evaluate_progress(
        self,
        goal: str,
        plan: list[str],
        results: list
    ) -> dict:
        """Evaluate if execution is on track."""

        prompt = f"""Goal: {goal}

Original plan: {plan}

Execution so far:
{results}

Evaluate:
1. Are we making progress toward the goal?
2. Is the remaining plan still valid?
3. Do we need to adjust our approach?

Respond with JSON:
{{"on_track": true/false, "reason": "...", "suggestion": "..."}}"""

        response = self.client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )

        import json
        return json.loads(response.content[0].text)

    def _extract_remaining_goal(self, original_goal: str, results: list) -> str:
        """Extract what still needs to be done."""

        prompt = f"""Original goal: {original_goal}

Completed so far:
{results}

What remains to be done? Express as a new goal."""

        response = self.client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text
```

</div>
</div>

---

## Best Practices

1. **Right-Size Decomposition**: Not too granular (overhead), not too coarse (complexity)
2. **Explicit Dependencies**: Make task relationships clear
3. **Validation Steps**: Include verification in the plan
4. **Checkpoints**: Create natural save points
5. **Failure Handling**: Plan for what to do when steps fail
6. **Progress Tracking**: Log execution for debugging
7. **Adaptive Planning**: Be ready to replan when reality diverges

<div class="callout-key">

**Key Concept Summary:** This guide covered the core concepts. Review the companion slides for visual summaries and the hands-on notebook for practice implementations.

</div>

---

*Goal decomposition transforms ambitious objectives into achievable task sequences. Master this skill to build agents that can tackle any complex challenge.*


## Practice Questions

1. Explain in your own words how the concepts in this guide relate to building production agents.
2. What are the key tradeoffs you need to consider when applying these techniques?
3. Describe a scenario where the approach from this guide would be the wrong choice, and what you would use instead.

---

**Next Steps:**

<a class="link-card" href="./02_goal_decomposition_slides.md">
  <div class="link-card-title">Goal Decomposition and Planning — Companion Slides</div>
  <div class="link-card-description">Visual slide deck with diagrams, speaker notes, and key takeaways.</div>
</a>

<a class="link-card" href="../notebooks/01_react_agents.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with working code and guided exercises.</div>
</a>
