# Auto-Generating Training Scenarios from Tool Schemas

## In Brief

Writing training scenarios by hand does not scale. For an RL agent that must learn to use tools across hundreds of distinct tasks, you need automatic scenario generation. The approach: give an LLM the tool schemas, describe the difficulty levels you want, and let it generate structurally diverse scenarios without any manual authoring.

---

## The Scenario Generation Problem

An RL agent learns from experience. To develop robust tool-use skills, it needs exposure to:
- Simple lookups (single tool, one step)
- Multi-step sequences (discover → inspect → query)
- Complex aggregations (GROUP BY, HAVING, nested queries)
- Edge cases (empty results, NULLs, non-existent tables)

Writing 100 distinct scenarios by hand is tedious and produces a biased distribution — human authors tend to write scenarios that feel natural, missing corner cases the agent will encounter in deployment.

The solution: use an LLM to generate the scenarios. Feed it the tool schemas and ask for diversity.

---

## How the Generation Works

```
Tool schemas (from MCP server)
    │
    ▼
LLM receives schemas + difficulty instructions
    │
    ▼
LLM generates N distinct scenarios:
    ├── Task description (natural language question)
    ├── Difficulty tier (single-tool / multi-tool / complex)
    ├── Expected tool sequence (which tools, in what order)
    └── Ground truth answer (for reward calculation)
    │
    ▼
Scenarios fed into ART training loop
```

The LLM does not need to know the actual data — only the schemas. It generates structurally valid scenarios, and the ground truth answers are computed by running the expected queries against the actual database.

---

## Three Scenario Types

### Type 1: Single-Tool Lookups

The agent calls one tool and answers directly from its output.

```
Task: "List all tables in the database"
Expected tools: [list_tables]
Expected answer: ["departments", "employees", "projects"]
Difficulty: easy
```

```
Task: "What columns does the projects table have?"
Expected tools: [describe_table]
Expected answer: [id, title, dept_id, lead_id, budget, status]
Difficulty: easy
```

These scenarios teach the agent that some questions can be answered without a SQL query — an important behavior for cost efficiency.

### Type 2: Multi-Step Queries (JOINs Required)

The agent must discover schema, inspect tables, then write a JOIN query.

```
Task: "What is the average salary for each department?"
Expected tools: [list_tables, describe_table("employees"),
                 describe_table("departments"), run_query]
Expected query:
    SELECT d.name, AVG(e.salary)
    FROM employees e
    JOIN departments d ON e.dept_id = d.id
    GROUP BY d.name
Difficulty: medium
```

```
Task: "Which departments have at least one employee earning over $100,000?"
Expected tools: [list_tables, describe_table("employees"),
                 describe_table("departments"), run_query]
Expected query:
    SELECT DISTINCT d.name
    FROM departments d
    JOIN employees e ON e.dept_id = d.id
    WHERE e.salary > 100000
Difficulty: medium
```

### Type 3: Multi-Table Chains (Three Tools in Sequence)

The most complex scenarios require joining all three tables and applying aggregation with filtering.

```
Task: "Who leads the most expensive active project, and what department do they work in?"
Expected tools: [list_tables, describe_table("projects"),
                 describe_table("employees"), describe_table("departments"),
                 run_query]
Expected query:
    SELECT e.name, d.name AS department, p.title, p.budget
    FROM projects p
    JOIN employees e ON p.lead_id = e.id
    JOIN departments d ON e.dept_id = d.id
    WHERE p.status = 'active'
    ORDER BY p.budget DESC
    LIMIT 1
Difficulty: hard
```

```
Task: "Which department has the highest total budget across active projects,
       and how many employees does it have?"
Expected tools: [all three describe_table calls, run_query]
Expected query:
    SELECT d.name,
           SUM(p.budget) AS project_budget,
           COUNT(e.id)   AS employee_count
    FROM departments d
    LEFT JOIN projects p  ON p.dept_id = d.id AND p.status = 'active'
    LEFT JOIN employees e ON e.dept_id = d.id AND e.is_active = 1
    GROUP BY d.name
    ORDER BY project_budget DESC
    LIMIT 1
Difficulty: hard
```

---

## The Scenario Generator

```python
# scenario_generator.py
"""
Auto-generate training scenarios from MCP tool schemas.

The generator uses an LLM to create diverse task descriptions, then
validates each scenario by running the expected query against the
actual database. Only scenarios with non-empty results are kept.

Usage:
    generator = ScenarioGenerator(
        db_path="training.db",
        model="claude-sonnet-4-6",
    )
    scenarios = await generator.generate(n=100)
"""

import json
import sqlite3
import asyncio
from dataclasses import dataclass, field
from typing import Literal
import anthropic


# --- Data structures ---

@dataclass
class Scenario:
    """A single training scenario for the text-to-SQL agent."""
    task: str                          # Natural language question
    difficulty: Literal["easy", "medium", "hard"]
    expected_tool_sequence: list[str]  # Tools the agent should call, in order
    ground_truth_sql: str | None       # The correct SQL query (None for non-query tasks)
    ground_truth_answer: str           # The expected final answer
    hint: str = ""                     # Optional hint for reward function


@dataclass
class GenerationConfig:
    """Controls the mix and diversity of generated scenarios."""
    n_easy: int = 20    # Single-tool lookup scenarios
    n_medium: int = 50  # Multi-step JOIN scenarios
    n_hard: int = 30    # Three-table chain scenarios
    max_retries: int = 3
    validate_sql: bool = True  # Run SQL against DB to confirm results


# --- Schema extraction ---

def extract_schema_summary(db_path: str) -> str:
    """
    Build a concise schema description for the LLM prompt.

    The LLM does not need full PRAGMA output — it needs enough to
    write valid SQL and generate realistic task descriptions.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]

        schema_parts = []
        for table in tables:
            cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
            col_defs = ", ".join(
                f"{col[1]} {col[2]}{'(nullable)' if not col[3] else ''}"
                for col in cols
            )
            schema_parts.append(f"  {table}({col_defs})")

        return "\n".join(schema_parts)
    finally:
        conn.close()


def get_sample_data(db_path: str, table: str, limit: int = 3) -> list[dict]:
    """
    Fetch a few rows from a table so the LLM can write realistic tasks.

    Seeing actual data values (like status='active') prevents the LLM
    from inventing filter values that do not exist in the database.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cursor = conn.execute(f"SELECT * FROM {table} LIMIT {limit}")
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]
    finally:
        conn.close()


# --- LLM-based generation ---

GENERATION_PROMPT_TEMPLATE = """
You are generating training scenarios for a reinforcement learning agent
that learns to navigate a SQLite database using three tools:
  - list_tables(): returns all table names
  - describe_table(table_name): returns column definitions
  - run_query(sql): executes a SELECT query, returns rows as dicts

Database schema:
{schema}

Sample data:
{sample_data}

Generate {n} {difficulty} training scenarios. Each scenario is a JSON object:
{{
  "task": "<natural language question a business analyst might ask>",
  "difficulty": "{difficulty}",
  "expected_tool_sequence": ["<tool1>", "<tool2>", ...],
  "ground_truth_sql": "<SQL query or null if no query needed>",
  "ground_truth_answer": "<concise answer in natural language>"
}}

Rules for {difficulty} scenarios:
{difficulty_rules}

Return a JSON array of {n} scenario objects. No markdown, no explanation.
Just the JSON array.
"""

DIFFICULTY_RULES = {
    "easy": """
- Use only ONE tool (either list_tables or describe_table)
- Task is answered directly from the tool output, no SQL needed
- Examples: "What tables exist?", "What columns does X have?"
- expected_tool_sequence has exactly one entry
- ground_truth_sql is null
""",
    "medium": """
- Require 3-4 tool calls: list_tables + 1-2 describe_table + run_query
- SQL uses one JOIN between two tables
- May include GROUP BY, ORDER BY, WHERE, LIMIT
- Tasks involve aggregations (avg, count, sum) or filtered lookups
- expected_tool_sequence has 3-4 entries
""",
    "hard": """
- Require 4-6 tool calls: list_tables + multiple describe_table + run_query
- SQL joins all three tables (employees, departments, projects)
- Must include either: HAVING clause, subquery, or multiple aggregations
- Tasks require cross-domain reasoning (e.g., project budget + employee salary)
- expected_tool_sequence has 4-6 entries
""",
}


async def generate_scenarios_for_difficulty(
    client: anthropic.AsyncAnthropic,
    schema: str,
    sample_data: str,
    difficulty: str,
    n: int,
    model: str = "claude-sonnet-4-6",
) -> list[Scenario]:
    """Generate n scenarios at the specified difficulty level."""
    prompt = GENERATION_PROMPT_TEMPLATE.format(
        schema=schema,
        sample_data=sample_data,
        n=n,
        difficulty=difficulty,
        difficulty_rules=DIFFICULTY_RULES[difficulty],
    )

    response = await client.messages.create(
        model=model,
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.content[0].text
    data = json.loads(raw_text)

    scenarios = []
    for item in data:
        scenarios.append(Scenario(
            task=item["task"],
            difficulty=item["difficulty"],
            expected_tool_sequence=item["expected_tool_sequence"],
            ground_truth_sql=item.get("ground_truth_sql"),
            ground_truth_answer=item["ground_truth_answer"],
        ))
    return scenarios


def validate_scenario(scenario: Scenario, db_path: str) -> bool:
    """
    Run the ground truth SQL against the database.

    Rejects scenarios where:
    - SQL is syntactically invalid
    - Query returns zero rows (the agent cannot learn from empty results)
    - Query raises an exception (bad column names, etc.)

    Returns True if the scenario is usable for training.
    """
    if scenario.ground_truth_sql is None:
        return True  # Non-query scenarios are always valid

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cursor = conn.execute(scenario.ground_truth_sql)
        rows = cursor.fetchall()
        return len(rows) > 0  # Reject empty-result scenarios
    except Exception:
        return False  # Invalid SQL -- reject
    finally:
        conn.close()


# --- Main generator class ---

class ScenarioGenerator:
    """
    Generates training scenarios from MCP tool schemas using an LLM.

    The generator extracts schema information from the database,
    prompts the LLM for scenarios at each difficulty level, and
    validates each scenario by running its SQL against the actual data.
    """

    def __init__(
        self,
        db_path: str,
        model: str = "claude-sonnet-4-6",
        config: GenerationConfig | None = None,
    ):
        self.db_path = db_path
        self.model = model
        self.config = config or GenerationConfig()
        self.client = anthropic.AsyncAnthropic()

        # Extract schema once at initialization
        self.schema = extract_schema_summary(db_path)
        self.sample_data = self._build_sample_data_str()

    def _build_sample_data_str(self) -> str:
        """Build a formatted sample data block for the generation prompt."""
        tables = ["employees", "departments", "projects"]
        parts = []
        for table in tables:
            rows = get_sample_data(self.db_path, table, limit=2)
            parts.append(f"{table}: {json.dumps(rows, indent=2)}")
        return "\n\n".join(parts)

    async def generate(self, n: int | None = None) -> list[Scenario]:
        """
        Generate the full scenario set.

        Args:
            n: Total scenarios to generate. If None, uses config totals.
               When specified, splits proportionally: 20% easy, 50% medium, 30% hard.

        Returns:
            Validated scenarios ready for training.
        """
        if n is not None:
            n_easy = max(1, int(n * 0.20))
            n_medium = max(1, int(n * 0.50))
            n_hard = max(1, n - n_easy - n_medium)
        else:
            n_easy = self.config.n_easy
            n_medium = self.config.n_medium
            n_hard = self.config.n_hard

        # Generate all difficulty levels concurrently
        easy_task = generate_scenarios_for_difficulty(
            self.client, self.schema, self.sample_data, "easy", n_easy, self.model
        )
        medium_task = generate_scenarios_for_difficulty(
            self.client, self.schema, self.sample_data, "medium", n_medium, self.model
        )
        hard_task = generate_scenarios_for_difficulty(
            self.client, self.schema, self.sample_data, "hard", n_hard, self.model
        )

        easy, medium, hard = await asyncio.gather(easy_task, medium_task, hard_task)
        all_scenarios = easy + medium + hard

        if self.config.validate_sql:
            validated = [s for s in all_scenarios if validate_scenario(s, self.db_path)]
            print(
                f"Validated {len(validated)}/{len(all_scenarios)} scenarios "
                f"({len(all_scenarios) - len(validated)} rejected)"
            )
            return validated

        return all_scenarios

    async def generate_edge_cases(self) -> list[Scenario]:
        """
        Generate edge case scenarios explicitly.

        Edge cases are rare in random generation but important for robustness:
        - Queries that return exactly one row
        - Queries involving NULL values
        - Queries with HAVING clauses
        - Tasks that require describe_table on a table with no relevant data
        """
        prompt = f"""
Generate 10 edge case training scenarios for a database agent.
Schema: {self.schema}

Edge cases to cover:
1. A task where the WHERE clause filters to exactly one result
2. A task requiring IS NOT NULL or IS NULL filtering
3. A task requiring a HAVING clause (filter after GROUP BY)
4. A task where list_tables() is sufficient (no SQL needed)
5. A task with two separate aggregations in one query (e.g., COUNT and AVG)

Same JSON format as before. Return only the JSON array.
"""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        data = json.loads(response.content[0].text)
        scenarios = [Scenario(**item) for item in data]
        return [s for s in scenarios if validate_scenario(s, self.db_path)]
```

---

## Controlling Diversity and Difficulty

### Distribution Design

The 20/50/30 split (easy/medium/hard) is deliberate:

| Tier | Count | Rationale |
|------|-------|-----------|
| Easy (20%) | 20 | Foundation skills: schema reading without SQL |
| Medium (50%) | 50 | Core skill: two-table JOINs with aggregation |
| Hard (30%) | 30 | Transfer skill: three-table chains |

Starting with more easy scenarios and shifting the distribution toward hard as training progresses is curriculum learning — a proven technique for faster convergence.

### Curriculum Learning Implementation

```python
# Progressive difficulty: start easy, add hard as agent improves
async def curriculum_training(agent, db_path: str):
    generator = ScenarioGenerator(db_path)

    # Phase 1: Easy scenarios only
    easy_scenarios = await generator.generate_scenarios_for_difficulty(
        "easy", n=50
    )
    await art.train(agent, easy_scenarios, epochs=5)

    # Phase 2: Mix easy and medium when agent achieves >70% on easy
    medium_scenarios = await generator.generate_scenarios_for_difficulty(
        "medium", n=100
    )
    mixed = easy_scenarios + medium_scenarios
    await art.train(agent, mixed, epochs=10)

    # Phase 3: Full distribution including hard
    hard_scenarios = await generator.generate_scenarios_for_difficulty(
        "hard", n=50
    )
    all_scenarios = mixed + hard_scenarios
    await art.train(agent, all_scenarios, epochs=15)
```

### Diversity Metrics

After generation, verify the scenario set is actually diverse:

```python
def analyze_scenario_diversity(scenarios: list[Scenario]) -> dict:
    """Check that generated scenarios cover the expected variety."""
    tool_sequences = [tuple(s.expected_tool_sequence) for s in scenarios]
    unique_sequences = set(tool_sequences)

    difficulties = {
        "easy":   sum(1 for s in scenarios if s.difficulty == "easy"),
        "medium": sum(1 for s in scenarios if s.difficulty == "medium"),
        "hard":   sum(1 for s in scenarios if s.difficulty == "hard"),
    }

    sql_patterns = {
        "join":     sum(1 for s in scenarios if s.ground_truth_sql and "JOIN" in s.ground_truth_sql.upper()),
        "group_by": sum(1 for s in scenarios if s.ground_truth_sql and "GROUP BY" in s.ground_truth_sql.upper()),
        "having":   sum(1 for s in scenarios if s.ground_truth_sql and "HAVING" in s.ground_truth_sql.upper()),
        "subquery": sum(1 for s in scenarios if s.ground_truth_sql and "SELECT" in (s.ground_truth_sql.upper()[10:] if s.ground_truth_sql else "")),
    }

    return {
        "total": len(scenarios),
        "unique_tool_sequences": len(unique_sequences),
        "difficulties": difficulties,
        "sql_patterns": sql_patterns,
    }
```

---

## End-to-End Usage

```python
# generate_and_save.py
"""Generate scenario set and save for training."""

import asyncio
import json
from scenario_generator import ScenarioGenerator, GenerationConfig
from dataclasses import asdict


async def main():
    config = GenerationConfig(
        n_easy=20,
        n_medium=50,
        n_hard=30,
        validate_sql=True,
    )

    generator = ScenarioGenerator(
        db_path="training.db",
        model="claude-sonnet-4-6",
        config=config,
    )

    # Generate main scenario set
    scenarios = await generator.generate()

    # Add edge cases
    edge_cases = await generator.generate_edge_cases()
    all_scenarios = scenarios + edge_cases

    print(f"Total scenarios: {len(all_scenarios)}")

    # Save to JSON for inspection and reproducibility
    with open("scenarios.json", "w") as f:
        json.dump([asdict(s) for s in all_scenarios], f, indent=2)

    # Print diversity report
    from scenario_generator import analyze_scenario_diversity
    report = analyze_scenario_diversity(all_scenarios)
    print(f"Unique tool sequences: {report['unique_tool_sequences']}")
    print(f"Distribution: {report['difficulties']}")
    print(f"SQL patterns: {report['sql_patterns']}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Common Pitfalls

- **Not validating SQL before training:** The LLM occasionally generates syntactically valid SQL that returns zero rows against the actual data. Filtering these out prevents the agent from seeing "correct tool use, empty result" as a confusing signal.
- **Skewing toward medium scenarios:** LLMs naturally generate JOIN-heavy scenarios because they are the most interesting to write. Explicitly request each difficulty tier separately to maintain balance.
- **Reusing the same generation call for every training run:** Generated scenarios become stale after many training epochs. Regenerate periodically (every 3-5 epochs) to prevent overfitting to specific phrasings.
- **Not including edge cases:** The standard generator rarely produces NULL-handling or HAVING scenarios. Call `generate_edge_cases()` explicitly to ensure coverage.

---

## Connections

- **Builds on:** Guide 02 (FastMCP Server) — the tool schemas used for generation
- **Builds on:** Module 03 (RULER Rewards) — reward functions that evaluate scenario completion
- **Applied in:** Module 05 (Training Loop) — scenarios fed directly into ART's training loop
- **Applied in:** Module 06 (Text-to-SQL Agent) — end-to-end training with generated scenarios

---

## Further Reading

See `resources/additional_readings.md` for papers on automatic curriculum generation and LLM-based data synthesis.
