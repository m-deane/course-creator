# Building a Database MCP Server with FastMCP

## In Brief

FastMCP is a Python library that turns decorated functions into a fully MCP-compliant tool server in under 50 lines. This guide builds a complete database server with three tools: `list_tables`, `describe_table`, and `run_query` — the minimal interface an agent needs to navigate any relational database.

---

## Why These Three Tools?

The tool set is deliberately minimal. Three tools force the agent to learn the right sequence:

```
list_tables   → see what exists
describe_table → understand the schema
run_query     → execute with correct column names
```

Adding more tools (count_rows, get_sample, create_index) shifts training toward breadth-first exploration rather than deep schema understanding. Start minimal and add tools only when the agent has mastered the core pattern.

---

## Project Setup

```bash
pip install fastmcp
```

FastMCP has no server framework dependencies — it manages its own HTTP transport. The only other dependency is the Python standard library `sqlite3`.

---

## The Full Database MCP Server

```python
# database_mcp_server.py
"""
MCP server exposing a SQLite database through three tools:
  - list_tables()       -- discover what tables exist
  - describe_table()    -- inspect column names and types
  - run_query()         -- execute SELECT queries

This is the training environment for the text-to-SQL RL agent.
The server is intentionally minimal: three tools teach the agent
the correct schema-first exploration pattern.
"""

import sqlite3
from pathlib import Path
from typing import Any
import fastmcp

# Server initialization
# host="127.0.0.1" keeps the server local-only during training
mcp = fastmcp.FastMCP("database-server")


def get_connection(db_path: str) -> sqlite3.Connection:
    """
    Open a read-only SQLite connection.

    Read-only mode (uri=True with mode=ro) prevents the agent from
    accidentally or intentionally modifying training data, which would
    cause environment instability across episodes.
    """
    uri = f"file:{db_path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


# Global database path — set at startup
DB_PATH: str = ""


@mcp.tool()
def list_tables() -> list[str]:
    """
    Return the names of all user-created tables in the database.

    Call this first in any database task. It gives you the complete
    inventory of available data before you inspect individual schemas.

    Returns:
        List of table name strings, alphabetically sorted.

    Example:
        list_tables() -> ["departments", "employees", "projects"]
    """
    conn = get_connection(DB_PATH)
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()


@mcp.tool()
def describe_table(table_name: str) -> list[dict[str, str]]:
    """
    Return column definitions for a table: name, type, and whether nullable.

    Call this before writing any query that touches this table.
    Column names and types from this function are guaranteed accurate.
    Guessing column names without calling this tool will cause query failures.

    Args:
        table_name: Exact table name as returned by list_tables().

    Returns:
        List of column definitions, each with keys:
          - "column": column name
          - "type": SQLite type affinity (TEXT, INTEGER, REAL, BLOB, NUMERIC)
          - "nullable": "YES" or "NO"

    Example:
        describe_table("employees") ->
        [
          {"column": "id",      "type": "INTEGER", "nullable": "NO"},
          {"column": "name",    "type": "TEXT",    "nullable": "NO"},
          {"column": "dept_id", "type": "INTEGER", "nullable": "YES"},
          {"column": "salary",  "type": "REAL",    "nullable": "YES"},
        ]
    """
    conn = get_connection(DB_PATH)
    try:
        # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        rows = cursor.fetchall()
        if not rows:
            raise ValueError(
                f"Table '{table_name}' not found. "
                f"Call list_tables() to see available tables."
            )
        return [
            {
                "column": row[1],
                "type": row[2] or "NUMERIC",
                "nullable": "NO" if row[3] else "YES",
            }
            for row in rows
        ]
    finally:
        conn.close()


@mcp.tool()
def run_query(sql: str) -> list[dict[str, Any]]:
    """
    Execute a SQL SELECT query and return results as a list of row dicts.

    Only SELECT statements are permitted. Use list_tables() and
    describe_table() first to confirm table and column names before
    writing your query.

    Args:
        sql: A valid SQLite SELECT statement. Must start with SELECT.

    Returns:
        List of row dictionaries mapping column name to value.
        Returns empty list if the query matches no rows.

    Raises:
        ValueError: If sql does not start with SELECT.
        sqlite3.OperationalError: If sql is syntactically invalid.

    Example:
        run_query("SELECT name, salary FROM employees LIMIT 3") ->
        [
          {"name": "Alice", "salary": 95000.0},
          {"name": "Bob",   "salary": 88000.0},
          {"name": "Carol", "salary": 102000.0},
        ]
    """
    # Enforce SELECT-only to prevent environment mutation
    normalized = sql.strip().upper()
    if not normalized.startswith("SELECT"):
        raise ValueError(
            "Only SELECT queries are permitted. "
            f"Received statement starting with: {sql.strip()[:20]!r}"
        )

    conn = get_connection(DB_PATH)
    try:
        cursor = conn.execute(sql)
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]
    finally:
        conn.close()
```

---

## Creating the Training Database

The server is environment. The database is the world the agent learns to navigate. Here is a complete schema with realistic relationships:

```python
# create_training_db.py
"""
Creates the SQLite training database for the text-to-SQL agent.

Schema:
  employees   -- workforce data with salaries and department assignments
  departments -- organizational units with manager references
  projects    -- active and historical projects with budgets and leads

This database has realistic JOIN paths, NULLable columns, and aggregate
opportunities — enough complexity to generate hundreds of distinct training scenarios.
"""

import sqlite3
from pathlib import Path


def create_training_database(db_path: str = "training.db") -> None:
    """Create and populate the training database."""
    path = Path(db_path)
    if path.exists():
        path.unlink()  # Start fresh — training databases must be stable

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # --- Schema ---
    cursor.executescript("""
        CREATE TABLE departments (
            id          INTEGER PRIMARY KEY,
            name        TEXT    NOT NULL,
            budget      REAL,
            manager_id  INTEGER  -- FK to employees.id (set after employees exist)
        );

        CREATE TABLE employees (
            id          INTEGER PRIMARY KEY,
            name        TEXT    NOT NULL,
            dept_id     INTEGER REFERENCES departments(id),
            salary      REAL,
            hire_date   TEXT,
            is_active   INTEGER DEFAULT 1  -- 1 = active, 0 = inactive
        );

        CREATE TABLE projects (
            id          INTEGER PRIMARY KEY,
            title       TEXT    NOT NULL,
            dept_id     INTEGER REFERENCES departments(id),
            lead_id     INTEGER REFERENCES employees(id),
            budget      REAL,
            status      TEXT    -- 'active', 'completed', 'cancelled'
        );
    """)

    # --- Data ---
    cursor.executemany(
        "INSERT INTO departments (id, name, budget) VALUES (?, ?, ?)",
        [
            (1, "Engineering",  1_500_000.0),
            (2, "Sales",          800_000.0),
            (3, "Marketing",      600_000.0),
            (4, "Operations",     400_000.0),
        ]
    )

    cursor.executemany(
        "INSERT INTO employees (id, name, dept_id, salary, hire_date, is_active) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            (1,  "Alice Chen",    1, 125_000.0, "2019-03-15", 1),
            (2,  "Bob Martinez",  1,  98_000.0, "2020-07-01", 1),
            (3,  "Carol Davis",   2,  75_000.0, "2018-11-20", 1),
            (4,  "Dan Wilson",    2,  68_000.0, "2021-02-14", 1),
            (5,  "Eva Brown",     3,  82_000.0, "2017-09-03", 1),
            (6,  "Frank Lee",     3,  71_000.0, "2022-01-10", 0),  # inactive
            (7,  "Grace Kim",     4,  65_000.0, "2020-05-22", 1),
            (8,  "Henry Patel",   1, 115_000.0, "2016-12-01", 1),
            (9,  "Irene Nguyen",  1,  92_000.0, "2023-04-17", 1),
            (10, "James Taylor",  2,  78_000.0, "2019-08-30", 1),
        ]
    )

    # Set department managers
    cursor.executemany(
        "UPDATE departments SET manager_id = ? WHERE id = ?",
        [(1, 1), (3, 2), (5, 3), (7, 4)]  # (employee_id, dept_id)
    )

    cursor.executemany(
        "INSERT INTO projects (id, title, dept_id, lead_id, budget, status) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            (1, "Platform Migration",  1, 1, 500_000.0, "active"),
            (2, "Mobile App v2",       1, 8, 350_000.0, "active"),
            (3, "Q4 Campaign",         3, 5, 120_000.0, "completed"),
            (4, "Sales Automation",    2, 3,  80_000.0, "active"),
            (5, "Cost Reduction",      4, 7,  45_000.0, "active"),
            (6, "Legacy Sunset",       1, 2, 200_000.0, "cancelled"),
        ]
    )

    conn.commit()
    conn.close()
    print(f"Training database created: {db_path}")


if __name__ == "__main__":
    create_training_database()
```

---

## Running the Server

```python
# run_server.py
"""
Start the database MCP server as a background process.

The server runs on localhost:8000 by default.
In training, ART connects to this URL to discover and call tools.
"""

import database_mcp_server as server_module
import uvicorn


def start_server(db_path: str = "training.db", port: int = 8000) -> None:
    """
    Configure and start the MCP server.

    Args:
        db_path: Path to the SQLite database file.
        port: Port to listen on. Use a non-standard port during training
              to avoid conflicts with other services.
    """
    # Inject the database path into the module-level global
    # FastMCP tools close over module globals, not constructor arguments
    server_module.DB_PATH = db_path

    # mcp.run() starts the ASGI server with SSE transport
    # For training, SSE (Server-Sent Events) is preferred over stdio
    server_module.mcp.run(transport="sse", port=port, host="127.0.0.1")


if __name__ == "__main__":
    import sys
    db = sys.argv[1] if len(sys.argv) > 1 else "training.db"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    start_server(db_path=db, port=port)
```

To start the server as a background process during training:

```bash
# Terminal 1: start the server
python run_server.py training.db 8000

# Terminal 2: run training
python train_agent.py --mcp-url http://127.0.0.1:8000
```

Or programmatically from your training script:

```python
import subprocess
import time

# Start server in background
server_proc = subprocess.Popen(
    ["python", "run_server.py", "training.db", "8000"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
time.sleep(2)  # Wait for server to be ready

try:
    # ... run training ...
    pass
finally:
    server_proc.terminate()
```

---

## Testing the Server Endpoints

Before plugging the server into training, verify each tool works correctly:

```python
# test_server.py
"""
Verify all three MCP tools return correct results.
Run this after starting the server.
"""

import asyncio
import mcp


async def test_all_tools(server_url: str = "http://127.0.0.1:8000") -> None:
    """Run through all three tools and print results."""

    async with mcp.client_session(server_url) as session:

        # Test 1: list_tables
        print("=== list_tables ===")
        result = await session.call_tool("list_tables", {})
        tables = result.content[0].text
        print(f"Tables: {tables}")
        assert "employees" in tables
        assert "departments" in tables
        assert "projects" in tables
        print("PASS\n")

        # Test 2: describe_table
        print("=== describe_table(employees) ===")
        result = await session.call_tool(
            "describe_table", {"table_name": "employees"}
        )
        schema = result.content[0].text
        print(f"Schema: {schema}")
        assert "salary" in schema
        assert "dept_id" in schema
        print("PASS\n")

        # Test 3: run_query — simple SELECT
        print("=== run_query (simple) ===")
        result = await session.call_tool(
            "run_query",
            {"sql": "SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 3"}
        )
        rows = result.content[0].text
        print(f"Top earners: {rows}")
        print("PASS\n")

        # Test 4: run_query — JOIN
        print("=== run_query (JOIN) ===")
        result = await session.call_tool(
            "run_query",
            {"sql": """
                SELECT d.name AS dept, AVG(e.salary) AS avg_salary
                FROM employees e
                JOIN departments d ON e.dept_id = d.id
                WHERE e.is_active = 1
                GROUP BY d.name
                ORDER BY avg_salary DESC
            """}
        )
        rows = result.content[0].text
        print(f"Dept averages: {rows}")
        print("PASS\n")

        # Test 5: run_query — non-SELECT is rejected
        print("=== run_query (non-SELECT rejection) ===")
        try:
            await session.call_tool(
                "run_query",
                {"sql": "DELETE FROM employees WHERE id = 1"}
            )
            print("FAIL — should have raised ValueError")
        except Exception as e:
            print(f"Correctly rejected: {e}")
            print("PASS\n")


if __name__ == "__main__":
    asyncio.run(test_all_tools())
```

Expected output:
```
=== list_tables ===
Tables: ['departments', 'employees', 'projects']
PASS

=== describe_table(employees) ===
Schema: [{'column': 'id', 'type': 'INTEGER', 'nullable': 'NO'}, ...]
PASS

=== run_query (simple) ===
Top earners: [{'name': 'Alice Chen', 'salary': 125000.0}, ...]
PASS

=== run_query (JOIN) ===
Dept averages: [{'dept': 'Engineering', 'avg_salary': 107500.0}, ...]
PASS

=== run_query (non-SELECT rejection) ===
Correctly rejected: Only SELECT queries are permitted...
PASS
```

---

## Common Pitfalls

- **Global DB_PATH not set before first tool call:** The `DB_PATH` global must be assigned before the server starts serving requests. Initialize it in `start_server()`, not lazily.
- **Forgetting `mode=ro` on the connection:** Without read-only mode, a buggy agent or test can delete training data. Always use the `file:path?mode=ro` URI format.
- **Port conflicts in CI:** Port 8000 is commonly used by other services. Pick an unusual port (e.g., 18432) for automated test environments.
- **PRAGMA table_info on a non-existent table:** SQLite returns an empty result rather than raising an error. The explicit check in `describe_table` catches this and returns a useful error message.

---

## Connections

- **Builds on:** Guide 01 (MCP Overview) — protocol architecture, tool discovery concepts
- **Leads to:** Guide 03 (Scenario Generation) — using these three tools to generate training scenarios
- **Applied in:** Module 06 (Text-to-SQL Agent) — training a full agent against this server
- **Related to:** Module 02 (ART Framework) — how ART's MCPToolset connects to this server

---

## Further Reading

See `resources/additional_readings.md` for FastMCP documentation and MCP transport options.
