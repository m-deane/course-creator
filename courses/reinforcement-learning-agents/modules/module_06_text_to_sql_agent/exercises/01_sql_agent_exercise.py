"""
Exercise 01: Build a Text-to-SQL Agent

Self-check exercise for Module 06. Work through four tasks:
  1. Create a SQLite database with a specific schema
  2. Write three tool functions for database access
  3. Implement a simple agent loop using those tools
  4. Score agent responses for correctness

Each task has auto-checks you can run after completing it.
All checks print a clear message telling you what passed or what to fix.

Requirements:
  pip install openai  (for the agent loop in Task 3)

No ART, no RULER — this exercise uses a simplified agent loop and
a rule-based scorer so you can run it without a GPU or API credits.

Usage:
  python 01_sql_agent_exercise.py
"""

import json
import re
import sqlite3
from pathlib import Path
from typing import Any


# =============================================================================
# TASK 1: Create the database
# =============================================================================
#
# Create a SQLite database called "bookstore.db" with two tables:
#
#   books:
#     id        INTEGER PRIMARY KEY AUTOINCREMENT
#     title     TEXT NOT NULL
#     author    TEXT NOT NULL
#     genre     TEXT NOT NULL
#     price     REAL NOT NULL
#     in_stock  INTEGER NOT NULL DEFAULT 1   (1 = in stock, 0 = out of stock)
#
#   sales:
#     id        INTEGER PRIMARY KEY AUTOINCREMENT
#     book_id   INTEGER NOT NULL REFERENCES books(id)
#     quantity  INTEGER NOT NULL
#     sale_date TEXT NOT NULL   (ISO format: '2024-01-15')
#
# Populate books with at least 8 rows spanning at least 3 genres.
# Populate sales with at least 6 rows.
#
# Requirements:
#   - Enable foreign key enforcement
#   - At least one book should be out of stock (in_stock = 0)
#   - Sales should reference valid book IDs
#
# =============================================================================


def create_bookstore_database(db_path: str = "bookstore.db") -> sqlite3.Connection:
    """
    Create and populate the bookstore database.

    Returns an open connection to the created database.
    """
    # YOUR CODE HERE
    # Hint: Use Path(db_path).unlink() to remove the existing file first
    # Hint: Set PRAGMA foreign_keys = ON after connecting
    # Hint: Use conn.row_factory = sqlite3.Row for column-name access
    raise NotImplementedError("Task 1: implement create_bookstore_database()")


def check_task_1() -> bool:
    """Auto-check for Task 1. Run after implementing create_bookstore_database()."""
    print("\n--- Task 1 Checks ---")
    passed = True

    try:
        conn = create_bookstore_database("bookstore.db")
    except NotImplementedError:
        print("FAIL: create_bookstore_database() not implemented yet")
        return False
    except Exception as exc:
        print(f"FAIL: create_bookstore_database() raised an exception: {exc}")
        return False

    cursor = conn.cursor()

    # Check 1: tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    expected_tables = {"books", "sales"}
    if not expected_tables.issubset(tables):
        missing = expected_tables - tables
        print(f"FAIL: Missing tables: {missing}")
        passed = False
    else:
        print("PASS: Both tables exist (books, sales)")

    # Check 2: books has at least 8 rows
    cursor.execute("SELECT COUNT(*) FROM books")
    book_count = cursor.fetchone()[0]
    if book_count < 8:
        print(f"FAIL: books table has {book_count} rows, need at least 8")
        passed = False
    else:
        print(f"PASS: books table has {book_count} rows")

    # Check 3: at least 3 distinct genres
    cursor.execute("SELECT COUNT(DISTINCT genre) FROM books")
    genre_count = cursor.fetchone()[0]
    if genre_count < 3:
        print(f"FAIL: books table has {genre_count} distinct genres, need at least 3")
        passed = False
    else:
        print(f"PASS: {genre_count} distinct genres in books table")

    # Check 4: at least one out-of-stock book
    cursor.execute("SELECT COUNT(*) FROM books WHERE in_stock = 0")
    out_of_stock = cursor.fetchone()[0]
    if out_of_stock < 1:
        print("FAIL: No out-of-stock books (in_stock = 0). Add at least one.")
        passed = False
    else:
        print(f"PASS: {out_of_stock} out-of-stock book(s)")

    # Check 5: sales table has at least 6 rows
    cursor.execute("SELECT COUNT(*) FROM sales")
    sales_count = cursor.fetchone()[0]
    if sales_count < 6:
        print(f"FAIL: sales table has {sales_count} rows, need at least 6")
        passed = False
    else:
        print(f"PASS: sales table has {sales_count} rows")

    # Check 6: no orphan sales (all book_ids reference valid books)
    cursor.execute("""
        SELECT COUNT(*) FROM sales s
        LEFT JOIN books b ON s.book_id = b.id
        WHERE b.id IS NULL
    """)
    orphan_count = cursor.fetchone()[0]
    if orphan_count > 0:
        print(f"FAIL: {orphan_count} sales row(s) reference non-existent book IDs")
        passed = False
    else:
        print("PASS: All sales rows reference valid book IDs")

    conn.close()
    return passed


# =============================================================================
# TASK 2: Write the tool functions
# =============================================================================
#
# Implement three tool functions that wrap database access.
# These mirror the MCP server tools from Guide 02, but as plain Python
# functions — no FastMCP required for this exercise.
#
# Each function takes a db_path argument so tests can pass a custom path.
#
# Tool functions must handle errors gracefully:
#   - If a table does not exist, return a dict with key "error"
#   - If SQL execution fails, return a dict with key "error"
#   - Never raise exceptions from tool functions — return error dicts instead
#
# =============================================================================


def tool_list_tables(db_path: str = "bookstore.db") -> list[str]:
    """
    Return the names of all tables in the database, sorted alphabetically.

    Example return value: ["books", "sales"]
    """
    # YOUR CODE HERE
    # Hint: Query sqlite_master WHERE type='table'
    raise NotImplementedError("Task 2: implement tool_list_tables()")


def tool_describe_table(table_name: str, db_path: str = "bookstore.db") -> dict[str, Any]:
    """
    Return the schema for a table: column names, types, and whether each
    column is a primary key.

    If the table does not exist, return {"error": "Table '...' does not exist."}.

    Example return value for books:
    {
        "table": "books",
        "columns": [
            {"name": "id", "type": "INTEGER", "primary_key": True},
            {"name": "title", "type": "TEXT", "primary_key": False},
            ...
        ]
    }
    """
    # YOUR CODE HERE
    # Hint: Use PRAGMA table_info(table_name) to get column metadata
    # Hint: Validate that the table exists first using sqlite_master
    raise NotImplementedError("Task 2: implement tool_describe_table()")


def tool_run_query(sql: str, db_path: str = "bookstore.db") -> dict[str, Any]:
    """
    Execute a SELECT query and return results.

    Returns {"rows": [...], "row_count": N} on success.
    Returns {"error": "..."} if the SQL is not a SELECT or if execution fails.

    Constraints:
    - Only SELECT statements are permitted (check before executing)
    - Catch sqlite3.OperationalError and return it as {"error": str(exc)}
    - Never raise exceptions

    Example return value:
    {"rows": [{"title": "Dune", "price": 14.99}], "row_count": 1}
    """
    # YOUR CODE HERE
    # Hint: Check if sql.strip().upper().startswith("SELECT")
    # Hint: Use conn.row_factory = sqlite3.Row, then dict(row) for each result
    raise NotImplementedError("Task 2: implement tool_run_query()")


def check_task_2() -> bool:
    """Auto-check for Task 2. Run after implementing all three tool functions."""
    print("\n--- Task 2 Checks ---")

    # Make sure Task 1 database exists
    if not Path("bookstore.db").exists():
        print("FAIL: bookstore.db does not exist. Complete Task 1 first.")
        return False

    passed = True

    # --- list_tables ---
    try:
        tables = tool_list_tables("bookstore.db")
    except NotImplementedError:
        print("FAIL: tool_list_tables() not implemented yet")
        return False
    except Exception as exc:
        print(f"FAIL: tool_list_tables() raised {type(exc).__name__}: {exc}")
        return False

    if not isinstance(tables, list):
        print(f"FAIL: tool_list_tables() should return a list, got {type(tables)}")
        passed = False
    elif "books" not in tables or "sales" not in tables:
        print(f"FAIL: tool_list_tables() should include 'books' and 'sales', got {tables}")
        passed = False
    else:
        print(f"PASS: tool_list_tables() returned {tables}")

    # --- describe_table: valid table ---
    try:
        result = tool_describe_table("books", "bookstore.db")
    except NotImplementedError:
        print("FAIL: tool_describe_table() not implemented yet")
        return False
    except Exception as exc:
        print(f"FAIL: tool_describe_table('books') raised {type(exc).__name__}: {exc}")
        return False

    if "error" in result:
        print(f"FAIL: tool_describe_table('books') returned error: {result['error']}")
        passed = False
    elif "columns" not in result:
        print(f"FAIL: tool_describe_table() result missing 'columns' key. Got: {result}")
        passed = False
    else:
        col_names = [c["name"] for c in result["columns"]]
        required_cols = {"id", "title", "author", "genre", "price", "in_stock"}
        missing = required_cols - set(col_names)
        if missing:
            print(f"FAIL: describe_table('books') missing columns: {missing}")
            passed = False
        else:
            # Check that id is marked as primary key
            id_col = next((c for c in result["columns"] if c["name"] == "id"), None)
            if id_col and not id_col.get("primary_key"):
                print("FAIL: 'id' column in books should have primary_key=True")
                passed = False
            else:
                print(f"PASS: tool_describe_table('books') returned {len(col_names)} columns")

    # --- describe_table: non-existent table ---
    try:
        error_result = tool_describe_table("nonexistent_table_xyz", "bookstore.db")
    except Exception as exc:
        print(f"FAIL: tool_describe_table('nonexistent') should not raise, raised: {exc}")
        passed = False
    else:
        if "error" not in error_result:
            print("FAIL: tool_describe_table() with bad table name should return {'error': ...}")
            passed = False
        else:
            print("PASS: tool_describe_table() returns error dict for non-existent table")

    # --- run_query: valid SELECT ---
    try:
        query_result = tool_run_query("SELECT COUNT(*) AS cnt FROM books", "bookstore.db")
    except NotImplementedError:
        print("FAIL: tool_run_query() not implemented yet")
        return False
    except Exception as exc:
        print(f"FAIL: tool_run_query() raised {type(exc).__name__}: {exc}")
        return False

    if "error" in query_result:
        print(f"FAIL: tool_run_query() returned error on valid SELECT: {query_result['error']}")
        passed = False
    elif "rows" not in query_result or "row_count" not in query_result:
        print(f"FAIL: tool_run_query() should return dict with 'rows' and 'row_count'. Got: {query_result}")
        passed = False
    else:
        print(f"PASS: tool_run_query() returned {query_result['row_count']} row(s)")

    # --- run_query: rejected non-SELECT ---
    try:
        reject_result = tool_run_query(
            "DELETE FROM books WHERE id = 1", "bookstore.db"
        )
    except Exception as exc:
        print(f"FAIL: tool_run_query() should not raise on non-SELECT, raised: {exc}")
        passed = False
    else:
        if "error" not in reject_result:
            print("FAIL: tool_run_query() should return error dict for non-SELECT statements")
            passed = False
        else:
            print("PASS: tool_run_query() rejects non-SELECT statements")

    # --- run_query: SQL error returns error dict ---
    try:
        sql_error_result = tool_run_query(
            "SELECT nonexistent_column FROM books", "bookstore.db"
        )
    except Exception as exc:
        print(f"FAIL: tool_run_query() should not raise on bad SQL, raised: {exc}")
        passed = False
    else:
        if "error" not in sql_error_result:
            print("FAIL: tool_run_query() should return error dict for bad SQL")
            passed = False
        else:
            print("PASS: tool_run_query() returns error dict for bad SQL")

    return passed


# =============================================================================
# TASK 3: Implement a simple agent loop
# =============================================================================
#
# Implement run_agent_loop() — a simplified version of the rollout function
# from Guide 03. It does not use ART or MCP; it calls tool functions directly.
#
# The agent:
#   1. Receives a system prompt and a user question
#   2. Generates a response via the OpenAI API
#   3. If the response contains a tool call, executes the matching tool function
#   4. Appends the tool result to the conversation and generates again
#   5. Stops when the model produces a response with no tool calls
#   6. Returns the final answer string
#
# Available tools for the agent (pass as tools list to the API):
#   - list_tables: no arguments
#   - describe_table: one argument "table_name" (string)
#   - run_query: one argument "sql" (string)
#
# Use the AGENT_SYSTEM_PROMPT defined below.
# Use model="gpt-4o-mini" (or any model that supports function calling).
#
# =============================================================================

AGENT_SYSTEM_PROMPT = """You are a text-to-SQL agent with access to a bookstore database.

To answer questions, use the provided tools in order:
1. Call list_tables() to discover available tables
2. Call describe_table(table_name) to understand each table's schema
3. Write a SELECT query based on the schema you discovered
4. Call run_query(sql) to get the answer

Return a direct, factual answer based on the query results.
If a query returns an error, correct your SQL and try again."""

OPENAI_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": (
                "Return all table names in the database. "
                "Always call this first before writing any SQL."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_table",
            "description": (
                "Return column names, types, and primary key info for a table. "
                "Call this before writing a query that references the table."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Exact table name from list_tables(). Case-sensitive.",
                    }
                },
                "required": ["table_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_query",
            "description": (
                "Execute a SELECT SQL query and return results. "
                "Only SELECT statements are permitted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A valid SQLite SELECT statement.",
                    }
                },
                "required": ["sql"],
            },
        },
    },
]


def dispatch_tool_call(tool_name: str, arguments: dict[str, Any]) -> Any:
    """
    Execute a tool function given its name and arguments dict.
    Returns the tool result.

    This replaces the MCP client from Guide 03 — same concept, no server needed.
    """
    if tool_name == "list_tables":
        return tool_list_tables("bookstore.db")
    elif tool_name == "describe_table":
        return tool_describe_table(arguments["table_name"], "bookstore.db")
    elif tool_name == "run_query":
        return tool_run_query(arguments["sql"], "bookstore.db")
    else:
        return {"error": f"Unknown tool: {tool_name}"}


def run_agent_loop(
    question: str,
    db_path: str = "bookstore.db",
    max_tool_calls: int = 8,
) -> str:
    """
    Run one agent episode: given a question, use tools to find and return the answer.

    Returns the agent's final answer as a string.

    Implementation guide:
    1. Import openai and create a client
    2. Build initial messages: system prompt + user question
    3. Call client.chat.completions.create() with the tool schemas
    4. If response has tool_calls:
       a. For each tool call, parse arguments with json.loads()
       b. Call dispatch_tool_call() to get the result
       c. Append assistant message and tool result to messages
       d. Increment tool_call_count
    5. If response has no tool_calls: extract content and return it
    6. If tool_call_count reaches max_tool_calls: return a timeout message

    The messages format for tool results:
        {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
    """
    # YOUR CODE HERE
    raise NotImplementedError("Task 3: implement run_agent_loop()")


def check_task_3() -> bool:
    """
    Auto-check for Task 3.

    Runs the agent on a simple question and verifies:
    - The function returns a string (not raises)
    - The string is non-empty
    - The string appears to contain a meaningful answer (not just an error message)

    Note: This check requires an OpenAI API key in your environment:
      export OPENAI_API_KEY=sk-...
    """
    print("\n--- Task 3 Checks ---")

    if not Path("bookstore.db").exists():
        print("FAIL: bookstore.db does not exist. Complete Task 1 first.")
        return False

    import os
    if not os.environ.get("OPENAI_API_KEY"):
        print("SKIP: OPENAI_API_KEY not set. Set it to run Task 3 checks.")
        print("  export OPENAI_API_KEY=sk-...")
        return True  # Don't fail — just skip

    question = "How many books are currently in stock?"

    try:
        answer = run_agent_loop(question)
    except NotImplementedError:
        print("FAIL: run_agent_loop() not implemented yet")
        return False
    except Exception as exc:
        print(f"FAIL: run_agent_loop() raised {type(exc).__name__}: {exc}")
        return False

    if not isinstance(answer, str):
        print(f"FAIL: run_agent_loop() should return a string, got {type(answer)}")
        return False

    if len(answer.strip()) == 0:
        print("FAIL: run_agent_loop() returned an empty string")
        return False

    # Check the answer contains a number (basic sanity check)
    has_number = bool(re.search(r'\d+', answer))
    if not has_number:
        print(f"FAIL: Answer doesn't contain a number. Got: {answer[:200]}")
        return False

    print(f"PASS: run_agent_loop() returned an answer:")
    print(f"  Question: {question}")
    print(f"  Answer: {answer[:300]}")
    return True


# =============================================================================
# TASK 4: Score agent responses
# =============================================================================
#
# Implement score_agent_answer() — a simplified reward function for text-to-SQL.
#
# The scorer compares the agent's answer string to a ground truth string
# and returns a float between 0.0 and 1.0.
#
# Scoring rules (implement in this order):
#
#   1.0  — agent_answer contains all key terms from ground_truth
#           (case-insensitive word matching, ignoring common stop words)
#
#   0.5  — agent_answer contains at least 50% of the key terms
#
#   0.1  — agent_answer is non-empty but contains fewer than 50% of key terms
#
#   0.0  — agent_answer is empty or is exactly "None"
#
# Key terms: extract all alphanumeric tokens from ground_truth that are
#   longer than 2 characters and not in STOP_WORDS.
#
# The scorer is deliberately simple — in production you would use RULER
# (an LLM judge). This rule-based version is runnable without an API.
#
# =============================================================================

STOP_WORDS = {
    "the", "and", "for", "are", "was", "has", "have", "been",
    "with", "this", "that", "from", "all", "its", "but", "not",
    "who", "which", "when", "than", "more", "each",
}


def score_agent_answer(agent_answer: str, ground_truth: str) -> float:
    """
    Score the agent's answer against the ground truth.

    Returns a float in [0.0, 1.0].

    Args:
        agent_answer: The string the agent returned as its final answer
        ground_truth: The expected answer string

    Returns:
        1.0 if all key terms match, 0.5 if >=50% match,
        0.1 if some answer given, 0.0 if empty.
    """
    # YOUR CODE HERE
    # Hint: Use re.findall(r'\b\w+\b', text.lower()) to extract tokens
    # Hint: Filter tokens by len > 2 and not in STOP_WORDS
    raise NotImplementedError("Task 4: implement score_agent_answer()")


def check_task_4() -> bool:
    """Auto-check for Task 4. Run after implementing score_agent_answer()."""
    print("\n--- Task 4 Checks ---")
    passed = True

    test_cases = [
        # (agent_answer, ground_truth, expected_score_range, description)
        (
            "",
            "There are 7 books currently in stock.",
            (0.0, 0.0),
            "Empty answer → 0.0",
        ),
        (
            "None",
            "There are 7 books currently in stock.",
            (0.0, 0.0),
            "String 'None' → 0.0",
        ),
        (
            "I don't know.",
            "There are 7 books currently in stock.",
            (0.05, 0.15),
            "Answer with no matching terms → ~0.1",
        ),
        (
            "There are 7 books in the bookstore.",
            "There are 7 books currently in stock.",
            (0.45, 0.55),
            "Partial match → ~0.5",
        ),
        (
            "There are 7 books currently in stock.",
            "There are 7 books currently in stock.",
            (0.95, 1.01),
            "Exact match → 1.0",
        ),
        (
            "7 books are currently in stock in the bookstore inventory.",
            "There are 7 books currently in stock.",
            (0.95, 1.01),
            "All key terms present (different phrasing) → 1.0",
        ),
    ]

    for agent_answer, ground_truth, (low, high), description in test_cases:
        try:
            score = score_agent_answer(agent_answer, ground_truth)
        except NotImplementedError:
            print("FAIL: score_agent_answer() not implemented yet")
            return False
        except Exception as exc:
            print(f"FAIL: score_agent_answer() raised {type(exc).__name__}: {exc}")
            passed = False
            continue

        if not isinstance(score, float):
            print(f"FAIL: score_agent_answer() should return float, got {type(score)}")
            passed = False
        elif not (low <= score <= high):
            print(f"FAIL: {description}")
            print(f"      Expected score in [{low}, {high}], got {score:.3f}")
            passed = False
        else:
            print(f"PASS: {description} (score={score:.3f})")

    return passed


# =============================================================================
# BONUS: Run a mini training scenario
# =============================================================================
#
# This bonus section ties everything together: it runs multiple agent episodes
# on the same question, scores each response, and prints the score distribution.
#
# No GRPO update is performed — this is for observing how the scoring works
# and understanding why you need multiple completions per scenario.
#
# You need OPENAI_API_KEY set and Task 3 complete to run this.
# =============================================================================


def run_mini_scenario(
    question: str,
    ground_truth: str,
    num_completions: int = 4,
) -> None:
    """
    Run num_completions agent episodes on the same question.
    Print the score for each and compute the group mean and std.
    """
    import os
    import statistics

    if not os.environ.get("OPENAI_API_KEY"):
        print("SKIP: OPENAI_API_KEY not set.")
        return

    print(f"\nQuestion: {question}")
    print(f"Ground truth: {ground_truth}")
    print(f"Running {num_completions} completions...")

    scores = []
    for i in range(num_completions):
        try:
            answer = run_agent_loop(question)
            score = score_agent_answer(answer, ground_truth)
            scores.append(score)
            print(f"  Completion {i+1}: score={score:.2f} | answer={answer[:100]}")
        except NotImplementedError:
            print("  Completion {i+1}: SKIP (run_agent_loop not implemented)")
            return

    if scores:
        mean = sum(scores) / len(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        print(f"\nGroup mean: {mean:.3f}, std: {std:.3f}")
        if std > 0:
            advantages = [(s - mean) / std for s in scores]
            print("Advantages (A_i = (r_i - mean) / std):")
            for i, (score, adv) in enumerate(zip(scores, advantages)):
                print(f"  Completion {i+1}: reward={score:.2f}, advantage={adv:+.3f}")
        else:
            print("All completions scored the same — no gradient signal.")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 60)
    print("Module 06 Exercise: Text-to-SQL Agent")
    print("=" * 60)

    t1 = check_task_1()
    t2 = check_task_2()
    t3 = check_task_3()
    t4 = check_task_4()

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Task 1 (Database):     {'PASS' if t1 else 'FAIL'}")
    print(f"  Task 2 (Tools):        {'PASS' if t2 else 'FAIL'}")
    print(f"  Task 3 (Agent loop):   {'PASS' if t3 else 'FAIL'}")
    print(f"  Task 4 (Scorer):       {'PASS' if t4 else 'FAIL'}")
    print("=" * 60)

    if all([t1, t2, t3, t4]):
        print("\nAll tasks complete. Try the bonus scenario:")
        print()
        run_mini_scenario(
            question="How many books are currently in stock?",
            ground_truth="7 books are currently in stock.",
            num_completions=4,
        )


if __name__ == "__main__":
    main()
