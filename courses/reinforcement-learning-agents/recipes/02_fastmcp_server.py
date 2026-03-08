"""
Recipe: FastMCP Server in 30 Lines
====================================

Minimal MCP server exposing a SQLite database.
Copy this and modify for your own database or tool set.

Usage:
    python 02_fastmcp_server.py
    # Server starts on port 8000
"""

import sqlite3
from fastmcp import FastMCP

DB_PATH = "company.db"
mcp = FastMCP("my-db-server")


@mcp.tool()
def list_tables() -> str:
    """List all tables in the database."""
    conn = sqlite3.connect(DB_PATH)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    conn.close()
    return "\n".join(t[0] for t in tables)


@mcp.tool()
def describe_table(table_name: str) -> str:
    """Show column names and types for a table."""
    conn = sqlite3.connect(DB_PATH)
    cols = conn.execute(f"PRAGMA table_info([{table_name}])").fetchall()
    conn.close()
    return "\n".join(f"  {c[1]} ({c[2]})" for c in cols)


@mcp.tool()
def run_query(sql: str) -> str:
    """Execute a SELECT query and return results."""
    if not sql.strip().upper().startswith("SELECT"):
        return "Error: Only SELECT queries allowed."
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(sql).fetchall()
        return "\n".join(str(r) for r in rows[:50])
    except sqlite3.Error as e:
        return f"SQL Error: {e}"
    finally:
        conn.close()


if __name__ == "__main__":
    mcp.run(port=8000)
