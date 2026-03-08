"""
MCP Database Server Template
=============================

Production-ready MCP server for exposing a SQLite database to AI agents.
Provides three tools: list_tables, describe_table, run_query.

Usage:
    # Start the server
    python mcp_database_server.py --db company.db --port 8000

    # Or import and customize
    from mcp_database_server import create_server
    server = create_server("my_database.db")
"""

import sqlite3
import argparse
from contextlib import contextmanager
from pathlib import Path

from fastmcp import FastMCP


def create_server(db_path: str, server_name: str = "database-server") -> FastMCP:
    """
    Create an MCP server with database tools.

    Parameters
    ----------
    db_path : str
        Path to SQLite database file
    server_name : str
        Name for the MCP server

    Returns
    -------
    FastMCP server instance
    """
    mcp = FastMCP(server_name)
    db_file = Path(db_path)

    @contextmanager
    def get_connection():
        """Thread-safe database connection."""
        conn = sqlite3.connect(str(db_file))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    @mcp.tool()
    def list_tables() -> str:
        """List all tables in the database with their row counts."""
        with get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row["name"] for row in cursor.fetchall()]

            results = []
            for table in tables:
                count = conn.execute(f"SELECT COUNT(*) as c FROM [{table}]").fetchone()["c"]
                results.append(f"  {table} ({count} rows)")

            return f"Tables in database:\n" + "\n".join(results)

    @mcp.tool()
    def describe_table(table_name: str) -> str:
        """
        Show the schema (columns, types) for a specific table.

        Parameters
        ----------
        table_name : str
            Name of the table to describe
        """
        with get_connection() as conn:
            # Check table exists
            check = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            ).fetchone()

            if not check:
                return f"Error: Table '{table_name}' not found. Use list_tables() to see available tables."

            # Get column info
            cursor = conn.execute(f"PRAGMA table_info([{table_name}])")
            columns = cursor.fetchall()

            lines = [f"Schema for '{table_name}':"]
            lines.append(f"  {'Column':<20} {'Type':<15} {'Nullable':<10} {'PK'}")
            lines.append(f"  {'-'*20} {'-'*15} {'-'*10} {'-'*3}")

            for col in columns:
                nullable = "YES" if not col["notnull"] else "NO"
                pk = "YES" if col["pk"] else ""
                lines.append(f"  {col['name']:<20} {col['type']:<15} {nullable:<10} {pk}")

            # Sample rows
            sample = conn.execute(f"SELECT * FROM [{table_name}] LIMIT 3").fetchall()
            if sample:
                lines.append(f"\nSample rows (first 3):")
                col_names = [col["name"] for col in columns]
                lines.append(f"  {' | '.join(col_names)}")
                for row in sample:
                    lines.append(f"  {' | '.join(str(row[c]) for c in col_names)}")

            return "\n".join(lines)

    @mcp.tool()
    def run_query(sql: str) -> str:
        """
        Execute a SQL query and return results.

        Parameters
        ----------
        sql : str
            SQL query to execute. Only SELECT queries are allowed.
        """
        # Safety: only allow SELECT queries
        stripped = sql.strip().upper()
        if not stripped.startswith("SELECT"):
            return "Error: Only SELECT queries are allowed. Use SELECT to read data."

        # Block dangerous patterns
        dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "EXEC"]
        for keyword in dangerous:
            if keyword in stripped:
                return f"Error: {keyword} statements are not allowed."

        with get_connection() as conn:
            try:
                cursor = conn.execute(sql)
                rows = cursor.fetchall()

                if not rows:
                    return "Query returned no results."

                # Format as table
                columns = [desc[0] for desc in cursor.description]
                lines = [" | ".join(columns)]
                lines.append(" | ".join("-" * len(c) for c in columns))

                for row in rows[:50]:  # Limit to 50 rows
                    lines.append(" | ".join(str(row[i]) for i in range(len(columns))))

                result = "\n".join(lines)
                if len(rows) > 50:
                    result += f"\n\n... showing 50 of {len(rows)} rows"

                return result

            except sqlite3.Error as e:
                return f"SQL Error: {e}"

    return mcp


def create_sample_database(db_path: str) -> None:
    """
    Create a sample company database for testing.

    Creates three tables: departments, employees, projects.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.executescript("""
        DROP TABLE IF EXISTS departments;
        DROP TABLE IF EXISTS employees;
        DROP TABLE IF EXISTS projects;

        CREATE TABLE departments (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            location TEXT NOT NULL,
            budget REAL NOT NULL
        );

        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            salary REAL NOT NULL,
            hire_date TEXT NOT NULL,
            FOREIGN KEY (department_id) REFERENCES departments(id)
        );

        CREATE TABLE projects (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            lead_id INTEGER NOT NULL,
            department_id INTEGER NOT NULL,
            budget REAL NOT NULL,
            status TEXT NOT NULL,
            start_date TEXT NOT NULL,
            FOREIGN KEY (lead_id) REFERENCES employees(id),
            FOREIGN KEY (department_id) REFERENCES departments(id)
        );

        INSERT INTO departments VALUES
            (1, 'Engineering', 'San Francisco', 2500000),
            (2, 'Data Science', 'New York', 1800000),
            (3, 'Product', 'San Francisco', 1200000),
            (4, 'Marketing', 'Chicago', 900000),
            (5, 'Operations', 'Austin', 700000);

        INSERT INTO employees VALUES
            (1, 'Alice Chen', 1, 'Senior Engineer', 165000, '2021-03-15'),
            (2, 'Bob Martinez', 1, 'Staff Engineer', 195000, '2019-07-01'),
            (3, 'Carol Wang', 2, 'Data Scientist', 145000, '2022-01-10'),
            (4, 'David Kim', 2, 'Senior Data Scientist', 170000, '2020-09-20'),
            (5, 'Eva Patel', 3, 'Product Manager', 155000, '2021-06-01'),
            (6, 'Frank Lee', 1, 'Engineer', 130000, '2023-02-14'),
            (7, 'Grace Zhou', 4, 'Marketing Manager', 125000, '2022-04-01'),
            (8, 'Henry Brown', 5, 'Operations Lead', 115000, '2021-11-15'),
            (9, 'Iris Nakamura', 2, 'ML Engineer', 160000, '2022-08-01'),
            (10, 'Jack Wilson', 3, 'Senior PM', 175000, '2020-03-10');

        INSERT INTO projects VALUES
            (1, 'ML Pipeline v2', 2, 1, 450000, 'active', '2024-01-15'),
            (2, 'Customer Churn Model', 4, 2, 200000, 'active', '2024-03-01'),
            (3, 'Platform Redesign', 5, 3, 350000, 'active', '2024-02-01'),
            (4, 'Data Warehouse Migration', 9, 2, 600000, 'active', '2023-11-01'),
            (5, 'Brand Refresh', 7, 4, 150000, 'completed', '2023-09-01'),
            (6, 'Microservices Refactor', 1, 1, 500000, 'active', '2024-04-01'),
            (7, 'Analytics Dashboard', 3, 2, 180000, 'planning', '2024-06-01');
    """)

    conn.commit()
    conn.close()
    print(f"Sample database created at: {db_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP Database Server")
    parser.add_argument("--db", default="company.db", help="Path to SQLite database")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument(
        "--create-sample", action="store_true", help="Create sample database first"
    )
    args = parser.parse_args()

    if args.create_sample:
        create_sample_database(args.db)

    server = create_server(args.db)
    server.run(port=args.port)
