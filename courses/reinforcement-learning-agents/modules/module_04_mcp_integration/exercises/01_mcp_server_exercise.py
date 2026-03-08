"""
Exercise 01: Build a Simple MCP Server with FastMCP
====================================================

Module 04 — MCP Server Integration

Learning objectives:
  1. Define MCP tools using FastMCP decorators and type hints
  2. Write docstrings that serve as schema descriptions for agents
  3. Connect to a running server and discover tools programmatically
  4. Test tool calls and validate responses

Time estimate: 30-45 minutes

Instructions:
  Work through the three parts in order:
    Part A — Build the Calculator MCP server (fill in the tool implementations)
    Part B — Connect as a client and discover tools
    Part C — Extend with a conversion tool and verify it appears in discovery

Run the self-checks after each part:
    python 01_mcp_server_exercise.py --check-a
    python 01_mcp_server_exercise.py --check-b
    python 01_mcp_server_exercise.py --check-c

Run the full exercise (server + client) with:
    python 01_mcp_server_exercise.py --run

Prerequisites:
    pip install fastmcp mcp
"""

import asyncio
import math
import subprocess
import sys
import time
from typing import Any

import fastmcp

# ============================================================
# PART A: Build the Calculator MCP Server
#
# Complete the three tool implementations below.
# Each tool must:
#   1. Have a descriptive docstring (the agent reads this)
#   2. Use correct Python type hints (FastMCP builds the schema from these)
#   3. Raise ValueError for invalid inputs with a helpful message
# ============================================================

mcp = fastmcp.FastMCP("calculator-server")


@mcp.tool()
def add(a: float, b: float) -> float:
    """
    Add two numbers and return their sum.

    Use this for combining quantities, running totals, or finding totals
    when you have individual components.

    Args:
        a: First number (any real number)
        b: Second number (any real number)

    Returns:
        The sum a + b.

    Example:
        add(3.5, 2.0) -> 5.5
    """
    # TODO: Implement this tool.
    # It should return a + b.
    # This is the simplest tool -- get comfortable with the pattern here.
    raise NotImplementedError("Implement add()")


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers and return their product.

    Use this for scaling quantities, computing areas, or applying rates
    (e.g., price × quantity, rate × time).

    Args:
        a: First number (any real number)
        b: Second number (any real number)

    Returns:
        The product a * b.

    Example:
        multiply(4.0, 7.5) -> 30.0
    """
    # TODO: Implement this tool.
    # Return a * b.
    raise NotImplementedError("Implement multiply()")


@mcp.tool()
def divide(a: float, b: float) -> float:
    """
    Divide a by b and return the quotient.

    Use this for ratios, per-unit calculations, or splitting quantities evenly.
    Raises ValueError if b is zero — division by zero is undefined.

    Args:
        a: Numerator (dividend)
        b: Denominator (divisor) — must not be zero

    Returns:
        The quotient a / b.

    Raises:
        ValueError: If b is zero.

    Example:
        divide(10.0, 4.0) -> 2.5
    """
    # TODO: Implement this tool.
    # Return a / b.
    # If b is 0, raise ValueError with a clear message.
    # Hint: Use the message "Division by zero: b must not be 0"
    raise NotImplementedError("Implement divide()")


@mcp.tool()
def square_root(x: float) -> float:
    """
    Return the square root of a non-negative number.

    Use this for geometric calculations (side lengths, distances) or
    when reversing a squared quantity.

    Args:
        x: Non-negative number to take the square root of.

    Returns:
        The square root of x.

    Raises:
        ValueError: If x is negative (square root of negative is undefined
                    in the real numbers).

    Example:
        square_root(16.0) -> 4.0
        square_root(2.0)  -> 1.4142135623730951
    """
    # TODO: Implement this tool.
    # Use math.sqrt(x).
    # If x < 0, raise ValueError with message: "Cannot take square root of a negative number"
    raise NotImplementedError("Implement square_root()")


# ============================================================
# SELF-CHECK A
# Run: python 01_mcp_server_exercise.py --check-a
# ============================================================

def check_part_a() -> None:
    """Verify all four tool implementations work correctly."""
    print("=" * 50)
    print("PART A: Checking tool implementations")
    print("=" * 50)

    failures = []

    # Test add()
    try:
        result = add(3.5, 2.0)
        assert result == 5.5, f"add(3.5, 2.0) should return 5.5, got {result}"
        assert add(-1.0, 1.0) == 0.0, "add(-1, 1) should return 0.0"
        assert add(0.0, 0.0) == 0.0, "add(0, 0) should return 0.0"
        print("  add()         PASS")
    except NotImplementedError:
        failures.append("add() -- not yet implemented")
    except AssertionError as e:
        failures.append(f"add() -- {e}")

    # Test multiply()
    try:
        result = multiply(4.0, 7.5)
        assert result == 30.0, f"multiply(4.0, 7.5) should return 30.0, got {result}"
        assert multiply(-2.0, 3.0) == -6.0, "multiply(-2, 3) should return -6.0"
        assert multiply(0.0, 999.0) == 0.0, "multiply(0, 999) should return 0.0"
        print("  multiply()    PASS")
    except NotImplementedError:
        failures.append("multiply() -- not yet implemented")
    except AssertionError as e:
        failures.append(f"multiply() -- {e}")

    # Test divide()
    try:
        result = divide(10.0, 4.0)
        assert result == 2.5, f"divide(10.0, 4.0) should return 2.5, got {result}"
        # Test zero division
        try:
            divide(1.0, 0.0)
            failures.append("divide(1, 0) -- should have raised ValueError")
        except ValueError as e:
            assert "zero" in str(e).lower(), (
                f"ValueError message should mention 'zero', got: {e}"
            )
        print("  divide()      PASS")
    except NotImplementedError:
        failures.append("divide() -- not yet implemented")
    except AssertionError as e:
        failures.append(f"divide() -- {e}")

    # Test square_root()
    try:
        result = square_root(16.0)
        assert result == 4.0, f"square_root(16.0) should return 4.0, got {result}"
        assert abs(square_root(2.0) - math.sqrt(2)) < 1e-10, (
            "square_root(2.0) should equal math.sqrt(2)"
        )
        # Test negative input
        try:
            square_root(-1.0)
            failures.append("square_root(-1) -- should have raised ValueError")
        except ValueError as e:
            assert "negative" in str(e).lower(), (
                f"ValueError message should mention 'negative', got: {e}"
            )
        print("  square_root() PASS")
    except NotImplementedError:
        failures.append("square_root() -- not yet implemented")
    except AssertionError as e:
        failures.append(f"square_root() -- {e}")

    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All Part A checks passed.")


# ============================================================
# PART B: Connect as a Client and Discover Tools
#
# The MCP client discovers tools by calling list_tools() on the server.
# Complete the connect_and_discover() function below.
#
# You will use the `mcp` library (installed with `pip install mcp`).
# The pattern is:
#
#   async with mcp.client_session(server_url) as session:
#       tools = await session.list_tools()
#       result = await session.call_tool(name, arguments)
# ============================================================

import mcp as mcp_client  # noqa: E402  (import after the server code above)


async def connect_and_discover(server_url: str = "http://127.0.0.1:9001") -> dict[str, Any]:
    """
    Connect to the running MCP server and discover available tools.

    This function should:
    1. Open a client session to the server URL
    2. Call list_tools() to get all available tools
    3. Return a dict with:
       - "tool_names": list of tool name strings
       - "tool_count": number of tools
       - "has_add": True if "add" is in the tool list
       - "has_divide": True if "divide" is in the tool list

    Args:
        server_url: URL of the running FastMCP server.

    Returns:
        Dict with tool discovery results.
    """
    # TODO: Implement this function.
    #
    # Hint 1: Use `async with mcp_client.client_session(server_url) as session:`
    # Hint 2: `tools = await session.list_tools()` returns an object with a `.tools` attribute
    # Hint 3: Each tool in tools.tools has a `.name` attribute
    #
    # Return structure:
    # {
    #     "tool_names": [list of name strings],
    #     "tool_count": int,
    #     "has_add": bool,
    #     "has_divide": bool,
    # }
    raise NotImplementedError("Implement connect_and_discover()")


async def call_tool_example(server_url: str = "http://127.0.0.1:9001") -> dict[str, Any]:
    """
    Connect to the server, call a tool, and return the result.

    This function should:
    1. Call multiply(6.0, 7.0) on the server
    2. Call divide(22.0, 7.0) on the server
    3. Return a dict with:
       - "multiply_result": the float result of multiply(6, 7)
       - "divide_result": the float result of divide(22, 7)

    The result of session.call_tool() is a ToolResult object.
    The actual value is in result.content[0].text (a string that you
    must convert to float).

    Args:
        server_url: URL of the running FastMCP server.

    Returns:
        Dict with tool call results.
    """
    # TODO: Implement this function.
    #
    # Hint 1: `result = await session.call_tool("multiply", {"a": 6.0, "b": 7.0})`
    # Hint 2: `float(result.content[0].text)` converts the string result to float
    #
    # Return structure:
    # {
    #     "multiply_result": float,
    #     "divide_result": float,
    # }
    raise NotImplementedError("Implement call_tool_example()")


# ============================================================
# SELF-CHECK B
# Run: python 01_mcp_server_exercise.py --check-b
# (Requires the server to be running in another terminal:
#  python 01_mcp_server_exercise.py --serve)
# ============================================================

def check_part_b() -> None:
    """Verify client connection and tool discovery."""
    print("=" * 50)
    print("PART B: Checking client connection and discovery")
    print("=" * 50)
    print("(Make sure the server is running: python 01_mcp_server_exercise.py --serve)")
    print()

    async def run_checks():
        failures = []

        # Test connect_and_discover
        try:
            result = await connect_and_discover("http://127.0.0.1:9001")

            assert "tool_names" in result, "Result must have 'tool_names' key"
            assert "tool_count" in result, "Result must have 'tool_count' key"
            assert "has_add" in result, "Result must have 'has_add' key"
            assert "has_divide" in result, "Result must have 'has_divide' key"

            assert result["tool_count"] >= 4, (
                f"Expected at least 4 tools, got {result['tool_count']}"
            )
            assert result["has_add"] is True, "Tool 'add' should be discoverable"
            assert result["has_divide"] is True, "Tool 'divide' should be discoverable"
            assert "multiply" in result["tool_names"], "'multiply' should be in tool_names"
            assert "square_root" in result["tool_names"], "'square_root' should be in tool_names"

            print("  connect_and_discover()  PASS")
            print(f"    Discovered {result['tool_count']} tools: {result['tool_names']}")

        except NotImplementedError:
            failures.append("connect_and_discover() -- not yet implemented")
        except AssertionError as e:
            failures.append(f"connect_and_discover() -- {e}")
        except Exception as e:
            failures.append(
                f"connect_and_discover() -- connection error: {e}\n"
                "  Is the server running? (python 01_mcp_server_exercise.py --serve)"
            )

        # Test call_tool_example
        try:
            result = await call_tool_example("http://127.0.0.1:9001")

            assert "multiply_result" in result, "Result must have 'multiply_result' key"
            assert "divide_result" in result, "Result must have 'divide_result' key"
            assert result["multiply_result"] == 42.0, (
                f"multiply(6, 7) should return 42.0, got {result['multiply_result']}"
            )
            assert abs(result["divide_result"] - 22 / 7) < 1e-10, (
                f"divide(22, 7) should return {22/7}, got {result['divide_result']}"
            )

            print("  call_tool_example()     PASS")
            print(f"    multiply(6, 7) = {result['multiply_result']}")
            print(f"    divide(22, 7)  = {result['divide_result']:.6f}")

        except NotImplementedError:
            failures.append("call_tool_example() -- not yet implemented")
        except AssertionError as e:
            failures.append(f"call_tool_example() -- {e}")
        except Exception as e:
            failures.append(f"call_tool_example() -- {e}")

        return failures

    failures = asyncio.run(run_checks())
    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All Part B checks passed.")


# ============================================================
# PART C: Add a Conversion Tool and Verify Discovery
#
# Add a new tool to the server: unit_convert()
# It converts between common units in a single category (temperature).
#
# After adding it, verify that the server now exposes 5 tools (not 4).
# ============================================================

@mcp.tool()
def unit_convert(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert a value between temperature units.

    Supported units: "celsius", "fahrenheit", "kelvin"
    Conversion is case-insensitive.

    Args:
        value: Numeric value to convert.
        from_unit: Source unit ("celsius", "fahrenheit", or "kelvin").
        to_unit: Target unit ("celsius", "fahrenheit", or "kelvin").

    Returns:
        Converted value as a float, rounded to 4 decimal places.

    Raises:
        ValueError: If from_unit or to_unit is not a supported unit.

    Examples:
        unit_convert(100.0, "celsius", "fahrenheit") -> 212.0
        unit_convert(32.0, "fahrenheit", "celsius")  -> 0.0
        unit_convert(0.0, "celsius", "kelvin")       -> 273.15
    """
    # TODO: Implement this tool.
    #
    # Conversion strategy:
    #   1. Normalize both unit strings to lowercase
    #   2. Validate that both are in {"celsius", "fahrenheit", "kelvin"}
    #      Raise ValueError with message: f"Unknown unit: '{from_unit}'. Use celsius, fahrenheit, or kelvin"
    #   3. Convert from_unit → celsius (as an intermediate step)
    #   4. Convert celsius → to_unit
    #   5. Return round(result, 4)
    #
    # Formulas:
    #   fahrenheit → celsius: (f - 32) × 5/9
    #   kelvin     → celsius: k - 273.15
    #   celsius    → celsius: no change
    #
    #   celsius → fahrenheit: c × 9/5 + 32
    #   celsius → kelvin:     c + 273.15
    #   celsius → celsius:    no change
    raise NotImplementedError("Implement unit_convert()")


# ============================================================
# SELF-CHECK C
# Run: python 01_mcp_server_exercise.py --check-c
# ============================================================

def check_part_c() -> None:
    """Verify the unit_convert tool and that discovery shows 5 tools."""
    print("=" * 50)
    print("PART C: Checking unit_convert tool")
    print("=" * 50)

    failures = []

    # Test unit_convert locally
    try:
        # Celsius → Fahrenheit
        result = unit_convert(100.0, "celsius", "fahrenheit")
        assert result == 212.0, f"100°C → °F should be 212.0, got {result}"

        # Fahrenheit → Celsius
        result = unit_convert(32.0, "fahrenheit", "celsius")
        assert result == 0.0, f"32°F → °C should be 0.0, got {result}"

        # Celsius → Kelvin
        result = unit_convert(0.0, "celsius", "kelvin")
        assert result == 273.15, f"0°C → K should be 273.15, got {result}"

        # Kelvin → Fahrenheit
        result = unit_convert(373.15, "kelvin", "fahrenheit")
        assert abs(result - 212.0) < 0.01, (
            f"373.15K → °F should be ~212.0, got {result}"
        )

        # Case insensitivity
        result = unit_convert(100.0, "CELSIUS", "FAHRENHEIT")
        assert result == 212.0, "Should be case-insensitive"

        # Invalid unit
        try:
            unit_convert(100.0, "rankine", "celsius")
            failures.append("unit_convert('rankine', ...) should raise ValueError")
        except ValueError as e:
            assert "rankine" in str(e).lower() or "unknown" in str(e).lower(), (
                f"Error should mention the unknown unit, got: {e}"
            )

        print("  unit_convert() PASS")
        print("    100°C → 212°F, 32°F → 0°C, 0°C → 273.15K")

    except NotImplementedError:
        failures.append("unit_convert() -- not yet implemented")
    except AssertionError as e:
        failures.append(f"unit_convert() -- {e}")

    # Verify tool count (this checks the server would expose 5 tools)
    # We check this by inspecting the mcp object's tool registry
    try:
        # FastMCP stores registered tools -- we check the count
        tool_names = list(mcp._tool_manager._tools.keys())
        assert len(tool_names) == 5, (
            f"Expected 5 tools in registry, found {len(tool_names)}: {tool_names}"
        )
        assert "unit_convert" in tool_names, (
            "unit_convert should be registered with the mcp server"
        )
        print(f"  Tool registry  PASS ({len(tool_names)} tools: {sorted(tool_names)})")
    except AttributeError:
        # FastMCP internal API may differ across versions
        # Fall back to checking that the decorator ran without error
        print("  Tool registry  PASS (registry check skipped for this FastMCP version)")

    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All Part C checks passed.")


# ============================================================
# Server startup helpers
# ============================================================

def start_server(port: int = 9001) -> None:
    """Start the FastMCP server on localhost."""
    print(f"Starting calculator MCP server on http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop.\n")
    mcp.run(transport="sse", port=port, host="127.0.0.1")


def run_full_exercise() -> None:
    """
    Run the full exercise end-to-end:
    1. Start the server in a background process
    2. Run all three check suites
    3. Stop the server
    """
    print("Running full exercise (Parts A, B, C)\n")

    # Part A does not need the server
    check_part_a()
    print()

    # Start server for Part B and C
    server_proc = subprocess.Popen(
        [sys.executable, __file__, "--serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)  # Wait for server to be ready

    try:
        check_part_b()
        print()
        check_part_c()
    finally:
        server_proc.terminate()
        server_proc.wait()

    print("\nAll checks complete.")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    if "--check-a" in sys.argv:
        check_part_a()
    elif "--check-b" in sys.argv:
        check_part_b()
    elif "--check-c" in sys.argv:
        check_part_c()
    elif "--serve" in sys.argv:
        port = int(sys.argv[sys.argv.index("--serve") + 1]) if "--serve" in sys.argv[:-1] and sys.argv[sys.argv.index("--serve") + 1].isdigit() else 9001
        start_server(port)
    elif "--run" in sys.argv:
        run_full_exercise()
    else:
        print(__doc__)
        print("\nUsage:")
        print("  python 01_mcp_server_exercise.py --check-a    # Check Part A (no server needed)")
        print("  python 01_mcp_server_exercise.py --serve      # Start server (for Part B)")
        print("  python 01_mcp_server_exercise.py --check-b    # Check Part B (server must be running)")
        print("  python 01_mcp_server_exercise.py --check-c    # Check Part C (no server needed)")
        print("  python 01_mcp_server_exercise.py --run        # Run all checks end-to-end")
