"""
Module 04 — Control Flow Exercise
Power Automate: Branching, Loops, and Error Handling

This exercise has three parts:

PART 1 — Pattern Identification
    Given a scenario description, identify the correct Power Automate
    control flow action to use.

PART 2 — Python Equivalents
    Implement Python functions that behave like the specified
    Power Automate control flow patterns.

PART 3 — Flow Debugging
    A simulated flow has three bugs in its control flow logic.
    Read the run history output, identify the bugs, and fix them.

Run this file to check your answers:
    python 01_control_flow_exercise.py

All checks print PASS or FAIL with an explanation.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Shared infrastructure (do not modify)
# ---------------------------------------------------------------------------

class ActionStatus(Enum):
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    SKIPPED = "Skipped"
    TIMED_OUT = "Timed Out"


@dataclass
class ActionResult:
    name: str
    status: ActionStatus
    output: Any = None
    error: Optional[str] = None


@dataclass
class RunHistory:
    actions: List[ActionResult] = field(default_factory=list)

    def record(self, name: str, status: ActionStatus, output: Any = None, error: str = None) -> ActionResult:
        result = ActionResult(name=name, status=status, output=output, error=error)
        self.actions.append(result)
        return result

    def statuses(self) -> List[str]:
        return [r.status.value for r in self.actions]

    def outputs_by_name(self) -> Dict[str, Any]:
        return {r.name: r.output for r in self.actions if r.output is not None}

    def print_history(self) -> None:
        icons = {"Succeeded": "✓", "Failed": "✗", "Skipped": "⊘", "Timed Out": "⏱"}
        for r in self.actions:
            icon = icons.get(r.status.value, "?")
            line = f"  {icon} {r.name}: {r.status.value}"
            if r.error:
                line += f" — {r.error}"
            print(line)


def check(label: str, condition: bool, explanation: str = "") -> None:
    """Print PASS or FAIL for a single check."""
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition and explanation:
        print(f"       {explanation}")


# ---------------------------------------------------------------------------
# PART 1: Pattern Identification
# ---------------------------------------------------------------------------
# For each scenario below, set the variable to the correct pattern name.
# Choose from: "Condition", "Switch", "Parallel Branch",
#              "Apply to Each", "Do Until", "Scope", "Terminate"

print("=" * 60)
print("PART 1: Pattern Identification")
print("=" * 60)

# Scenario 1:
# A flow receives a new support ticket. If the severity is "Critical",
# send an immediate Teams alert. If it is not Critical, create a standard
# work item. Two paths, binary decision.
SCENARIO_1_ANSWER = ""  # Replace with the pattern name

# Scenario 2:
# A flow must send onboarding emails to every new employee added to
# a SharePoint list in the last 7 days. There are between 1 and 50 new
# employees per week.
SCENARIO_2_ANSWER = ""  # Replace with the pattern name

# Scenario 3:
# A flow submits a document to an external rendering API and must wait
# for the render job to finish. The rendering takes between 10 seconds
# and 3 minutes depending on document size. The flow polls until the
# status field equals "Ready".
SCENARIO_3_ANSWER = ""  # Replace with the pattern name

# Scenario 4:
# A flow processes an order. When the order is placed, three things need
# to happen at the same time: send a confirmation email to the customer,
# notify the warehouse, and update the inventory system. All three are
# independent and should not wait for each other.
SCENARIO_4_ANSWER = ""  # Replace with the pattern name

# Scenario 5:
# A flow receives an HTTP request with a "region" field. The region can be
# "EMEA", "APAC", "AMER", or "LATAM". Each region has a different approval
# contact. There are four distinct routing paths.
SCENARIO_5_ANSWER = ""  # Replace with the pattern name

# Scenario 6:
# A flow calls an external payment API. The API occasionally returns a 503
# Service Unavailable error. The flow must catch this error, log it, and
# send an alert to the finance team without crashing the entire run.
SCENARIO_6_ANSWER = ""  # Replace with the pattern name

# Scenario 7:
# At the start of a flow, the code checks whether the triggering item has
# been deleted since the trigger fired. If it has been deleted, there is
# nothing to process and the flow should stop immediately and be marked
# as succeeded (not an error — just nothing to do).
SCENARIO_7_ANSWER = ""  # Replace with the pattern name


CORRECT_ANSWERS_PART1 = {
    "SCENARIO_1_ANSWER": "Condition",
    "SCENARIO_2_ANSWER": "Apply to Each",
    "SCENARIO_3_ANSWER": "Do Until",
    "SCENARIO_4_ANSWER": "Parallel Branch",
    "SCENARIO_5_ANSWER": "Switch",
    "SCENARIO_6_ANSWER": "Scope",
    "SCENARIO_7_ANSWER": "Terminate",
}

STUDENT_ANSWERS_PART1 = {
    "SCENARIO_1_ANSWER": SCENARIO_1_ANSWER,
    "SCENARIO_2_ANSWER": SCENARIO_2_ANSWER,
    "SCENARIO_3_ANSWER": SCENARIO_3_ANSWER,
    "SCENARIO_4_ANSWER": SCENARIO_4_ANSWER,
    "SCENARIO_5_ANSWER": SCENARIO_5_ANSWER,
    "SCENARIO_6_ANSWER": SCENARIO_6_ANSWER,
    "SCENARIO_7_ANSWER": SCENARIO_7_ANSWER,
}

for key, correct in CORRECT_ANSWERS_PART1.items():
    student = STUDENT_ANSWERS_PART1[key]
    num = key.replace("SCENARIO_", "").replace("_ANSWER", "")
    check(
        f"Scenario {num}: correct pattern",
        student.strip().lower() == correct.lower(),
        f"Expected '{correct}', got '{student}'"
    )


# ---------------------------------------------------------------------------
# PART 2: Python Equivalents of Power Automate Patterns
# ---------------------------------------------------------------------------
# Implement each function so that its tests pass.

print()
print("=" * 60)
print("PART 2: Python Equivalents")
print("=" * 60)


# --------------------------------------------------------------------------
# Exercise 2.1 — Condition pattern
# --------------------------------------------------------------------------
# Implement a function that behaves like a Power Automate Condition action.
#
# Rules:
# - If amount >= 10_000: return "CFO approval required"
# - If 1_000 <= amount < 10_000: return "Manager approval required"
# - If amount < 1_000: return "Auto-approved"
#
# This is a nested condition: the first check is >= 10_000, and the No
# branch contains the second check.

def route_approval(amount: float) -> str:
    """
    Return the correct approval routing string for the given amount.

    Parameters
    ----------
    amount : float
        Invoice or expense amount in USD.

    Returns
    -------
    str
        One of: "CFO approval required", "Manager approval required",
        "Auto-approved"
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement route_approval()")


# Tests for 2.1
def test_route_approval() -> None:
    cases = [
        (500.00, "Auto-approved"),
        (999.99, "Auto-approved"),
        (1_000.00, "Manager approval required"),
        (5_000.00, "Manager approval required"),
        (9_999.99, "Manager approval required"),
        (10_000.00, "CFO approval required"),
        (50_000.00, "CFO approval required"),
    ]
    for amount, expected in cases:
        result = route_approval(amount)
        check(
            f"route_approval({amount}) == '{expected}'",
            result == expected,
            f"Got '{result}'"
        )


try:
    test_route_approval()
except NotImplementedError:
    check("route_approval — implemented", False, "NotImplementedError: function not yet implemented")
except Exception:
    check("route_approval — no runtime error", False, traceback.format_exc())


# --------------------------------------------------------------------------
# Exercise 2.2 — Switch pattern
# --------------------------------------------------------------------------
# Implement a function that behaves like a Power Automate Switch action.
#
# Given a ticket category string, return the Teams channel name to post to.
#
# Case mapping:
#   "Hardware"  → "#it-hardware"
#   "Software"  → "#it-software"
#   "Network"   → "#it-network"
#   "Security"  → "#it-security"
#   anything else → "#it-general"   (the Default case)

def get_ticket_channel(category: str) -> str:
    """
    Return the Teams channel for the given ticket category.

    Parameters
    ----------
    category : str
        Ticket category from the form submission.

    Returns
    -------
    str
        Teams channel name including the # prefix.
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement get_ticket_channel()")


# Tests for 2.2
def test_get_ticket_channel() -> None:
    cases = [
        ("Hardware", "#it-hardware"),
        ("Software", "#it-software"),
        ("Network", "#it-network"),
        ("Security", "#it-security"),
        ("Printing", "#it-general"),
        ("Password Reset", "#it-general"),
        ("", "#it-general"),
    ]
    for category, expected in cases:
        result = get_ticket_channel(category)
        check(
            f"get_ticket_channel('{category}') == '{expected}'",
            result == expected,
            f"Got '{result}'"
        )


try:
    test_get_ticket_channel()
except NotImplementedError:
    check("get_ticket_channel — implemented", False, "NotImplementedError: function not yet implemented")
except Exception:
    check("get_ticket_channel — no runtime error", False, traceback.format_exc())


# --------------------------------------------------------------------------
# Exercise 2.3 — Apply to Each pattern
# --------------------------------------------------------------------------
# Implement a function that behaves like Apply to Each.
#
# Given a list of employee records, return a list of personalized
# reminder strings. Skip any record where the email field is empty or None.
#
# Each reminder string format:
#   "Dear {name}, your training is due by {due_date}."
#
# Records with an empty or None email must be silently skipped.

def build_reminders(employees: List[Dict[str, Any]]) -> List[str]:
    """
    Build reminder strings for employees with valid emails.

    Parameters
    ----------
    employees : list of dict
        Each dict has keys: "name" (str), "email" (str or None),
        "due_date" (str, e.g. "2024-04-01").

    Returns
    -------
    list of str
        Reminder strings for employees with non-empty, non-None emails.
        Employees with empty or None emails are excluded.
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement build_reminders()")


# Tests for 2.3
def test_build_reminders() -> None:
    employees = [
        {"name": "Alice Chen", "email": "alice@contoso.com", "due_date": "2024-04-01"},
        {"name": "Bob Rivera", "email": None, "due_date": "2024-04-01"},
        {"name": "Carol Kim", "email": "carol@contoso.com", "due_date": "2024-04-15"},
        {"name": "Dan Okafor", "email": "", "due_date": "2024-04-15"},
        {"name": "Eva Santos", "email": "eva@contoso.com", "due_date": "2024-04-30"},
    ]
    result = build_reminders(employees)

    check(
        "build_reminders: correct count (3 valid emails)",
        len(result) == 3,
        f"Expected 3 reminders, got {len(result)}"
    )
    check(
        "build_reminders: Alice included",
        any("Alice Chen" in r for r in result),
        "Reminder for Alice Chen not found"
    )
    check(
        "build_reminders: Bob excluded (None email)",
        not any("Bob Rivera" in r for r in result),
        "Bob Rivera should be excluded (None email)"
    )
    check(
        "build_reminders: Dan excluded (empty email)",
        not any("Dan Okafor" in r for r in result),
        "Dan Okafor should be excluded (empty email)"
    )
    check(
        "build_reminders: correct format",
        result[0] == "Dear Alice Chen, your training is due by 2024-04-01.",
        f"Unexpected format: '{result[0]}'"
    )


try:
    test_build_reminders()
except NotImplementedError:
    check("build_reminders — implemented", False, "NotImplementedError: function not yet implemented")
except Exception:
    check("build_reminders — no runtime error", False, traceback.format_exc())


# --------------------------------------------------------------------------
# Exercise 2.4 — Do Until pattern
# --------------------------------------------------------------------------
# Implement a function that behaves like Do Until.
#
# Simulate polling a job status. The job_statuses list contains the status
# returned on each poll attempt (index 0 = first poll, index 1 = second, etc.).
#
# Poll until the status is "Completed" or until max_polls attempts are made.
# Return a tuple: (final_status: str, polls_made: int)
#
# If the status becomes "Completed" on poll N, return ("Completed", N).
# If max_polls is reached without "Completed", return the last status and max_polls.

def poll_until_complete(
    job_statuses: List[str],
    max_polls: int = 10,
) -> Tuple[str, int]:
    """
    Simulate Do Until polling behavior.

    Parameters
    ----------
    job_statuses : list of str
        Sequence of status values returned on each poll.
        If the list is shorter than max_polls, the last value repeats.
    max_polls : int
        Maximum number of polls before stopping (equivalent to Do Until Count limit).

    Returns
    -------
    tuple of (str, int)
        (final_status, number_of_polls_made)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement poll_until_complete()")


# Tests for 2.4
def test_poll_until_complete() -> None:
    # Case 1: Completes on poll 3
    status, polls = poll_until_complete(
        ["Running", "Running", "Completed", "Completed"],
        max_polls=10,
    )
    check(
        "poll_until_complete: completes on poll 3",
        status == "Completed" and polls == 3,
        f"Expected ('Completed', 3), got ('{status}', {polls})"
    )

    # Case 2: Never completes — hits max_polls
    status, polls = poll_until_complete(
        ["Running", "Running", "Running"],
        max_polls=5,
    )
    check(
        "poll_until_complete: hits max_polls limit",
        polls == 5,
        f"Expected 5 polls (max), got {polls}"
    )
    check(
        "poll_until_complete: returns last status when limit hit",
        status == "Running",
        f"Expected 'Running', got '{status}'"
    )

    # Case 3: Completes on first poll
    status, polls = poll_until_complete(["Completed"], max_polls=10)
    check(
        "poll_until_complete: completes on first poll",
        status == "Completed" and polls == 1,
        f"Expected ('Completed', 1), got ('{status}', {polls})"
    )


try:
    test_poll_until_complete()
except NotImplementedError:
    check("poll_until_complete — implemented", False, "NotImplementedError: function not yet implemented")
except Exception:
    check("poll_until_complete — no runtime error", False, traceback.format_exc())


# ---------------------------------------------------------------------------
# PART 3: Flow Debugging
# ---------------------------------------------------------------------------
# The flow below has THREE bugs in its control flow logic.
# Read the EXPECTED behavior described in the docstring, then compare it
# to what the flow actually does.
#
# Find and fix all three bugs. The tests will tell you which checks fail.
#
# EXPECTED BEHAVIOR:
# - Process each expense report in BUGGY_EXPENSES
# - Reports under $500: tag as "low_value" and auto-approve
# - Reports $500 and above: tag as "high_value" and route to manager
# - If the approval action raises an exception, record it as "error" in the results
#   and continue processing the remaining reports (do not stop the loop)
# - After the loop, return a summary dict:
#     {"processed": N, "low_value": N, "high_value": N, "errors": N}

print()
print("=" * 60)
print("PART 3: Flow Debugging")
print("=" * 60)

BUGGY_EXPENSES = [
    {"id": "E01", "amount": 120.00, "submitter": "Alice"},
    {"id": "E02", "amount": 750.00, "submitter": "Bob"},
    {"id": "E03", "amount": 45.00, "submitter": "Carol"},
    {"id": "E04", "amount": 1200.00, "submitter": "Dan"},
    {"id": "E05", "amount": 500.00, "submitter": "Eva"},
]


def approve_expense(expense: dict) -> str:
    """
    Simulates the approval connector action.
    Raises RuntimeError for expenses over $1,000 (simulates API failure).
    """
    if expense["amount"] > 1000:
        raise RuntimeError(f"Approval system rejected {expense['id']}: amount exceeds single-transaction limit")
    return f"Approved: {expense['id']}"


def buggy_expense_processor(expenses: List[dict]) -> dict:
    """
    Process expense reports.

    BUG NOTICE: This function has exactly three bugs. Find and fix them.

    Returns a summary dict:
        {"processed": int, "low_value": int, "high_value": int, "errors": int}
    """
    summary = {"processed": 0, "low_value": 0, "high_value": 0, "errors": 0}

    for expense in expenses:
        summary["processed"] += 1

        # BUG AREA 1: Condition threshold check
        # Expected: low_value for amount < 500; high_value for amount >= 500
        if expense["amount"] > 500:       # <-- is this threshold correct?
            summary["low_value"] += 1
            tag = "low_value"
        else:
            summary["high_value"] += 1
            tag = "high_value"

        # BUG AREA 2: Error handling
        # Expected: exceptions from approve_expense() should be caught,
        #           recorded as "errors", and the loop should continue.
        result = approve_expense(expense)  # <-- what happens if this raises?
        summary["errors"] += 0            # This line is never reached on exception

    return summary


# Tests for Part 3
def test_buggy_expense_processor() -> None:
    """Tests that verify the FIXED version of buggy_expense_processor."""
    result = buggy_expense_processor(BUGGY_EXPENSES)

    check(
        "Part 3: all 5 expenses processed",
        result["processed"] == 5,
        f"Expected processed=5, got {result['processed']}"
    )
    check(
        "Part 3: correct low_value count (< $500 → E01=$120, E03=$45)",
        result["low_value"] == 2,
        f"Expected low_value=2 (E01=$120 and E03=$45 are under $500), got {result['low_value']}"
    )
    check(
        "Part 3: correct high_value count ($500+ → E02, E04, E05)",
        result["high_value"] == 3,
        f"Expected high_value=3, got {result['high_value']}"
    )
    check(
        "Part 3: correct error count (E04=$1200 raises RuntimeError)",
        result["errors"] == 1,
        f"Expected errors=1 (E04 raises RuntimeError), got {result['errors']}"
    )


try:
    test_buggy_expense_processor()
except RuntimeError as exc:
    # The buggy version raises RuntimeError from approve_expense.
    # This check fires when Bug 2 (missing try/except) is not yet fixed.
    check(
        "Part 3: all 5 expenses processed",
        False,
        "Unhandled RuntimeError escaped the loop — Bug 2 not fixed yet: add try/except around approve_expense()"
    )
    check("Part 3: correct low_value count (< $500 → E01=$120, E03=$45)", False, "Fix Bug 2 first")
    check("Part 3: correct high_value count ($500+ → E02, E04, E05)", False, "Fix Bug 2 first")
    check("Part 3: correct error count (E04=$1200 raises RuntimeError)", False, "Fix Bug 2 first")
except Exception:
    check("Part 3 — no unexpected exception", False, traceback.format_exc())


# ---------------------------------------------------------------------------
# Final score summary
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("Run complete. Review PASS/FAIL lines above.")
print()
print("Part 1: 7 pattern identification checks")
print("Part 2: 19 implementation checks (4 functions)")
print("Part 3: 4 debugging checks (3 bugs to find)")
print()
print("If all checks pass, you are ready for Module 05.")
print("=" * 60)
