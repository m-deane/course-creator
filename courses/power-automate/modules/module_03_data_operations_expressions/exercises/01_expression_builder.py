"""
Module 03 — Exercise: Expression Builder
=========================================

Self-check exercise covering Power Automate expression syntax across all
five function categories: String, Date/Time, Logical, Type Conversion, and
Collection.

How to use
----------
1. Read each scenario below.
2. Write the Power Automate expression as a Python string in the `answer_*`
   variable.
3. Run this file: `python 01_expression_builder.py`
4. The validator will check each expression against the expected pattern and
   provide specific feedback on errors.

Rules
-----
- Write expressions exactly as you would type them in the Power Automate
  Expression editor — no leading @, no surrounding @{...}
- Use single quotes for string literals: 'hello' not "hello"
- Function names are case-sensitive in the validator (they match the real PA syntax)
- Dynamic content references (e.g. triggerBody()?['Subject']) are accepted
  as-is — the validator does not execute them, it checks syntax only

Scoring
-------
Each scenario is worth 1 point. A perfect score is 12/12.
"""

import re
import sys

# ---------------------------------------------------------------------------
# Expression validator (lightweight — checks syntax, not semantics)
# ---------------------------------------------------------------------------

KNOWN_FUNCTIONS: frozenset[str] = frozenset([
    'concat', 'substring', 'replace', 'split', 'toLower', 'toUpper', 'trim',
    'startsWith', 'endsWith', 'indexOf',
    'utcNow', 'formatDateTime', 'addDays', 'addHours', 'addMinutes',
    'addSeconds', 'addMonths', 'convertTimeZone',
    'if', 'equals', 'greater', 'greaterOrEquals', 'less', 'lessOrEquals',
    'and', 'or', 'not', 'coalesce',
    'int', 'float', 'string', 'bool', 'json', 'xml',
    'length', 'first', 'last', 'contains', 'empty', 'union',
    'skip', 'take', 'min', 'max', 'add', 'sub', 'mul', 'div', 'mod',
    'outputs', 'triggerBody', 'triggerOutputs', 'variables', 'items',
    'body', 'actions', 'item', 'workflow', 'guid',
])


def _check_balanced_parens(expr: str) -> list[str]:
    errors = []
    depth = 0
    in_quote = False
    for i, ch in enumerate(expr):
        if ch == "'":
            in_quote = not in_quote
        elif not in_quote:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth < 0:
                    errors.append(f'Unexpected ) at position {i}')
                    depth = 0
    if depth > 0:
        errors.append(f'{depth} unclosed parenthesis(es) — add {depth} closing )')
    return errors


def _check_string_quotes(expr: str) -> list[str]:
    errors = []
    # Remove content inside single-quoted regions to avoid false positives
    outside = re.sub(r"'[^']*'", "''", expr)
    if '"' in outside:
        errors.append("Found double quotes. Use single quotes for string literals: 'hello' not \"hello\"")
    return errors


def _check_at_prefix(expr: str) -> list[str]:
    if expr.strip().startswith('@'):
        return ['Remove the leading @. In the Expression editor, type the function name directly.']
    return []


def _get_outer_function(expr: str) -> str | None:
    m = re.match(r'^([a-zA-Z][a-zA-Z0-9]*)\(', expr.strip())
    return m.group(1) if m else None


def validate(expression: str, required_functions: list[str] | None = None) -> tuple[bool, list[str]]:
    """Validate a Power Automate expression string.

    Parameters
    ----------
    expression : str
        The expression to validate.
    required_functions : list[str], optional
        Function names that must appear somewhere in the expression.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list_of_error_messages)
    """
    errors: list[str] = []

    errors += _check_at_prefix(expression)
    errors += _check_string_quotes(expression)
    errors += _check_balanced_parens(expression)

    outer = _get_outer_function(expression)
    if outer and outer not in KNOWN_FUNCTIONS:
        lower_matches = [f for f in KNOWN_FUNCTIONS if f.lower() == outer.lower()]
        hint = f" Did you mean '{lower_matches[0]}'?" if lower_matches else ''
        errors.append(f"Unknown function '{outer}'.{hint} Check camelCase spelling.")

    if required_functions:
        for func in required_functions:
            # Check that the function name appears followed by ( in the expression
            pattern = rf'\b{re.escape(func)}\s*\('
            if not re.search(pattern, expression):
                errors.append(f"Expression must use the '{func}' function.")

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Helper: run one scenario and print result
# ---------------------------------------------------------------------------

_score = 0
_total = 0


def check(
    scenario_number: int,
    description: str,
    answer: str | None,
    required_functions: list[str],
    example_correct: str,
) -> None:
    global _score, _total
    _total += 1

    print(f'Scenario {scenario_number}: {description}')

    if answer is None:
        print('  [SKIPPED] Set the answer variable to your expression string.')
        print()
        return

    is_valid, errors = validate(answer, required_functions=required_functions)

    if is_valid:
        _score += 1
        print(f'  [PASS] {answer!r}')
    else:
        print(f'  [FAIL] {answer!r}')
        for err in errors:
            print(f'    - {err}')
        print(f'  Example of a correct expression: {example_correct}')
    print()


# ---------------------------------------------------------------------------
# SCENARIO 1 — String: Build a file name
# ---------------------------------------------------------------------------
# Task: Produce the string 'invoice_2024_nov.pdf'
# Given string pieces: 'invoice', '2024', 'nov', 'pdf'
# Constraint: must use concat() with at least 3 arguments and string separators
#
# Expected output: 'invoice_2024_nov.pdf'
# Required functions: concat

answer_01 = None  # <- write your expression here

# ---------------------------------------------------------------------------
# SCENARIO 2 — String: Extract year from a reference code
# ---------------------------------------------------------------------------
# Task: From 'REF-2024-001', extract '2024'
# The year always starts at index 4 and is 4 characters long.
# Required functions: substring

answer_02 = None  # <- write your expression here

# ---------------------------------------------------------------------------
# SCENARIO 3 — String: Normalise a department name
# ---------------------------------------------------------------------------
# Task: Take an input like '  FINANCE & OPERATIONS  ' and produce
#       'finance & operations' (lowercased, whitespace stripped)
# Required functions: trim, toLower

answer_03 = None  # <- write your expression here

# ---------------------------------------------------------------------------
# SCENARIO 4 — String: Replace underscores with spaces in a column name
# ---------------------------------------------------------------------------
# Task: Convert 'purchase_order_number' to 'purchase order number'
# Required functions: replace

answer_04 = None  # <- write your expression here

# ---------------------------------------------------------------------------
# SCENARIO 5 — Date/Time: Format today as ISO date
# ---------------------------------------------------------------------------
# Task: Return the current UTC date formatted as 'yyyy-MM-dd'
#       (for example, '2024-11-15')
# Required functions: formatDateTime, utcNow

answer_05 = None  # <- write your expression here

# ---------------------------------------------------------------------------
# SCENARIO 6 — Date/Time: Subject line with full date
# ---------------------------------------------------------------------------
# Task: Produce a report subject line: 'Weekly Report - November 15, 2024'
#       where the date portion comes from the current UTC time.
# Required functions: concat, formatDateTime, utcNow

answer_06 = None  # <- write your expression here

# ---------------------------------------------------------------------------
# SCENARIO 7 — Date/Time: Compute a due date 30 days from now
# ---------------------------------------------------------------------------
# Task: Return the date 30 days from now, formatted as 'MMMM d, yyyy'
# Required functions: formatDateTime, addDays, utcNow

answer_07 = None  # <- write your expression here

# ---------------------------------------------------------------------------
# SCENARIO 8 — Logical: Approval routing
# ---------------------------------------------------------------------------
# Task: If the variable 'Amount' is greater than 10000, return
#       'VP approval required', otherwise return 'Auto-approved'
# Required functions: if, greater

answer_08 = None  # <- write your expression here
# Hint: variables('Amount') is the dynamic content reference for a variable

# ---------------------------------------------------------------------------
# SCENARIO 9 — Logical: Safe email fallback
# ---------------------------------------------------------------------------
# Task: Return the manager's email if it is not null/empty,
#       otherwise return 'helpdesk@contoso.com'
# Required functions: coalesce

answer_09 = None  # <- write your expression here
# Hint: coalesce(triggerBody()?['ManagerEmail'], 'helpdesk@contoso.com')

# ---------------------------------------------------------------------------
# SCENARIO 10 — Type conversion: Compute invoice total
# ---------------------------------------------------------------------------
# Task: Multiply the string fields 'Quantity' and 'UnitPrice' together.
#       'Quantity' is an integer-valued string; 'UnitPrice' is a float-valued string.
#       Return the result as a string (use string() to convert the product back).
# Required functions: string, mul, int, float

answer_10 = None  # <- write your expression here
# Hint: string(mul(int(triggerBody()?['Quantity']), float(triggerBody()?['UnitPrice'])))

# ---------------------------------------------------------------------------
# SCENARIO 11 — Collection: Check for attachments before processing
# ---------------------------------------------------------------------------
# Task: Return true if the Attachments field is NOT empty, false otherwise.
#       (This would be used as a condition in the flow designer.)
# Required functions: not, empty

answer_11 = None  # <- write your expression here
# Hint: not(empty(triggerBody()?['Attachments']))

# ---------------------------------------------------------------------------
# SCENARIO 12 — Collection + String: Count items and pluralise
# ---------------------------------------------------------------------------
# Task: Given an array variable 'ApproverList', produce the string:
#       '3 approvers assigned' (where 3 is the length of the array)
# Required functions: concat, string, length

answer_12 = None  # <- write your expression here
# Hint: concat(string(length(variables('ApproverList'))), ' approvers assigned')

# ---------------------------------------------------------------------------
# Run all checks
# ---------------------------------------------------------------------------

print('=' * 60)
print('Module 03 — Expression Builder Self-Check')
print('=' * 60)
print()

check(
    1,
    "Build file name: 'invoice_2024_nov.pdf'",
    answer_01,
    required_functions=['concat'],
    example_correct="concat('invoice', '_', '2024', '_', 'nov', '.pdf')",
)

check(
    2,
    "Extract '2024' from 'REF-2024-001'",
    answer_02,
    required_functions=['substring'],
    example_correct="substring('REF-2024-001', 4, 4)",
)

check(
    3,
    "Normalise '  FINANCE & OPERATIONS  ' → 'finance & operations'",
    answer_03,
    required_functions=['trim', 'toLower'],
    example_correct="trim(toLower('  FINANCE & OPERATIONS  '))",
)

check(
    4,
    "Replace underscores: 'purchase_order_number' → 'purchase order number'",
    answer_04,
    required_functions=['replace'],
    example_correct="replace('purchase_order_number', '_', ' ')",
)

check(
    5,
    "Current UTC date as 'yyyy-MM-dd'",
    answer_05,
    required_functions=['formatDateTime', 'utcNow'],
    example_correct="formatDateTime(utcNow(), 'yyyy-MM-dd')",
)

check(
    6,
    "Subject line: 'Weekly Report - November 15, 2024'",
    answer_06,
    required_functions=['concat', 'formatDateTime', 'utcNow'],
    example_correct="concat('Weekly Report - ', formatDateTime(utcNow(), 'MMMM d, yyyy'))",
)

check(
    7,
    "Due date: 30 days from now formatted as 'MMMM d, yyyy'",
    answer_07,
    required_functions=['formatDateTime', 'addDays', 'utcNow'],
    example_correct="formatDateTime(addDays(utcNow(), 30), 'MMMM d, yyyy')",
)

check(
    8,
    "Approval routing: if Amount > 10000 → 'VP approval required'",
    answer_08,
    required_functions=['if', 'greater'],
    example_correct="if(greater(variables('Amount'), 10000), 'VP approval required', 'Auto-approved')",
)

check(
    9,
    "Email fallback: manager email or 'helpdesk@contoso.com'",
    answer_09,
    required_functions=['coalesce'],
    example_correct="coalesce(triggerBody()?['ManagerEmail'], 'helpdesk@contoso.com')",
)

check(
    10,
    "Invoice total: string(Quantity × UnitPrice) from string inputs",
    answer_10,
    required_functions=['string', 'mul', 'int', 'float'],
    example_correct="string(mul(int(triggerBody()?['Quantity']), float(triggerBody()?['UnitPrice'])))",
)

check(
    11,
    "Has attachments: not(empty(Attachments))",
    answer_11,
    required_functions=['not', 'empty'],
    example_correct="not(empty(triggerBody()?['Attachments']))",
)

check(
    12,
    "Count and label: '3 approvers assigned'",
    answer_12,
    required_functions=['concat', 'string', 'length'],
    example_correct="concat(string(length(variables('ApproverList'))), ' approvers assigned')",
)

# ---------------------------------------------------------------------------
# Final score
# ---------------------------------------------------------------------------

print('=' * 60)
print(f'Score: {_score}/{_total}')

if _score == _total:
    print('All expressions correct. Module 03 expression builder complete.')
elif _score >= _total * 0.75:
    print('Good progress. Review the FAIL messages above and revise the remaining expressions.')
elif _score >= _total * 0.5:
    print('Half complete. Focus on the categories where you have multiple failures.')
else:
    print('Keep going. Re-read Guide 01 alongside each failing scenario.')

print()
print('Tip: Every FAIL message shows an example of a correct expression.')
print('     Use it to understand the expected pattern, then write your own version.')
