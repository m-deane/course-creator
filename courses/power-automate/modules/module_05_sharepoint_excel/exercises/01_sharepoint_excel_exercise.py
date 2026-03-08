"""
Module 05 Self-Check Exercises: SharePoint and Excel Integration

These exercises reinforce the concepts from Guide 01 (SharePoint), Guide 02 (Excel),
and the notebook (01_sharepoint_graph_api.ipynb).

Each exercise has:
  - A clear task description
  - Starter code or a function stub
  - Self-checking asserts that print a meaningful message on failure

Run with:
    python 01_sharepoint_excel_exercise.py

All exercises are self-contained — no network calls, no credentials required.
"""

# =============================================================================
# Exercise 1: Build OData Filter Queries
# =============================================================================
#
# OData filter queries are used in:
#   - Power Automate: "Get items" Filter Query field
#   - Excel connector: "List rows" Filter Query field
#   - Graph API: $filter parameter (with fields/ prefix for SharePoint lists)
#
# Rules:
#   - String values use single quotes:   Status eq 'Approved'
#   - Numbers have no quotes:            Amount gt 1000
#   - Dates use ISO 8601 with quotes:    Created ge '2024-01-01T00:00:00Z'
#   - Logical operators: and, or
#   - String functions: startswith(Column, 'prefix'), substringof('text', Column)


def build_odata_filter(conditions: list, operator: str = "and") -> str:
    """
    Build an OData filter string for Power Automate's 'Filter Query' field.

    Each condition is a tuple of (column_name, odata_operator, value).
    String values are automatically single-quoted.
    Numeric values are left unquoted.
    Conditions are joined by the given logical operator.

    Parameters
    ----------
    conditions : list of tuple
        Each tuple: (column_name, operator, value)
        Examples:
            ("Status", "eq", "Pending")       → Status eq 'Pending'
            ("Amount", "gt", 500)             → Amount gt 500
            ("Priority", "le", 2)             → Priority le 2
    operator : str
        "and" or "or" — joins multiple conditions

    Returns
    -------
    str
        OData filter string, e.g. "Status eq 'Pending' and Amount gt 500"

    Examples
    --------
    >>> build_odata_filter([("Status", "eq", "Pending")])
    "Status eq 'Pending'"

    >>> build_odata_filter([("Status", "eq", "Pending"), ("Amount", "gt", 500)])
    "Status eq 'Pending' and Amount gt 500"

    >>> build_odata_filter(
    ...     [("Dept", "eq", "Finance"), ("Dept", "eq", "Legal")],
    ...     operator="or"
    ... )
    "Dept eq 'Finance' or Dept eq 'Legal'"
    """
    # TODO: Implement this function.
    # Hint: iterate over conditions, quote string values, join with operator.
    pass


# --- Self-checks for Exercise 1 ---

def _test_exercise_1():
    print("=" * 60)
    print("Exercise 1: Build OData Filter Queries")
    print("=" * 60)

    # 1a: Single condition, string value
    result = build_odata_filter([("Status", "eq", "Pending")])
    assert result is not None, (
        "FAIL 1a: build_odata_filter returned None — implement the function."
    )
    assert result == "Status eq 'Pending'", (
        f"FAIL 1a: Expected \"Status eq 'Pending'\", got {result!r}\n"
        "  String values must be wrapped in single quotes."
    )
    print("  PASS 1a: Single string condition")

    # 1b: Single condition, numeric value
    result = build_odata_filter([("Amount", "gt", 1000)])
    assert result == "Amount gt 1000", (
        f"FAIL 1b: Expected 'Amount gt 1000', got {result!r}\n"
        "  Numeric values must NOT be wrapped in quotes."
    )
    print("  PASS 1b: Single numeric condition")

    # 1c: Two conditions joined with 'and' (default)
    result = build_odata_filter([("Status", "eq", "Pending"), ("Amount", "gt", 500)])
    assert result == "Status eq 'Pending' and Amount gt 500", (
        f"FAIL 1c: Expected \"Status eq 'Pending' and Amount gt 500\", got {result!r}\n"
        "  Two conditions must be joined with ' and '."
    )
    print("  PASS 1c: Two conditions with 'and'")

    # 1d: Two conditions joined with 'or'
    result = build_odata_filter(
        [("Department", "eq", "Finance"), ("Department", "eq", "Legal")],
        operator="or",
    )
    assert result == "Department eq 'Finance' or Department eq 'Legal'", (
        f"FAIL 1d: Expected \"Department eq 'Finance' or Department eq 'Legal'\", got {result!r}\n"
        "  The operator parameter must support 'or' as well as 'and'."
    )
    print("  PASS 1d: Two conditions with 'or'")

    # 1e: Mixed types — string and integer
    result = build_odata_filter([("Region", "eq", "West"), ("Priority", "le", 2)])
    assert result == "Region eq 'West' and Priority le 2", (
        f"FAIL 1e: Expected \"Region eq 'West' and Priority le 2\", got {result!r}\n"
        "  Mixed types: string 'West' must be quoted, integer 2 must not."
    )
    print("  PASS 1e: Mixed string and integer conditions")

    print("  All Exercise 1 checks passed.\n")


# =============================================================================
# Exercise 2: Match Column Types to Handling Strategies
# =============================================================================
#
# Different SharePoint column types require different treatment when reading
# values from Power Automate dynamic content or Graph API responses.


def get_column_read_strategy(column_type: str) -> str:
    """
    Return the correct reading strategy for a SharePoint column type.

    Parameters
    ----------
    column_type : str
        One of: "choice", "person", "lookup", "managed_metadata", "text", "number"

    Returns
    -------
    str
        A short description of how to read this column type. Must be one of:
        - "plain_string"    : value is a plain text string
        - "sub_property"    : value is an object; access .Email, .DisplayName, etc.
        - "lookup_id"       : value has LookupId (int) and LookupValue (string)
        - "term_guid"       : value has Label (string) and TermGuid (GUID string)
        - "plain_number"    : value is a number
        - "unknown"         : for any unrecognised type

    Column type → strategy reference:
        choice           → plain_string     (e.g., "Approved", "Pending")
        person           → sub_property     (access via /Email or /DisplayName)
        lookup           → lookup_id        (LookupId integer + LookupValue string)
        managed_metadata → term_guid        (Label string + TermGuid GUID)
        text             → plain_string
        number           → plain_number

    Examples
    --------
    >>> get_column_read_strategy("choice")
    "plain_string"

    >>> get_column_read_strategy("person")
    "sub_property"
    """
    # TODO: Implement this function using if/elif or a dict mapping.
    pass


# --- Self-checks for Exercise 2 ---

def _test_exercise_2():
    print("=" * 60)
    print("Exercise 2: Column Type Read Strategies")
    print("=" * 60)

    test_cases = [
        ("choice", "plain_string",
         "Choice columns return a plain string like 'Approved' or 'Pending'."),
        ("text", "plain_string",
         "Text columns return a plain string."),
        ("person", "sub_property",
         "Person columns are objects — access .Email or .DisplayName via /Email or /DisplayName."),
        ("lookup", "lookup_id",
         "Lookup columns have LookupId (int) and LookupValue (string) sub-properties."),
        ("managed_metadata", "term_guid",
         "Managed metadata columns have Label and TermGuid properties."),
        ("number", "plain_number",
         "Number columns return a numeric value directly."),
        ("nonexistent_type", "unknown",
         "Unrecognised column types should return 'unknown'."),
    ]

    for col_type, expected, hint in test_cases:
        result = get_column_read_strategy(col_type)
        assert result is not None, (
            f"FAIL: get_column_read_strategy('{col_type}') returned None — implement the function."
        )
        assert result == expected, (
            f"FAIL: get_column_read_strategy('{col_type}') returned {result!r}, "
            f"expected {expected!r}.\n  Hint: {hint}"
        )
        print(f"  PASS: '{col_type}' → '{expected}'")

    print("  All Exercise 2 checks passed.\n")


# =============================================================================
# Exercise 3: Column Type Write Strategies
# =============================================================================
#
# When writing to SharePoint columns in Power Automate or Graph API,
# different column types need values in different formats.


def format_column_write_value(column_type: str, raw_value) -> dict:
    """
    Format a value for writing to a SharePoint column.

    Different column types require different JSON structures when writing via
    Graph API or Power Automate's 'Create item' / 'Update item' actions.

    Parameters
    ----------
    column_type : str
        One of: "text", "number", "choice", "boolean", "date"
    raw_value : str | int | float | bool
        The raw Python value to write

    Returns
    -------
    dict
        A single-key dict mapping the column type to the correctly formatted value:
        {
            "formatted_value": <value>,
            "notes": <string describing format used>
        }

    Formatting rules:
        text     → pass the string as-is
        number   → pass as float (convert if needed)
        choice   → pass the choice label string as-is
        boolean  → pass the string "TRUE" or "FALSE" (uppercase)
        date     → format as "YYYY-MM-DD" (strip time component if present)

    Examples
    --------
    >>> format_column_write_value("text", "Hello")
    {"formatted_value": "Hello", "notes": "passed as plain string"}

    >>> format_column_write_value("number", "1299.99")
    {"formatted_value": 1299.99, "notes": "cast to float"}

    >>> format_column_write_value("boolean", True)
    {"formatted_value": "TRUE", "notes": "converted to uppercase TRUE/FALSE string"}

    >>> format_column_write_value("date", "2024-03-15T14:30:00Z")
    {"formatted_value": "2024-03-15", "notes": "stripped time component to date only"}
    """
    # TODO: Implement this function.
    # Hint: use isinstance() to check types, and string slicing or split('T')[0]
    # to strip the time component from ISO datetime strings.
    pass


# --- Self-checks for Exercise 3 ---

def _test_exercise_3():
    print("=" * 60)
    print("Exercise 3: Column Write Value Formatting")
    print("=" * 60)

    # 3a: text column — pass through unchanged
    result = format_column_write_value("text", "Contract renewal")
    assert result is not None, (
        "FAIL 3a: format_column_write_value returned None — implement the function."
    )
    assert result.get("formatted_value") == "Contract renewal", (
        f"FAIL 3a: text column — expected 'Contract renewal', got {result.get('formatted_value')!r}\n"
        "  Text columns pass the string value directly."
    )
    print("  PASS 3a: text column")

    # 3b: number column — string "1299.99" must be cast to float 1299.99
    result = format_column_write_value("number", "1299.99")
    assert result.get("formatted_value") == 1299.99, (
        f"FAIL 3b: number column — expected 1299.99 (float), got {result.get('formatted_value')!r}\n"
        "  Number columns require a numeric type, not a string. Use float()."
    )
    print("  PASS 3b: number column string-to-float cast")

    # 3c: number column — integer 5 must become float 5.0
    result = format_column_write_value("number", 5)
    assert result.get("formatted_value") == 5.0, (
        f"FAIL 3c: number column — expected 5.0 (float), got {result.get('formatted_value')!r}\n"
        "  Integer inputs should also be cast to float for consistency."
    )
    print("  PASS 3c: number column int-to-float cast")

    # 3d: choice column — string value passes through
    result = format_column_write_value("choice", "Approved")
    assert result.get("formatted_value") == "Approved", (
        f"FAIL 3d: choice column — expected 'Approved', got {result.get('formatted_value')!r}\n"
        "  Choice columns take the label string directly."
    )
    print("  PASS 3d: choice column")

    # 3e: boolean True → "TRUE"
    result = format_column_write_value("boolean", True)
    assert result.get("formatted_value") == "TRUE", (
        f"FAIL 3e: boolean True — expected 'TRUE' (uppercase string), "
        f"got {result.get('formatted_value')!r}\n"
        "  Excel and SharePoint text columns expect uppercase 'TRUE'/'FALSE' strings."
    )
    print("  PASS 3e: boolean True → 'TRUE'")

    # 3f: boolean False → "FALSE"
    result = format_column_write_value("boolean", False)
    assert result.get("formatted_value") == "FALSE", (
        f"FAIL 3f: boolean False — expected 'FALSE' (uppercase string), "
        f"got {result.get('formatted_value')!r}"
    )
    print("  PASS 3f: boolean False → 'FALSE'")

    # 3g: date with time component — strip time
    result = format_column_write_value("date", "2024-03-15T14:30:00Z")
    assert result.get("formatted_value") == "2024-03-15", (
        f"FAIL 3g: date with time — expected '2024-03-15', "
        f"got {result.get('formatted_value')!r}\n"
        "  Date-only columns need the time component stripped. Use split('T')[0]."
    )
    print("  PASS 3g: date with time → date only")

    # 3h: date without time component — pass through unchanged
    result = format_column_write_value("date", "2024-03-15")
    assert result.get("formatted_value") == "2024-03-15", (
        f"FAIL 3h: date without time — expected '2024-03-15', "
        f"got {result.get('formatted_value')!r}"
    )
    print("  PASS 3h: date without time → unchanged")

    print("  All Exercise 3 checks passed.\n")


# =============================================================================
# Exercise 4: Construct Graph API Requests for SharePoint Operations
# =============================================================================
#
# Build the URL and body for a Graph API request given a high-level description
# of the operation.


GRAPH_BASE = "https://graph.microsoft.com/v1.0"


def build_graph_request(
    operation: str,
    site_id: str,
    list_id: str,
    item_id: str = None,
    fields: dict = None,
    odata_filter: str = None,
) -> dict:
    """
    Construct the HTTP method, URL, and body for a Graph API SharePoint operation.

    Parameters
    ----------
    operation : str
        One of: "get_items", "create_item", "update_item", "delete_item"
    site_id : str
        Graph API site ID
    list_id : str
        List GUID
    item_id : str, optional
        Required for update_item and delete_item
    fields : dict, optional
        Column values — required for create_item and update_item
    odata_filter : str, optional
        OData filter string — used only for get_items

    Returns
    -------
    dict with keys:
        "method"  : HTTP verb string ("GET", "POST", "PATCH", "DELETE")
        "url"     : Full Graph API URL string
        "body"    : Request body dict (None for GET and DELETE)

    URL patterns:
        get_items   : GET  {GRAPH_BASE}/sites/{site_id}/lists/{list_id}/items?$expand=fields[&$filter=...]
        create_item : POST {GRAPH_BASE}/sites/{site_id}/lists/{list_id}/items
                      Body: {"fields": fields}
        update_item : PATCH {GRAPH_BASE}/sites/{site_id}/lists/{list_id}/items/{item_id}/fields
                      Body: fields (not wrapped in {"fields": ...})
        delete_item : DELETE {GRAPH_BASE}/sites/{site_id}/lists/{list_id}/items/{item_id}

    Examples
    --------
    >>> build_graph_request("get_items", "site-123", "list-456")
    {
        "method": "GET",
        "url": "https://graph.microsoft.com/v1.0/sites/site-123/lists/list-456/items?$expand=fields",
        "body": None
    }

    >>> build_graph_request("create_item", "site-123", "list-456", fields={"Title": "Test"})
    {
        "method": "POST",
        "url": "https://graph.microsoft.com/v1.0/sites/site-123/lists/list-456/items",
        "body": {"fields": {"Title": "Test"}}
    }
    """
    # TODO: Implement this function.
    # Hint: use a dict or if/elif on the operation string.
    pass


# --- Self-checks for Exercise 4 ---

def _test_exercise_4():
    print("=" * 60)
    print("Exercise 4: Graph API Request Construction")
    print("=" * 60)

    SITE = "contoso.sharepoint.com,abc,def"
    LIST = "list-guid-1234"
    ITEM = "42"

    # 4a: get_items — no filter
    result = build_graph_request("get_items", SITE, LIST)
    assert result is not None, (
        "FAIL 4a: build_graph_request returned None — implement the function."
    )
    assert result.get("method") == "GET", (
        f"FAIL 4a: get_items must use GET method, got {result.get('method')!r}"
    )
    expected_url_base = f"{GRAPH_BASE}/sites/{SITE}/lists/{LIST}/items"
    assert result.get("url", "").startswith(expected_url_base), (
        f"FAIL 4a: URL should start with {expected_url_base!r}, got {result.get('url')!r}"
    )
    assert "$expand=fields" in result.get("url", ""), (
        "FAIL 4a: get_items URL must include '$expand=fields' to return column values"
    )
    assert result.get("body") is None, (
        f"FAIL 4a: GET request should have no body, got {result.get('body')!r}"
    )
    print("  PASS 4a: get_items without filter")

    # 4b: get_items — with filter
    result = build_graph_request("get_items", SITE, LIST, odata_filter="fields/Status eq 'Pending'")
    assert "fields/Status eq" in result.get("url", "") or "$filter" in result.get("url", ""), (
        "FAIL 4b: get_items with odata_filter must include the filter in the URL"
    )
    print("  PASS 4b: get_items with filter")

    # 4c: create_item
    item_fields = {"Title": "Test Item", "Status": "Pending"}
    result = build_graph_request("create_item", SITE, LIST, fields=item_fields)
    assert result.get("method") == "POST", (
        f"FAIL 4c: create_item must use POST method, got {result.get('method')!r}"
    )
    assert result.get("url") == f"{GRAPH_BASE}/sites/{SITE}/lists/{LIST}/items", (
        f"FAIL 4c: create_item URL is wrong: {result.get('url')!r}"
    )
    assert result.get("body") == {"fields": item_fields}, (
        f"FAIL 4c: create_item body must be {{\"fields\": fields}}, got {result.get('body')!r}"
    )
    print("  PASS 4c: create_item")

    # 4d: update_item
    update_fields = {"Status": "Approved"}
    result = build_graph_request("update_item", SITE, LIST, item_id=ITEM, fields=update_fields)
    assert result.get("method") == "PATCH", (
        f"FAIL 4d: update_item must use PATCH method, got {result.get('method')!r}"
    )
    expected_url = f"{GRAPH_BASE}/sites/{SITE}/lists/{LIST}/items/{ITEM}/fields"
    assert result.get("url") == expected_url, (
        f"FAIL 4d: update_item URL must end with /items/{{item_id}}/fields\n"
        f"  Expected: {expected_url!r}\n  Got: {result.get('url')!r}"
    )
    assert result.get("body") == update_fields, (
        f"FAIL 4d: update_item body must be the fields dict directly (not wrapped in 'fields'), "
        f"got {result.get('body')!r}"
    )
    print("  PASS 4d: update_item")

    # 4e: delete_item
    result = build_graph_request("delete_item", SITE, LIST, item_id=ITEM)
    assert result.get("method") == "DELETE", (
        f"FAIL 4e: delete_item must use DELETE method, got {result.get('method')!r}"
    )
    expected_url = f"{GRAPH_BASE}/sites/{SITE}/lists/{LIST}/items/{ITEM}"
    assert result.get("url") == expected_url, (
        f"FAIL 4e: delete_item URL must end with /items/{{item_id}}\n"
        f"  Expected: {expected_url!r}\n  Got: {result.get('url')!r}"
    )
    assert result.get("body") is None, (
        f"FAIL 4e: DELETE request should have no body, got {result.get('body')!r}"
    )
    print("  PASS 4e: delete_item")

    print("  All Exercise 4 checks passed.\n")


# =============================================================================
# Exercise 5: Excel OData Filter with Date Expression
# =============================================================================
#
# The Excel Online (Business) connector uses the same OData filter syntax as SharePoint.
# Date filtering requires ISO 8601 format. When filtering relative to today,
# the date must be computed and embedded in the filter string.


def build_excel_date_filter(column_name: str, days_back: int) -> str:
    """
    Build an OData filter string for an Excel date column covering the past N days.

    The filter should return rows where the column value is greater than or equal to
    a date that is `days_back` days before today (today's date, not datetime).

    The date must be formatted as 'YYYY-MM-DD' (no time component) and wrapped in
    single quotes in the OData string.

    Parameters
    ----------
    column_name : str
        The Excel table column name to filter on
    days_back : int
        Number of days to look back from today

    Returns
    -------
    str
        OData filter string, e.g. "OrderDate ge '2024-03-01'"

    Notes
    -----
    This function must compute a real date (not a hardcoded string) so the filter
    is always relative to the current date when the function is called.

    Examples
    --------
    >>> # Assuming today is 2024-03-08
    >>> build_excel_date_filter("OrderDate", 7)
    "OrderDate ge '2024-03-01'"
    """
    # TODO: Implement this function.
    # Hint: use datetime.date.today() and timedelta(days=days_back)
    # then format with strftime('%Y-%m-%d')
    from datetime import date, timedelta
    pass


# --- Self-checks for Exercise 5 ---

def _test_exercise_5():
    print("=" * 60)
    print("Exercise 5: Excel Date Filter Construction")
    print("=" * 60)

    from datetime import date, timedelta

    result = build_excel_date_filter("OrderDate", 7)
    assert result is not None, (
        "FAIL 5a: build_excel_date_filter returned None — implement the function."
    )
    assert isinstance(result, str), (
        f"FAIL 5a: Expected a string, got {type(result)}"
    )

    # Verify the column name is in the filter
    assert "OrderDate" in result, (
        f"FAIL 5a: Filter must include the column name 'OrderDate', got {result!r}"
    )

    # Verify the ge operator is used
    assert " ge " in result, (
        f"FAIL 5a: Filter must use the 'ge' operator for 'greater than or equal to', got {result!r}\n"
        "  Correct format: OrderDate ge '2024-03-01'"
    )

    # Verify the date is a real date (not hardcoded)
    expected_date = (date.today() - timedelta(days=7)).strftime("%Y-%m-%d")
    assert f"'{expected_date}'" in result, (
        f"FAIL 5a: Filter must contain the date '{expected_date}' (7 days back from today).\n"
        f"  Got: {result!r}\n"
        "  The date must be computed dynamically, not hardcoded."
    )
    print(f"  PASS 5a: 7 days back filter is correct: {result!r}")

    # 5b: Test with 30 days
    result_30 = build_excel_date_filter("Created", 30)
    expected_date_30 = (date.today() - timedelta(days=30)).strftime("%Y-%m-%d")
    assert f"'{expected_date_30}'" in result_30, (
        f"FAIL 5b: 30-day filter — expected date {expected_date_30!r} in filter, got {result_30!r}"
    )
    assert "Created" in result_30, (
        f"FAIL 5b: Column name 'Created' must appear in the filter, got {result_30!r}"
    )
    print(f"  PASS 5b: 30 days back filter is correct: {result_30!r}")

    print("  All Exercise 5 checks passed.\n")


# =============================================================================
# Run all exercises
# =============================================================================

if __name__ == "__main__":
    print("\nModule 05: SharePoint and Excel Integration — Self-Check Exercises\n")

    results = []

    for test_fn, label in [
        (_test_exercise_1, "Exercise 1: Build OData Filter Queries"),
        (_test_exercise_2, "Exercise 2: Column Type Read Strategies"),
        (_test_exercise_3, "Exercise 3: Column Write Value Formatting"),
        (_test_exercise_4, "Exercise 4: Graph API Request Construction"),
        (_test_exercise_5, "Exercise 5: Excel Date Filter Construction"),
    ]:
        try:
            test_fn()
            results.append((label, True, None))
        except AssertionError as err:
            results.append((label, False, str(err)))
        except Exception as err:
            results.append((label, False, f"Unexpected error: {err}"))

    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    for label, ok, msg in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {label}")
        if msg:
            # Indent the error message for readability
            for line in msg.strip().splitlines():
                print(f"         {line}")

    print(f"\n{passed}/{len(results)} exercises passed.")
    if passed < len(results):
        print("Review the FAIL messages above and update your implementations.")
    else:
        print("All exercises complete — well done.")
