# Tool Design: Principles for Effective Tool Schemas

> **Reading time:** ~10 min | **Module:** 2 — Tool Use & Function Calling | **Prerequisites:** Module 2 — Tool Fundamentals

Well-designed tools are the difference between agents that work reliably and agents that confuse, fail, or misbehave. This guide covers principles for creating tools that LLMs can understand and use correctly.

<div class="callout-insight">

**Insight:** Your tool descriptions are prompts. The LLM reads your tool name, description, and parameter definitions to decide when and how to use each tool. Invest the same care in tool design as you would in system prompts.

</div>

---

## The SOLID Principles of Tool Design

### 1. Single Responsibility

Each tool should do one thing well:


<span class="filename">agent.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# Bad: Tool does too many things
{
    "name": "manage_user",
    "description": "Create, update, delete, or fetch users",
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {"enum": ["create", "update", "delete", "fetch"]},
            "user_id": {"type": "string"},
            "user_data": {"type": "object"}
        }
    }
}

# Good: Separate tools for each action
{
    "name": "create_user",
    "description": "Create a new user account",
    "input_schema": {
        "type": "object",
        "properties": {
            "email": {"type": "string"},
            "name": {"type": "string"}
        },
        "required": ["email", "name"]
    }
}

{
    "name": "get_user",
    "description": "Fetch user details by ID",
    "input_schema": {
        "type": "object",
        "properties": {
            "user_id": {"type": "string"}
        },
        "required": ["user_id"]
    }
}
```

</div>
</div>

### 2. Clear Naming

Use verbs that describe the action:

```python
# Bad: Vague or noun-based names
"user"           # What does this do?
"data"           # Too generic
"process"        # Process what?

# Good: Action-oriented names
"get_user"       # Clearly fetches user
"create_order"   # Clearly creates order
"search_products"  # Clearly searches
"calculate_tax"  # Clearly calculates
"send_email"     # Clearly sends
```

Naming conventions:
- `get_*` - Retrieve data
- `create_*` - Make something new
- `update_*` - Modify existing data
- `delete_*` - Remove data
- `search_*` - Find matching items
- `calculate_*` - Compute a value
- `send_*` - Transmit something

### 3. Descriptive Descriptions

Write descriptions that explain when to use AND when not to use:

```python
# Bad: Too brief
{
    "name": "search_web",
    "description": "Search the web"
}

# Good: Comprehensive guidance
{
    "name": "search_web",
    "description": """Search the web for current information.

Use this tool when:
- The user asks about recent events (after your knowledge cutoff)
- You need to verify current facts (prices, availability, etc.)
- The user explicitly asks you to search online

Do NOT use this tool when:
- The question is about well-established facts you know
- The user is asking for your opinion or analysis
- The information is already in the conversation context

Returns: Top 5 search results with titles, snippets, and URLs."""
}
```

### 4. Explicit Parameters

Every parameter should be self-documenting:

```python
{
    "name": "book_flight",
    "description": "Search and book flights",
    "input_schema": {
        "type": "object",
        "properties": {
            "origin": {
                "type": "string",
                "description": "Departure airport code (e.g., 'JFK', 'LAX'). Use IATA codes."
            },
            "destination": {
                "type": "string",
                "description": "Arrival airport code (e.g., 'LHR', 'NRT'). Use IATA codes."
            },
            "departure_date": {
                "type": "string",
                "description": "Departure date in YYYY-MM-DD format (e.g., '2024-03-15')"
            },
            "passengers": {
                "type": "integer",
                "description": "Number of passengers (1-9)",
                "minimum": 1,
                "maximum": 9,
                "default": 1
            },
            "cabin_class": {
                "type": "string",
                "description": "Cabin class preference",
                "enum": ["economy", "premium_economy", "business", "first"],
                "default": "economy"
            }
        },
        "required": ["origin", "destination", "departure_date"]
    }
}
```

### 5. Safe Defaults

Non-required parameters should have sensible defaults:

```python
{
    "name": "search_database",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results to return",
                "default": 10,  # Sensible default
                "minimum": 1,
                "maximum": 100
            },
            "sort_by": {
                "type": "string",
                "description": "Field to sort by",
                "default": "relevance",  # Safe default
                "enum": ["relevance", "date", "popularity"]
            },
            "include_deleted": {
                "type": "boolean",
                "description": "Include soft-deleted records",
                "default": False  # Safe: don't show deleted by default
            }
        },
        "required": ["query"]
    }
}
```

---

## Schema Design Patterns

### Pattern 1: Constrained Inputs

Use enums and ranges to limit possible values:


<span class="filename">agent.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
{
    "name": "set_thermostat",
    "input_schema": {
        "type": "object",
        "properties": {
            "temperature": {
                "type": "number",
                "description": "Target temperature in Fahrenheit",
                "minimum": 55,   # Prevent freezing
                "maximum": 85    # Prevent overheating
            },
            "mode": {
                "type": "string",
                "enum": ["heat", "cool", "auto", "off"],
                "description": "HVAC mode"
            },
            "room": {
                "type": "string",
                "enum": ["living_room", "bedroom", "kitchen", "office"],
                "description": "Which room to adjust"
            }
        },
        "required": ["temperature", "mode", "room"]
    }
}
```

</div>
</div>

### Pattern 2: Nested Objects

For complex, related parameters:

```python
{
    "name": "create_event",
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "description": {"type": "string"},
            "time": {
                "type": "object",
                "description": "Event timing details",
                "properties": {
                    "start": {
                        "type": "string",
                        "description": "Start time in ISO 8601 format"
                    },
                    "end": {
                        "type": "string",
                        "description": "End time in ISO 8601 format"
                    },
                    "timezone": {
                        "type": "string",
                        "description": "Timezone (e.g., 'America/New_York')"
                    }
                },
                "required": ["start"]
            },
            "location": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "address": {"type": "string"},
                    "virtual_url": {"type": "string"}
                }
            }
        },
        "required": ["title", "time"]
    }
}
```

### Pattern 3: Arrays for Multiple Items

When operations can apply to multiple items:

```python
{
    "name": "send_notifications",
    "input_schema": {
        "type": "object",
        "properties": {
            "recipients": {
                "type": "array",
                "description": "List of user IDs to notify",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 100
            },
            "message": {
                "type": "string",
                "description": "Notification message",
                "maxLength": 500
            },
            "priority": {
                "type": "string",
                "enum": ["low", "normal", "high", "urgent"],
                "default": "normal"
            }
        },
        "required": ["recipients", "message"]
    }
}
```

### Pattern 4: Conditional Parameters

Use `anyOf` or `oneOf` for mutually exclusive options:

```python
{
    "name": "search_content",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "source": {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            "type": {"const": "database"},
                            "table": {"type": "string"}
                        },
                        "required": ["type", "table"]
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {"const": "file_system"},
                            "path": {"type": "string"}
                        },
                        "required": ["type", "path"]
                    }
                ]
            }
        },
        "required": ["query", "source"]
    }
}
```

---

## Tool Set Organization

### Group Related Tools


<span class="filename">agent.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# Database tools
database_tools = [
    {"name": "db_query", ...},
    {"name": "db_insert", ...},
    {"name": "db_update", ...},
    {"name": "db_delete", ...},
]

# Communication tools
communication_tools = [
    {"name": "send_email", ...},
    {"name": "send_sms", ...},
    {"name": "send_slack", ...},
]

# All tools for the agent
all_tools = database_tools + communication_tools
```

</div>
</div>

### Progressive Tool Disclosure

Start with fewer tools, add as needed:

```python
def get_tools_for_task(task_type: str) -> list:
    """Return appropriate tools for the task type."""
    base_tools = [search_tool, get_info_tool]

    if task_type == "data_analysis":
        return base_tools + [query_db_tool, calculate_tool, visualize_tool]
    elif task_type == "customer_support":
        return base_tools + [get_customer_tool, create_ticket_tool, refund_tool]
    elif task_type == "content_creation":
        return base_tools + [generate_image_tool, spell_check_tool]

    return base_tools
```

### Tool Limit Considerations

More tools = more tokens + more confusion:

| Tool Count | Impact |
|------------|--------|
| 1-5 | Clear selection, low overhead |
| 6-15 | Manageable, may need clear descriptions |
| 16-30 | Higher latency, needs excellent descriptions |
| 30+ | Consider sub-agents or dynamic tool loading |

---

## Description Best Practices

### Template for Tool Descriptions


<span class="filename">agent.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def create_tool_description(
    purpose: str,
    when_to_use: list[str],
    when_not_to_use: list[str],
    returns: str,
    notes: str = None
) -> str:
    """Generate a comprehensive tool description."""

    desc = f"{purpose}\n\n"
    desc += "Use this tool when:\n"
    desc += "\n".join(f"- {u}" for u in when_to_use)
    desc += "\n\nDo NOT use this tool when:\n"
    desc += "\n".join(f"- {n}" for n in when_not_to_use)
    desc += f"\n\nReturns: {returns}"

    if notes:
        desc += f"\n\nNotes: {notes}"

    return desc


# Example usage
search_tool = {
    "name": "search_knowledge_base",
    "description": create_tool_description(
        purpose="Search the internal knowledge base for company policies and procedures.",
        when_to_use=[
            "User asks about company policies",
            "You need to verify internal procedures",
            "Question relates to HR, IT, or operations guidelines"
        ],
        when_not_to_use=[
            "User asks about external/public information",
            "Question is about general knowledge",
            "Information is already in the conversation"
        ],
        returns="Relevant document excerpts with titles and last-updated dates",
        notes="Results are ranked by relevance. Check document dates for currency."
    ),
    "input_schema": {...}
}
```

</div>
</div>

---

## Validation and Error Prevention

### Input Validation Patterns

Build validation into your tool execution:


<span class="filename">agent.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def execute_book_flight(origin: str, destination: str, departure_date: str, **kwargs):
    """Execute flight booking with validation."""

    errors = []

    # Validate airport codes
    valid_airports = load_airport_codes()
    if origin.upper() not in valid_airports:
        errors.append(f"Invalid origin airport: {origin}")
    if destination.upper() not in valid_airports:
        errors.append(f"Invalid destination airport: {destination}")

    # Validate date
    try:
        date = datetime.strptime(departure_date, "%Y-%m-%d")
        if date < datetime.now():
            errors.append("Departure date cannot be in the past")
        if date > datetime.now() + timedelta(days=365):
            errors.append("Cannot book more than 1 year in advance")
    except ValueError:
        errors.append(f"Invalid date format: {departure_date}. Use YYYY-MM-DD")

    if errors:
        return {
            "status": "error",
            "errors": errors,
            "suggestion": "Please correct the above issues and try again"
        }

    # Proceed with booking...
```

</div>
</div>

### Helpful Error Messages

```python
def execute_tool(name: str, arguments: dict) -> dict:
    """Execute tool with helpful error messages."""

    try:
        result = tool_handlers[name](**arguments)
        return {"status": "success", "data": result}

    except KeyError as e:
        return {
            "status": "error",
            "error": f"Missing required field: {e}",
            "available_fields": list(arguments.keys())
        }

    except ValueError as e:
        return {
            "status": "error",
            "error": f"Invalid value: {e}",
            "suggestion": "Check the parameter constraints in the tool description"
        }

    except PermissionError:
        return {
            "status": "error",
            "error": "Permission denied",
            "suggestion": "This action may require elevated privileges"
        }

    except Exception as e:
        return {
            "status": "error",
            "error": f"Unexpected error: {type(e).__name__}",
            "details": str(e)
        }
```

---

## Testing Tools

### Tool Schema Validation


<span class="filename">agent.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import jsonschema

def validate_tool_schema(tool: dict) -> list[str]:
    """Validate a tool definition."""
    issues = []

    # Required fields
    if "name" not in tool:
        issues.append("Missing 'name' field")
    if "description" not in tool:
        issues.append("Missing 'description' field")
    if "input_schema" not in tool:
        issues.append("Missing 'input_schema' field")

    # Name format
    if "name" in tool:
        name = tool["name"]
        if not name.islower() or " " in name:
            issues.append(f"Name should be lowercase with underscores: {name}")
        if len(name) > 64:
            issues.append(f"Name too long (max 64 chars): {name}")

    # Description quality
    if "description" in tool:
        desc = tool["description"]
        if len(desc) < 20:
            issues.append("Description too short - add more detail")
        if "when" not in desc.lower():
            issues.append("Description should explain when to use the tool")

    return issues
```

</div>
</div>

### Tool Behavior Testing

```python
def test_tool_behavior(tool_name: str, test_cases: list[dict]) -> dict:
    """Test a tool's behavior with various inputs."""

    results = []

    for case in test_cases:
        try:
            result = execute_tool(tool_name, case["input"])

            passed = True
            if "expected_status" in case:
                passed = passed and result.get("status") == case["expected_status"]
            if "expected_contains" in case:
                result_str = json.dumps(result)
                passed = passed and case["expected_contains"] in result_str

            results.append({
                "input": case["input"],
                "passed": passed,
                "result": result
            })

        except Exception as e:
            results.append({
                "input": case["input"],
                "passed": False,
                "error": str(e)
            })

    return {
        "total": len(test_cases),
        "passed": sum(r["passed"] for r in results),
        "details": results
    }
```

<div class="callout-key">

**Key Concept Summary:** This guide covered the core concepts. Review the companion slides for visual summaries and the hands-on notebook for practice implementations.

</div>

---

*Tool design is UX design for AI. Your tools are the interface between natural language intent and programmatic action. Design them thoughtfully.*


## Practice Questions

1. Explain in your own words how the concepts in this guide relate to building production agents.
2. What are the key tradeoffs you need to consider when applying these techniques?
3. Describe a scenario where the approach from this guide would be the wrong choice, and what you would use instead.

---

**Next Steps:**

<a class="link-card" href="./02_tool_design_slides.md">
  <div class="link-card-title">Tool Design Patterns — Companion Slides</div>
  <div class="link-card-description">Visual slide deck with diagrams, speaker notes, and key takeaways.</div>
</a>

<a class="link-card" href="../notebooks/01_basic_tools.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with working code and guided exercises.</div>
</a>
