# Quiz: Module 2 - Tool Use & Function Calling

**Estimated Time:** 20 minutes
**Total Points:** 100
**Passing Score:** 70%

---

## Instructions

This quiz evaluates your understanding of tool fundamentals, tool design principles, and error handling in agent systems. Provide complete, thoughtful answers that demonstrate practical knowledge.

---

## Part A: Tool Fundamentals (30 points)

### Question 1 (8 points)
Describe the complete tool calling flow from user query to final response. Include all major steps.

**Your Answer:**

---

### Question 2 (7 points)
What are the four essential components of a tool definition? Provide a brief explanation of each.

**Your Answer:**

---

### Question 3 (8 points)
Consider this tool definition:

```python
{
    "name": "do_stuff",
    "description": "Does stuff with data",
    "parameters": {
        "type": "object",
        "properties": {
            "data": {"type": "string"}
        }
    }
}
```

Identify at least 3 problems with this tool definition and explain why each is problematic.

**Your Answer:**

---

### Question 4 (7 points)
**True or False:** From the LLM's perspective, tool calling is fundamentally just generating structured JSON that matches a provided schema.

**Your Answer:**

---

## Part B: Tool Design (35 points)

### Question 5 (10 points)
List the five core principles of effective tool design from the module and provide a brief example for any two of them.

**Your Answer:**

---

### Question 6 (8 points)
You need to create a tool for sending emails. Which design is better and why?

**Option A:**
```python
{
    "name": "send_email",
    "description": "Send an email to someone",
    "parameters": {
        "properties": {
            "recipient": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
            "cc": {"type": "array"},
            "bcc": {"type": "array"},
            "attachments": {"type": "array"},
            "priority": {"type": "string"},
            "read_receipt": {"type": "boolean"}
        }
    }
}
```

**Option B:**
```python
{
    "name": "send_email",
    "description": "Send an email. Use for simple email communication only. For bulk emails or automated campaigns, use send_campaign instead.",
    "parameters": {
        "properties": {
            "recipient": {
                "type": "string",
                "description": "Email address of recipient"
            },
            "subject": {
                "type": "string",
                "description": "Email subject line (max 200 chars)"
            },
            "body": {
                "type": "string",
                "description": "Email body content in plain text"
            }
        },
        "required": ["recipient", "subject", "body"]
    }
}
```

**Your Answer:**

---

### Question 7 (9 points)
Design a tool schema for a function that searches a company's internal knowledge base. Include name, description, and at least 3 well-defined parameters.

**Your Answer:**

---

### Question 8 (8 points)
What does "bounded output" mean in tool design, and why is it important for agent reliability?

**Your Answer:**

---

## Part C: Error Handling (35 points)

### Question 9 (10 points)
List four different types of errors that can occur during tool execution and provide a specific example of each.

**Your Answer:**

---

### Question 10 (8 points)
Which error handling approach is most appropriate for a tool that calls an external API that occasionally times out?

A) Return an empty result and continue
B) Implement retry logic with exponential backoff (3 attempts), then return a clear error message if all fail
C) Raise an exception and crash the agent
D) Return cached data from a previous successful call

**Your Answer:**

---

### Question 11 (9 points)
Write example error messages for the following scenarios. Each message should be actionable and inform the agent what to do next:

a) User tries to access a file that doesn't exist
b) Tool requires API authentication but key is missing
c) Search query returned too many results (over limit)

**Your Answer:**

---

### Question 12 (8 points)
What is the difference between "fail fast" and "graceful degradation" error handling strategies? Give a scenario where each is appropriate.

**Your Answer:**

---

## Part D: Security & Best Practices (Bonus - 10 points)

### Question 13 (5 points)
Why is it dangerous to allow an agent to execute arbitrary code or shell commands without restrictions? Name at least two specific risks.

**Your Answer:**

---

### Question 14 (5 points)
You're building a file management tool for an agent. What is one critical security control you should implement?

**Your Answer:**

---

# Answer Key

## Part A: Tool Fundamentals

**Question 1:** (8 points)
*Sample Answer:*
1. User submits query to agent
2. LLM analyzes query and determines if a tool is needed
3. LLM generates tool call with function name and arguments in JSON format
4. Your code receives the tool call and executes the actual function
5. Tool returns result (success data or error)
6. Result is passed back to LLM as context
7. LLM generates final response incorporating tool results
8. Response sent to user

*Grading:* 1 point per step, minimum 6 steps for full credit

**Question 2:** (7 points)
*Answer:*
1. **Name** - Identifies the tool (e.g., "search_web")
2. **Description** - Explains when and how to use it
3. **Parameters** - Defines inputs needed with types and validation
4. **Required** - Specifies which parameters are mandatory

*Grading:* 1.5 points per component with explanation

**Question 3:** (8 points)
*Problems:*
1. **Vague name** - "do_stuff" doesn't describe the action; should use clear verb (e.g., "process_data", "transform_data")
2. **Unhelpful description** - "Does stuff" provides no guidance on when to use this tool or what it accomplishes
3. **No parameter descriptions** - "data" lacks description of expected format, constraints, or examples
4. **No required fields** - Should specify which parameters are mandatory
5. **No type constraints** - "string" is too generic; should specify format (JSON, CSV, etc.)

*Grading:* 2-3 points each for identifying valid problems

**Question 4:** True (7 points)
*Explanation:* Tool calling is the LLM generating structured JSON matching the tool schema. The actual execution happens in your code. The model doesn't "call" anything—it predicts text that looks like a function call.

## Part B: Tool Design

**Question 5:** (10 points)
*Five Principles:*
1. **Single Responsibility** - One tool, one purpose (e.g., separate "search_users" and "update_user" instead of "manage_users")
2. **Clear Naming** - Use descriptive verbs (e.g., "calculate_tax" not "tax")
3. **Helpful Descriptions** - Explain when AND when not to use
4. **Safe Defaults** - Provide sensible default values
5. **Bounded Output** - Limit response sizes (e.g., max 100 results)

*Grading:* 1 point per principle, 2.5 points for two good examples

**Question 6:** Answer: Option B (8 points)
*Explanation:* Option B is better because:
- Focused scope (single responsibility)
- Clear guidance on when NOT to use it
- Well-documented parameters with descriptions and constraints
- Specified required fields
- Limited to essential parameters only (avoids complexity)

Option A has too many parameters, no descriptions, no required fields, and unclear scope.

**Question 7:** (9 points)
*Sample Answer:*
```python
{
    "name": "search_knowledge_base",
    "description": "Search the company's internal knowledge base for documents, policies, and procedures. Use when employees ask about internal processes, company policies, or need to find internal documentation. Returns relevant document excerpts with metadata.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query in natural language (e.g., 'expense reimbursement policy')"
            },
            "category": {
                "type": "string",
                "enum": ["policies", "procedures", "documentation", "all"],
                "description": "Filter by document category"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (1-20)",
                "default": 5
            }
        },
        "required": ["query"]
    }
}
```

*Grading:*
- Clear name (2 pts)
- Helpful description (2 pts)
- 3+ well-defined parameters (5 pts)

**Question 8:** (8 points)
*Sample Answer:* Bounded output means limiting the size/volume of data returned by a tool (e.g., max 100 search results, max 10KB response size). This is important because: (1) prevents context window overflow, (2) keeps costs predictable, (3) ensures agent can process results in reasonable time, and (4) prevents performance degradation from handling massive datasets.

## Part C: Error Handling

**Question 9:** (10 points)
*Sample Answer:*
1. **Validation Error** - User provides email in wrong format (e.g., "notanemail" instead of "user@example.com")
2. **Network Error** - External API is unreachable or times out
3. **Permission Error** - Agent tries to access a file/resource it doesn't have permission for
4. **Resource Not Found** - Requested document/user/item doesn't exist in database

*Grading:* 2.5 points per error type with valid example

**Question 10:** Answer: B (8 points)
*Explanation:* Retry with exponential backoff is standard practice for transient network errors. It gives the service time to recover while avoiding infinite loops. Clear error messages inform the agent to try alternative approaches if all retries fail.

**Question 11:** (9 points)
*Sample Answers:*

a) "ERROR: File 'report.pdf' not found at path /documents/. Available files: [list of existing files]. Please verify the filename or choose from available files."

b) "ERROR: Authentication required. API key is missing or invalid. Please configure the API_KEY environment variable before using this tool."

c) "ERROR: Search returned 5,234 results (limit: 100). Please refine your search query with more specific terms or filters. Example: Add date range or category filter."

*Grading:* 3 points each for clear, actionable messages

**Question 12:** (8 points)
*Sample Answer:*
**Fail fast** immediately stops execution on error, preventing cascading failures. Use when: Data integrity is critical (e.g., financial transactions).

**Graceful degradation** continues operation with reduced functionality when errors occur. Use when: Partial results are better than none (e.g., search returning cached results if live API fails).

## Part D: Security & Best Practices

**Question 13:** (5 points)
*Risks:*
1. **Code injection attacks** - Malicious users could execute arbitrary commands (rm -rf /, data theft)
2. **System compromise** - Agent could delete files, access sensitive data, install malware
3. **Resource abuse** - Infinite loops, cryptocurrency mining, denial of service
4. **Privilege escalation** - If agent runs with elevated permissions, could compromise entire system

*Grading:* 2.5 points per valid risk

**Question 14:** (5 points)
*Sample Answer:* Implement path validation to restrict file operations to a specific allowed directory (e.g., sandboxed workspace). Validate all paths and reject attempts to access parent directories (../) or absolute paths outside allowed zones.

*Acceptable alternatives:*
- Run agent in isolated container/sandbox
- Implement file size limits
- Check file extensions/types against allowlist
- Require explicit approval for destructive operations

---

## Scoring Guide

- **90-100 points:** Excellent - Strong understanding of tool design and error handling
- **80-89 points:** Good - Solid grasp with minor gaps
- **70-79 points:** Passing - Core concepts understood, practice implementation
- **Below 70:** Review module and practice building tools

**Key Topics to Review if Struggling:**
- Tool calling flow and JSON schema design
- Single responsibility principle
- Bounded output and safe defaults
- Retry logic and graceful degradation
- Security controls for tool execution
- Error message design for agent comprehension
