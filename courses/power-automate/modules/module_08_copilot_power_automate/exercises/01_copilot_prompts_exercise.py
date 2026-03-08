"""
Module 08: Copilot Prompts and AI Builder — Self-Check Exercises

These exercises build practical skills for using Copilot in Power Automate effectively:
- Writing specific Copilot prompts that produce good flow structures
- Identifying which AI Builder action fits a given use case
- Recognizing and improving poorly-written prompts

Each exercise is self-validating. Run the file to check your answers:
    python 01_copilot_prompts_exercise.py

No Power Automate account or API credentials required.
"""

# =============================================================================
# Exercise 1: Identify Missing Elements in Copilot Prompts
# =============================================================================
#
# A good Copilot prompt includes:
#   (T) Trigger  — what starts the flow
#   (S) Systems  — specific Microsoft 365 / Power Platform services named
#   (A) Actions  — what the flow does
#   (C) Conditions — any branching or filtering logic (when applicable)
#
# For each prompt below, identify which elements are MISSING.
# Complete the MISSING_ELEMENTS dict. Values are lists of element codes from {T, S, A, C}.
# If nothing is missing, use an empty list.

PROMPTS = {
    "prompt_1": "save email attachments and notify the team",
    "prompt_2": (
        "When a new item is added to the SharePoint list 'Purchase Requests' on the "
        "Procurement site, send an approval request to the requestor's manager in Teams. "
        "If approved, update the item status to 'Approved'. If rejected, update to "
        "'Rejected' and send the requestor an email explaining the reason from the "
        "approver's comments."
    ),
    "prompt_3": "do something with files every day",
    "prompt_4": (
        "When a new row is added to the Excel file 'Sales Data.xlsx' in OneDrive, "
        "post a message to the #sales-updates Teams channel with the new row values."
    ),
    "prompt_5": "when a form is submitted notify me",
}

MISSING_ELEMENTS = {
    # Each key maps to a list of missing element codes: "T", "S", "A", "C"
    # "C" (condition) is only required when the scenario implies branching
    # Example: {"prompt_X": ["T", "S"]}
    "prompt_1": [],  # TODO: fill in
    "prompt_2": [],  # TODO: fill in
    "prompt_3": [],  # TODO: fill in
    "prompt_4": [],  # TODO: fill in
    "prompt_5": [],  # TODO: fill in
}


def test_exercise_1():
    """Test identification of missing prompt elements."""
    correct_answers = {
        "prompt_1": sorted(["T", "S", "C"]),
        # prompt_1 is missing: explicit trigger (what kind of email? what app?),
        # specific systems (which email? which team notification service?),
        # and no filtering condition (all emails? only with attachments?)
        "prompt_2": [],
        # prompt_2 has all elements: trigger (new SharePoint item), systems (SharePoint, Teams, Email),
        # actions (send approval, update status, send email), conditions (if approved/rejected)
        "prompt_3": sorted(["T", "S", "A", "C"]),
        # prompt_3 is missing everything: trigger is vague ("every day" is a schedule
        # but "do something with files" has no named system, no specific action, no condition)
        "prompt_4": [],
        # prompt_4 has all required elements for its scenario:
        # trigger (new row in Excel), systems (Excel/OneDrive, Teams), action (post message)
        # No condition required — it's a simple linear flow
        "prompt_5": sorted(["S", "C"]),
        # prompt_5 has a trigger (form submitted) and action (notify)
        # but is missing: which form service (Forms? SharePoint list?), how to notify (email? Teams?)
        # and no condition (which forms? all submissions? specific responses?)
    }

    all_passed = True
    for prompt_key, expected in correct_answers.items():
        student_answer = sorted(MISSING_ELEMENTS.get(prompt_key, []))
        if student_answer != expected:
            print(f"  INCORRECT — {prompt_key}")
            print(f"    Your answer : {student_answer}")
            print(f"    Expected    : {expected}")
            print(f"    Explanation : See the comment above '{prompt_key}' in correct_answers")
            all_passed = False

    if all_passed:
        print("Exercise 1 passed: all missing element identifications are correct.")
    else:
        print("Exercise 1: some answers need revision. See explanations above.")


# =============================================================================
# Exercise 2: Match AI Builder Actions to Use Cases
# =============================================================================
#
# For each business scenario below, select the most appropriate AI Builder action.
# Use the action key strings exactly as they appear in AVAILABLE_ACTIONS.
#
# AVAILABLE_ACTIONS:
#   "sentiment_analysis"        - Classify text as positive/negative/neutral/mixed
#   "entity_extraction"         - Extract people, organizations, dates, locations from text
#   "category_classification"   - Assign text to one of your defined categories
#   "invoice_processing"        - Extract structured fields from invoice PDFs/images
#   "receipt_processing"        - Extract structured fields from receipt images
#   "gpt_text_generation"       - Generate, summarize, or transform text using a prompt
#   "key_phrase_extraction"     - Extract the key topics/phrases from text
#
# NOTE: Some scenarios may have more than one reasonable answer. The test accepts
# the primary best-fit action. See explanations in the test for alternatives.

USE_CASES = {
    "case_1": (
        "Incoming customer support emails need to be automatically routed: "
        "unhappy customers get prioritized and assigned to senior agents."
    ),
    "case_2": (
        "A flow processes vendor invoices attached to emails. "
        "The vendor name, invoice number, and total amount need to be written "
        "to a SharePoint list automatically."
    ),
    "case_3": (
        "Survey responses from a Microsoft Form need to be tagged with the main "
        "topic areas discussed so the marketing team can filter by theme."
    ),
    "case_4": (
        "Contract documents arrive as PDFs via email. A flow needs to extract "
        "all company names, dates, and monetary values mentioned in the contract "
        "body to populate a contract tracking spreadsheet."
    ),
    "case_5": (
        "Customer complaint emails need a professional acknowledgment reply drafted "
        "automatically. The draft should reference the specific issue mentioned "
        "and commit to a follow-up within 24 hours."
    ),
    "case_6": (
        "Expense receipts photographed on mobile devices need to be processed: "
        "merchant name, date, and total should be extracted for expense report submission."
    ),
    "case_7": (
        "A flow monitors internal project update emails and automatically generates "
        "a two-sentence summary of each update for the executive dashboard feed."
    ),
    "case_8": (
        "IT support tickets submitted via a web form need to be classified into "
        "categories: Hardware, Software, Network, Access Request, or Other."
    ),
}

ACTION_SELECTIONS = {
    # Replace None with one of the action key strings from AVAILABLE_ACTIONS
    "case_1": None,  # TODO
    "case_2": None,  # TODO
    "case_3": None,  # TODO
    "case_4": None,  # TODO
    "case_5": None,  # TODO
    "case_6": None,  # TODO
    "case_7": None,  # TODO
    "case_8": None,  # TODO
}


def test_exercise_2():
    """Test AI Builder action selection for given use cases."""
    # Primary best-fit answers
    primary_answers = {
        "case_1": "sentiment_analysis",
        "case_2": "invoice_processing",
        "case_3": "key_phrase_extraction",
        "case_4": "entity_extraction",
        "case_5": "gpt_text_generation",
        "case_6": "receipt_processing",
        "case_7": "gpt_text_generation",
        "case_8": "category_classification",
    }

    # Alternative acceptable answers with explanations
    alternatives = {
        "case_3": {
            "category_classification": (
                "category_classification also works if the marketing team has a defined list "
                "of categories. key_phrase_extraction is better when topics are not predefined."
            ),
        },
        "case_7": {
            # No strong alternative — GPT is definitively the right choice for generation
        },
    }

    explanations = {
        "case_1": (
            "sentiment_analysis classifies tone (negative = unhappy customer = high priority). "
            "The negative_score threshold drives the routing condition."
        ),
        "case_2": (
            "invoice_processing is specifically trained on invoice documents and returns "
            "vendor name, invoice number, total, line items as structured fields."
        ),
        "case_3": (
            "key_phrase_extraction returns the main topics from free-form text without requiring "
            "a predefined category list. Ideal when themes emerge organically from survey data."
        ),
        "case_4": (
            "entity_extraction identifies and labels named entities: Person, Organization, "
            "DateTime, Quantity. Exactly matches the need to extract company names, dates, amounts."
        ),
        "case_5": (
            "gpt_text_generation with a role-based prompt (e.g. 'You are a customer service manager, "
            "write an empathetic acknowledgment...') produces professional, context-aware draft replies."
        ),
        "case_6": (
            "receipt_processing is specifically trained on retail/restaurant/travel receipts "
            "and returns merchant name, date, line items, total as structured fields."
        ),
        "case_7": (
            "gpt_text_generation with a summarization prompt (e.g. 'Summarize in 2 sentences:') "
            "is the right choice for generating new text from existing text."
        ),
        "case_8": (
            "category_classification accepts your defined category list (Hardware, Software, "
            "Network, Access Request, Other) and assigns the best match. Not GPT — you want "
            "consistent categorical output, not generated text."
        ),
    }

    all_passed = True
    for case_key, primary in primary_answers.items():
        student_answer = ACTION_SELECTIONS.get(case_key)

        if student_answer is None:
            print(f"  TODO — {case_key}: no answer provided yet")
            all_passed = False
            continue

        case_alternatives = {primary} | set(alternatives.get(case_key, {}).keys())
        if student_answer not in case_alternatives:
            print(f"  INCORRECT — {case_key}")
            print(f"    Your answer : {student_answer}")
            print(f"    Best answer : {primary}")
            print(f"    Explanation : {explanations[case_key]}")
            all_passed = False
        elif student_answer != primary and student_answer in alternatives.get(case_key, {}):
            alt_explanation = alternatives[case_key][student_answer]
            print(f"  ACCEPTABLE — {case_key}: '{student_answer}' is valid but not the primary choice")
            print(f"    Primary answer : {primary}")
            print(f"    Why alternative works: {alt_explanation}")

    if all_passed:
        print("Exercise 2 passed: all AI Builder action selections are correct.")
    elif all(ACTION_SELECTIONS[k] is not None for k in ACTION_SELECTIONS):
        print("Exercise 2: some answers need revision. See explanations above.")


# =============================================================================
# Exercise 3: Rewrite Weak Prompts
# =============================================================================
#
# Each WEAK_PROMPT below is poorly written and would produce a vague or incorrect
# flow from Copilot. For each one, write an IMPROVED_PROMPT that would generate
# a more accurate and useful flow structure.
#
# Evaluation criteria (all three must be present in your improved prompt):
#   1. Specific trigger with the named service/connector
#   2. Named Microsoft 365 / Power Platform systems for actions
#   3. At least one specific condition or filter (where the scenario implies one)
#
# Fill in the IMPROVED_PROMPTS dict below with string values.

WEAK_PROMPTS = {
    "weak_1": "send a reminder when something is due soon",
    "weak_2": "process files and update the database",
    "weak_3": "approval flow for requests",
}

IMPROVED_PROMPTS = {
    "weak_1": "",  # TODO: write an improved prompt for weak_1
    "weak_2": "",  # TODO: write an improved prompt for weak_2
    "weak_3": "",  # TODO: write an improved prompt for weak_3
}


def test_exercise_3():
    """
    Test that improved prompts contain the required elements.

    Checks for structural completeness rather than exact wording.
    A prompt passes if it contains keywords indicating each required element.
    """
    # Keywords that indicate each requirement is present.
    # Each requirement is a list of keyword sets; at least one set must match.
    requirements = {
        "weak_1": {
            "trigger_service": [
                {"sharepoint", "planner", "outlook", "calendar", "tasks", "scheduled",
                 "recurrence", "when a", "when an"},
            ],
            "named_action_service": [
                {"email", "outlook", "teams", "notification", "reminder", "message",
                 "send", "post"},
            ],
            "specificity": [
                # Must mention what "due soon" means concretely
                {"days", "hours", "before", "due date", "upcoming", "within"},
            ],
        },
        "weak_2": {
            "trigger_service": [
                {"sharepoint", "onedrive", "teams", "when a new file", "when a file",
                 "when a new item"},
            ],
            "named_system": [
                {"sharepoint", "excel", "dataverse", "sql", "list", "table"},
            ],
            "file_type_or_action": [
                {"pdf", "invoice", "excel", "csv", "document", "extract", "process",
                 "convert", "attachment"},
            ],
        },
        "weak_3": {
            "trigger_service": [
                {"sharepoint", "forms", "when a", "when an", "new request",
                 "item is added", "response is submitted"},
            ],
            "approval_action": [
                {"approval", "approve", "approver", "manager", "teams", "outlook"},
            ],
            "branch_condition": [
                {"if approved", "if rejected", "approved", "rejected", "yes", "no",
                 "denied", "accepted"},
            ],
        },
    }

    all_passed = True
    for prompt_key, checks in requirements.items():
        student_prompt = IMPROVED_PROMPTS.get(prompt_key, "").lower()

        if not student_prompt.strip():
            print(f"  TODO — {prompt_key}: no improved prompt provided yet")
            all_passed = False
            continue

        prompt_passed = True
        for check_name, keyword_option_groups in checks.items():
            # For each requirement, at least one keyword from any option group must appear.
            # keyword_option_groups is a list of sets; the requirement is met if any single
            # keyword from any set appears in the student's prompt.
            requirement_met = any(
                any(kw in student_prompt for kw in keyword_group)
                for keyword_group in keyword_option_groups
            )
            if not requirement_met:
                print(f"  INCOMPLETE — {prompt_key}: missing '{check_name}'")
                print(f"    Hint: include one of these themes: {keyword_option_groups}")
                prompt_passed = False
                all_passed = False

        if prompt_passed:
            print(f"  PASSED — {prompt_key}: contains all required elements")

    if all_passed:
        print("Exercise 3 passed: all improved prompts include required elements.")
    elif all(IMPROVED_PROMPTS[k].strip() for k in IMPROVED_PROMPTS):
        print("Exercise 3: some prompts need strengthening. See hints above.")


# =============================================================================
# Exercise 4: Expression Use Cases
# =============================================================================
#
# For each expression description, identify which Power Automate expression
# function is the PRIMARY function needed to accomplish it.
# Use lowercase function names exactly as they appear in Power Automate expressions.
#
# Reference function list:
#   formatDateTime  - Format a date/time value as a string
#   utcNow          - Get the current UTC date and time
#   substring       - Extract a portion of a string
#   contains        - Check if a string or array contains a value
#   length          - Get the length of a string or array
#   toLower         - Convert a string to lowercase
#   toUpper         - Convert a string to uppercase
#   concat          - Join two or more strings together
#   replace         - Replace all occurrences of a string within a string
#   trim            - Remove leading and trailing whitespace from a string
#   split           - Split a string into an array by a delimiter
#   int             - Convert a string to an integer
#   float           - Convert a string to a float
#   string          - Convert a value to its string representation
#   if              - Return one value if condition is true, another if false
#   empty           - Check whether a string, array, or object is empty
#   first           - Get the first item from an array
#   last            - Get the last item from an array

EXPRESSION_TASKS = {
    "task_1": "Get today's date formatted as YYYY-MM-DD for a SharePoint date field",
    "task_2": "Check if an email subject line contains the word 'URGENT' (case-insensitive)",
    "task_3": "Get only the first 200 characters of a long email body",
    "task_4": "Combine a customer's first name and last name into a single 'Full Name' field",
    "task_5": "Check if the list of attachments on an email is empty (no attachments)",
    "task_6": "Convert the user's input '42' (a string) to a number for arithmetic",
    "task_7": "Get the last item in an array of approval comments",
    "task_8": "Remove extra whitespace from a form field value before saving to SharePoint",
}

EXPRESSION_FUNCTION_SELECTIONS = {
    "task_1": None,  # TODO: primary function name
    "task_2": None,  # TODO: primary function name
    "task_3": None,  # TODO: primary function name
    "task_4": None,  # TODO: primary function name
    "task_5": None,  # TODO: primary function name
    "task_6": None,  # TODO: primary function name
    "task_7": None,  # TODO: primary function name
    "task_8": None,  # TODO: primary function name
}


def test_exercise_4():
    """Test expression function identification."""
    primary_answers = {
        "task_1": "formatDateTime",
        # utcNow() provides the date, but formatDateTime is the PRIMARY function
        # that applies the YYYY-MM-DD format. Full expression: formatDateTime(utcNow(), 'yyyy-MM-dd')
        "task_2": "contains",
        # Use toLower first, then contains. But "contains" is the PRIMARY function
        # that checks presence. Full: contains(toLower(triggerBody()?['Subject']), 'urgent')
        "task_3": "substring",
        # Full: substring(triggerBody()?['Body'], 0, 200)
        "task_4": "concat",
        # Full: concat(triggerBody()?['FirstName'], ' ', triggerBody()?['LastName'])
        "task_5": "empty",
        # Full: empty(triggerBody()?['Attachments'])
        "task_6": "int",
        # Full: int(triggerBody()?['UserInput'])
        "task_7": "last",
        # Full: last(variables('ApprovalComments'))
        "task_8": "trim",
        # Full: trim(triggerBody()?['FormField'])
    }

    full_expressions = {
        "task_1": "formatDateTime(utcNow(), 'yyyy-MM-dd')",
        "task_2": "contains(toLower(triggerBody()?['Subject']), 'urgent')",
        "task_3": "substring(triggerBody()?['Body'], 0, 200)",
        "task_4": "concat(triggerBody()?['FirstName'], ' ', triggerBody()?['LastName'])",
        "task_5": "empty(triggerBody()?['Attachments'])",
        "task_6": "int(triggerBody()?['UserInput'])",
        "task_7": "last(variables('ApprovalComments'))",
        "task_8": "trim(triggerBody()?['FormField'])",
    }

    all_passed = True
    for task_key, expected in primary_answers.items():
        student_answer = EXPRESSION_FUNCTION_SELECTIONS.get(task_key)

        if student_answer is None:
            print(f"  TODO — {task_key}: no answer provided yet")
            all_passed = False
            continue

        if student_answer != expected:
            print(f"  INCORRECT — {task_key}")
            print(f"    Your answer    : {student_answer}")
            print(f"    Expected       : {expected}")
            print(f"    Full expression: {full_expressions[task_key]}")
            all_passed = False

    if all_passed:
        print("Exercise 4 passed: all expression function selections are correct.")
    elif all(EXPRESSION_FUNCTION_SELECTIONS[k] is not None for k in EXPRESSION_FUNCTION_SELECTIONS):
        print("Exercise 4: some answers need revision. See full expressions above.")


# =============================================================================
# Main: Run all exercises
# =============================================================================

def main():
    print("=" * 65)
    print("Module 08: Copilot Prompts and AI Builder — Self-Check")
    print("=" * 65)
    print()

    print("Exercise 1: Identifying Missing Prompt Elements")
    print("-" * 45)
    test_exercise_1()
    print()

    print("Exercise 2: Matching AI Builder Actions to Use Cases")
    print("-" * 45)
    test_exercise_2()
    print()

    print("Exercise 3: Rewriting Weak Copilot Prompts")
    print("-" * 45)
    test_exercise_3()
    print()

    print("Exercise 4: Expression Function Identification")
    print("-" * 45)
    test_exercise_4()
    print()

    print("=" * 65)
    print("Complete all TODO items above and re-run to verify your answers.")
    print("=" * 65)


if __name__ == "__main__":
    main()
