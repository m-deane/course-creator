"""
Module 06: Approval Flows and Business Process Patterns
Self-Check Exercises

These exercises build your design thinking for Power Automate approval flows.
You will:
  - Select the correct approval type for a given business scenario
  - Design Adaptive Card JSON for specific approval experiences
  - Map business process patterns to their implementation components

Run with: python 01_approval_flow_exercise.py

No Power Automate environment required — all exercises are logic and JSON design.
"""

import json
from typing import Any


# =============================================================================
# Exercise 1: Match Approval Types to Business Scenarios
# =============================================================================
#
# Power Automate offers four approval types:
#
#   "first_to_respond"      Approve/Reject — First to respond
#   "everyone_must_approve" Approve/Reject — Everyone must approve
#   "custom_first"          Custom Responses — First to respond
#   "custom_everyone"       Custom Responses — Everyone must respond
#
# For each scenario below, assign the correct approval type string.
# Read the scenario carefully — the routing behavior matters.

def get_approval_type_for_scenario(scenario_id: int) -> str:
    """
    Return the correct approval type string for each scenario.

    Parameters
    ----------
    scenario_id : int
        1 = Time-off request routed to any available team lead
        2 = Budget over $50,000 requiring CFO and board chair sign-off
        3 = IT change request that can be approved, rejected, or deferred
        4 = Vendor contract requiring legal, finance, and ops review
            where each reviewer must respond with: Approve, Reject,
            or Request Revisions (not a simple yes/no)

    Returns
    -------
    str
        One of: "first_to_respond", "everyone_must_approve",
                "custom_first", "custom_everyone"
    """
    # TODO: Return the correct approval type for each scenario_id
    # Replace None with the appropriate string for each case

    if scenario_id == 1:
        # Time-off request: any team lead can authorize, no custom responses needed
        return None

    elif scenario_id == 2:
        # Major budget: both CFO and board chair must individually approve
        # Standard yes/no is sufficient
        return None

    elif scenario_id == 3:
        # IT change: single approver, but three possible responses
        # (Approve, Reject, Defer) — first responder decides
        return None

    elif scenario_id == 4:
        # Vendor contract: three approvers, each must choose from
        # Approve / Reject / Request Revisions (not just yes/no)
        return None

    else:
        raise ValueError(f"Unknown scenario_id: {scenario_id}")


# Self-check tests for Exercise 1
def test_exercise_1():
    """Verify correct approval type selection for each scenario."""

    result_1 = get_approval_type_for_scenario(1)
    assert result_1 == "first_to_respond", (
        f"Scenario 1: Time-off can be approved by any team lead — "
        f"'first_to_respond' is correct. Got: {result_1!r}"
    )

    result_2 = get_approval_type_for_scenario(2)
    assert result_2 == "everyone_must_approve", (
        f"Scenario 2: Both CFO and board chair must approve individually — "
        f"'everyone_must_approve' is correct. Got: {result_2!r}"
    )

    result_3 = get_approval_type_for_scenario(3)
    assert result_3 == "custom_first", (
        f"Scenario 3: Single approver with 3 response options — "
        f"'custom_first' is correct. Got: {result_3!r}"
    )

    result_4 = get_approval_type_for_scenario(4)
    assert result_4 == "custom_everyone", (
        f"Scenario 4: Three approvers, each must respond with custom options — "
        f"'custom_everyone' is correct. Got: {result_4!r}"
    )

    print("[PASS] Exercise 1: All approval type selections are correct")


# =============================================================================
# Exercise 2: Design Adaptive Card JSON for a Specific Approval
# =============================================================================
#
# Build the Adaptive Card JSON (as a Python dict) for a purchase order
# approval card with these exact requirements:
#
# Required inputs:
#   - id="decision": ChoiceSet with choices Approve ("approve") and
#     Reject ("reject"), required, expanded style
#   - id="comments": multiline text input, not required
#
# Required facts (in a FactSet):
#   - "PO Number:" → the value from the parameter po_number
#   - "Vendor:" → the value from the parameter vendor
#   - "Total Value:" → the value from the parameter total_value
#   - "Requested by:" → the value from the parameter requested_by
#
# Card structure requirements:
#   - type: "AdaptiveCard"
#   - version: "1.4"
#   - At least one TextBlock header visible (size Large or ExtraLarge)
#   - All four facts present in a FactSet
#   - Both inputs present with correct ids
#   - At least one Action.Submit action

def build_purchase_order_card(
    po_number: str,
    vendor: str,
    total_value: str,
    requested_by: str
) -> dict:
    """
    Build an Adaptive Card for a purchase order approval.

    Parameters
    ----------
    po_number : str
        Purchase order identifier, e.g. "PO-2024-00892"
    vendor : str
        Vendor name, e.g. "Contoso Supplies Ltd."
    total_value : str
        Formatted total value string, e.g. "$12,450.00"
    requested_by : str
        Requestor display name, e.g. "Marcus Webb"

    Returns
    -------
    dict
        Complete Adaptive Card dictionary meeting all requirements above.

    Example
    -------
    >>> card = build_purchase_order_card(
    ...     po_number="PO-2024-00892",
    ...     vendor="Contoso Supplies Ltd.",
    ...     total_value="$12,450.00",
    ...     requested_by="Marcus Webb"
    ... )
    >>> assert card["type"] == "AdaptiveCard"
    """
    # TODO: Build and return the Adaptive Card dictionary
    # Use the parameters to populate the FactSet values
    return None


# Self-check tests for Exercise 2
def test_exercise_2():
    """Verify the purchase order card structure and content."""

    card = build_purchase_order_card(
        po_number="PO-2024-00892",
        vendor="Contoso Supplies Ltd.",
        total_value="$12,450.00",
        requested_by="Marcus Webb"
    )

    assert card is not None, \
        "build_purchase_order_card() returned None. Implement the function."

    assert isinstance(card, dict), \
        "Return value must be a dict, not a string or other type."

    # Top-level fields
    assert card.get("type") == "AdaptiveCard", \
        f"Card 'type' must be 'AdaptiveCard'. Got: {card.get('type')!r}"

    assert card.get("version") == "1.4", \
        f"Card 'version' must be '1.4'. Got: {card.get('version')!r}"

    body = card.get("body", [])
    assert isinstance(body, list) and len(body) >= 2, \
        "Card body must be a list with at least 2 elements (header + facts)"

    # Must have a large TextBlock header
    large_text_blocks = [
        elem for elem in body
        if isinstance(elem, dict)
        and elem.get("type") == "TextBlock"
        and elem.get("size") in ("Large", "ExtraLarge")
    ]
    assert len(large_text_blocks) >= 1, \
        "Card must have at least one TextBlock with size='Large' or size='ExtraLarge'"

    # Must have a FactSet with all four required facts
    def find_all_fact_sets(elements: list) -> list:
        sets = []
        for elem in elements:
            if not isinstance(elem, dict):
                continue
            if elem.get("type") == "FactSet":
                sets.append(elem)
            if "items" in elem:
                sets.extend(find_all_fact_sets(elem["items"]))
        return sets

    fact_sets = find_all_fact_sets(body)
    assert len(fact_sets) >= 1, \
        "Card must contain at least one FactSet element"

    # Collect all fact titles and values across all FactSets
    all_facts = []
    for fs in fact_sets:
        all_facts.extend(fs.get("facts", []))

    fact_titles = [f.get("title", "") for f in all_facts]
    fact_values = [f.get("value", "") for f in all_facts]

    assert any("PO" in t or "po" in t.lower() for t in fact_titles), \
        "FactSet must include a fact with 'PO Number' in the title"

    assert "PO-2024-00892" in fact_values, \
        "FactSet must include the po_number value 'PO-2024-00892'"

    assert "Contoso Supplies Ltd." in fact_values, \
        "FactSet must include the vendor value 'Contoso Supplies Ltd.'"

    assert "$12,450.00" in fact_values, \
        "FactSet must include the total_value '$12,450.00'"

    assert "Marcus Webb" in fact_values, \
        "FactSet must include the requested_by value 'Marcus Webb'"

    # Collect all input ids
    def collect_input_ids(elements: list) -> list:
        ids = []
        input_types = {
            "Input.Text", "Input.Number", "Input.Date",
            "Input.Time", "Input.Toggle", "Input.ChoiceSet"
        }
        for elem in elements:
            if not isinstance(elem, dict):
                continue
            if elem.get("type") in input_types and elem.get("id"):
                ids.append(elem["id"])
            if "items" in elem:
                ids.extend(collect_input_ids(elem["items"]))
        return ids

    input_ids = collect_input_ids(body)

    assert "decision" in input_ids, \
        "Card must have an input with id='decision'"

    assert "comments" in input_ids, \
        "Card must have an input with id='comments'"

    # Verify decision input is a ChoiceSet with correct choices
    def find_input_by_id(elements: list, target_id: str) -> dict | None:
        for elem in elements:
            if not isinstance(elem, dict):
                continue
            if elem.get("id") == target_id:
                return elem
            if "items" in elem:
                found = find_input_by_id(elem["items"], target_id)
                if found:
                    return found
        return None

    decision_elem = find_input_by_id(body, "decision")
    assert decision_elem is not None, "Could not locate the 'decision' input element"
    assert decision_elem.get("type") == "Input.ChoiceSet", \
        f"'decision' must be Input.ChoiceSet. Got: {decision_elem.get('type')!r}"

    choices = decision_elem.get("choices", [])
    assert len(choices) == 2, \
        f"'decision' must have exactly 2 choices (Approve, Reject). Found {len(choices)}"

    choice_values = {c.get("value") for c in choices}
    assert "approve" in choice_values and "reject" in choice_values, \
        f"Choices must have values 'approve' and 'reject'. Found: {choice_values}"

    # Verify comments is multiline
    comments_elem = find_input_by_id(body, "comments")
    assert comments_elem is not None, "Could not locate the 'comments' input element"
    assert comments_elem.get("isMultiline") is True, \
        "The 'comments' input must have isMultiline=True"

    # Must have at least one Action.Submit
    actions = card.get("actions", [])
    submit_actions = [a for a in actions if isinstance(a, dict) and a.get("type") == "Action.Submit"]
    assert len(submit_actions) >= 1, \
        "Card must have at least one Action.Submit in the 'actions' array"

    print("[PASS] Exercise 2: Purchase order card is structurally correct with all required fields")


# =============================================================================
# Exercise 3: Map Business Patterns to Implementation Components
# =============================================================================
#
# Each business requirement below maps to one or more specific Power Automate
# implementation components. Match each requirement to its correct component(s).
#
# Available components:
#   "parallel_branches"         Two branches running simultaneously
#   "delay_action"              Pause flow for a specified duration
#   "do_until_loop"             Repeat actions until a condition is met
#   "switch_action"             Branch based on a value with multiple cases
#   "initialize_variable"       Create a variable before using it in branches
#   "create_approval"           Non-blocking approval creation
#   "wait_for_approval"         Pause flow until an existing approval gets a response
#   "start_and_wait_approval"   Single blocking approval action
#   "condition_action"          Two-branch yes/no decision
#   "office365_get_manager"     Look up a user's manager email
#   "update_sharepoint_item"    Write data to a SharePoint list record
#   "send_email"                Deliver a notification email

def get_components_for_requirement(requirement_id: int) -> list[str]:
    """
    Return the list of Power Automate components needed for a requirement.

    Parameters
    ----------
    requirement_id : int
        1 = "Route approval to the submitter's manager automatically
             (manager's email is not known in advance)"
        2 = "After creating the approval, immediately send the requestor
             a 'Your request is under review' email without waiting
             for the approver to respond"
        3 = "Escalate to a VP if the manager does not respond within 72 hours"
        4 = "Route high-value requests (over $10,000) to the CFO and
             standard requests to the department head"
        5 = "Record the approver's name, decision, and timestamp in
             the original SharePoint request record after they respond"

    Returns
    -------
    list[str]
        Sorted list of component name strings from the available components above.
        Order within the list does not matter — tests check membership.
    """
    # TODO: Return the correct list of components for each requirement_id.
    # Some requirements need only one component; others need several.

    if requirement_id == 1:
        # Automatic manager lookup — what connector action does this?
        return []

    elif requirement_id == 2:
        # Send email immediately after creating approval, before response arrives.
        # Key: you need a non-blocking approval + the email action.
        # You also need to wait for the response somewhere later.
        return []

    elif requirement_id == 3:
        # Escalation on timeout: need to create the approval non-blocking,
        # then run a timer and a wait branch in parallel,
        # with a variable to track whether a response arrived.
        return []

    elif requirement_id == 4:
        # Conditional routing by value with more than 2 cases.
        # You need to set a variable for the approver email before the approval.
        return []

    elif requirement_id == 5:
        # After approval response, write data back to SharePoint.
        return []

    else:
        raise ValueError(f"Unknown requirement_id: {requirement_id}")


# Self-check tests for Exercise 3
def test_exercise_3():
    """Verify correct component mapping for each requirement."""

    # Requirement 1: automatic manager lookup
    r1 = set(get_components_for_requirement(1))
    assert "office365_get_manager" in r1, (
        "Requirement 1: The Office 365 'Get manager (V2)' action resolves the manager email. "
        f"Got: {r1}"
    )

    # Requirement 2: send email while approval is pending
    r2 = set(get_components_for_requirement(2))
    assert "create_approval" in r2, (
        "Requirement 2: Use 'Create an approval' (non-blocking) so the flow continues immediately. "
        f"Got: {r2}"
    )
    assert "send_email" in r2, (
        "Requirement 2: 'Send an email' action sends the acknowledgment before waiting. "
        f"Got: {r2}"
    )
    assert "wait_for_approval" in r2, (
        "Requirement 2: 'Wait for an approval' action retrieves the response later. "
        f"Got: {r2}"
    )

    # Requirement 3: escalation on timeout
    r3 = set(get_components_for_requirement(3))
    assert "parallel_branches" in r3, (
        "Requirement 3: Parallel branches let the timer and the approval response race. "
        f"Got: {r3}"
    )
    assert "delay_action" in r3, (
        "Requirement 3: A 'Delay' action creates the 72-hour timer in the timeout branch. "
        f"Got: {r3}"
    )
    assert "create_approval" in r3, (
        "Requirement 3: The approval must be non-blocking (Create) to allow parallel branches. "
        f"Got: {r3}"
    )
    assert "initialize_variable" in r3, (
        "Requirement 3: A variable tracks whether the approval was responded to, "
        "since parallel branches cannot share state otherwise. "
        f"Got: {r3}"
    )

    # Requirement 4: conditional routing by amount
    r4 = set(get_components_for_requirement(4))
    assert "switch_action" in r4, (
        "Requirement 4: A Switch action handles multiple routing cases more cleanly "
        "than nested Conditions. Got: {r4}"
    )
    assert "initialize_variable" in r4, (
        "Requirement 4: Variables store the approver email and approval type before the "
        "approval action, which reads them dynamically. "
        f"Got: {r4}"
    )

    # Requirement 5: write response to SharePoint
    r5 = set(get_components_for_requirement(5))
    assert "update_sharepoint_item" in r5, (
        "Requirement 5: 'Update item' (SharePoint) writes the decision back to the record. "
        f"Got: {r5}"
    )

    print("[PASS] Exercise 3: All component mappings are correct")


# =============================================================================
# Exercise 4: Design the Condition Logic for a Multi-Stage Pipeline
# =============================================================================
#
# A two-stage approval pipeline produces one of these final outcomes depending
# on what happened at each stage:
#
#   Stage 1 result: "approve" or "reject"
#   Stage 2 result: "approve" or "reject"  (only reached if Stage 1 approved)
#
# Complete the function below so it returns the correct final_status and
# notification_targets for each combination.

def get_pipeline_outcome(
    stage1_decision: str,
    stage2_decision: str | None
) -> dict[str, Any]:
    """
    Determine the final outcome and notification targets for a two-stage pipeline.

    Parameters
    ----------
    stage1_decision : str
        The Stage 1 approver's decision: "approve" or "reject"
    stage2_decision : str or None
        The Stage 2 approver's decision: "approve", "reject", or None.
        None when Stage 1 rejected (Stage 2 never ran).

    Returns
    -------
    dict with keys:
        final_status : str
            One of: "Fully Approved", "Rejected - Stage 1", "Rejected - Stage 2"
        notification_targets : list[str]
            List of who receives a notification. Must include the appropriate
            subset of: "requestor", "stage1_approver", "stage2_approver", "finance_team"

    Rules:
        - Stage 1 rejects: status="Rejected - Stage 1",
          notify: requestor and stage1_approver
        - Stage 2 rejects: status="Rejected - Stage 2",
          notify: requestor, stage1_approver, and stage2_approver
        - Both approve: status="Fully Approved",
          notify: requestor, stage1_approver, stage2_approver, and finance_team

    Example
    -------
    >>> get_pipeline_outcome("approve", "approve")
    {'final_status': 'Fully Approved',
     'notification_targets': ['requestor', 'stage1_approver', 'stage2_approver', 'finance_team']}
    """
    # TODO: Implement the pipeline outcome logic
    return {}


# Self-check tests for Exercise 4
def test_exercise_4():
    """Verify pipeline outcome logic for all three scenarios."""

    # Scenario A: Stage 1 rejects immediately
    result_a = get_pipeline_outcome("reject", None)
    assert isinstance(result_a, dict), \
        "Return value must be a dict with 'final_status' and 'notification_targets'"

    assert result_a.get("final_status") == "Rejected - Stage 1", (
        "When Stage 1 rejects, final_status should be 'Rejected - Stage 1'. "
        f"Got: {result_a.get('final_status')!r}"
    )
    targets_a = set(result_a.get("notification_targets", []))
    assert "requestor" in targets_a, \
        "Rejection at Stage 1: requestor must be notified"
    assert "stage1_approver" in targets_a, \
        "Rejection at Stage 1: stage1_approver must be notified"
    assert "stage2_approver" not in targets_a, \
        "Rejection at Stage 1: stage2_approver was not involved, should not be notified"
    assert "finance_team" not in targets_a, \
        "Rejection at Stage 1: finance_team should not be notified on early rejection"

    # Scenario B: Stage 1 approves, Stage 2 rejects
    result_b = get_pipeline_outcome("approve", "reject")
    assert result_b.get("final_status") == "Rejected - Stage 2", (
        "When Stage 2 rejects, final_status should be 'Rejected - Stage 2'. "
        f"Got: {result_b.get('final_status')!r}"
    )
    targets_b = set(result_b.get("notification_targets", []))
    assert "requestor" in targets_b, \
        "Rejection at Stage 2: requestor must be notified"
    assert "stage1_approver" in targets_b, \
        "Rejection at Stage 2: stage1_approver must be notified (their approval is on record)"
    assert "stage2_approver" in targets_b, \
        "Rejection at Stage 2: stage2_approver must be notified"
    assert "finance_team" not in targets_b, \
        "Rejection at Stage 2: finance_team should not be notified on rejection"

    # Scenario C: Both stages approve
    result_c = get_pipeline_outcome("approve", "approve")
    assert result_c.get("final_status") == "Fully Approved", (
        "When both stages approve, final_status should be 'Fully Approved'. "
        f"Got: {result_c.get('final_status')!r}"
    )
    targets_c = set(result_c.get("notification_targets", []))
    assert "requestor" in targets_c, \
        "Full approval: requestor must be notified"
    assert "stage1_approver" in targets_c, \
        "Full approval: stage1_approver must be notified"
    assert "stage2_approver" in targets_c, \
        "Full approval: stage2_approver must be notified"
    assert "finance_team" in targets_c, \
        "Full approval: finance_team must be notified so they can process the approved request"

    print("[PASS] Exercise 4: Pipeline outcome logic is correct for all three scenarios")


# =============================================================================
# Exercise 5: SLA Calculation
# =============================================================================
#
# Given an approval's sent timestamp and response timestamp,
# calculate whether the SLA was breached and by how much.
#
# SLA rule: approvers must respond within 48 hours (2880 minutes).

from datetime import datetime, timezone


def calculate_sla_result(
    sent_at: str,
    responded_at: str,
    sla_minutes: int = 2880
) -> dict[str, Any]:
    """
    Calculate SLA compliance for an approval response.

    Parameters
    ----------
    sent_at : str
        ISO 8601 UTC timestamp when the approval was sent,
        e.g. "2024-03-15T09:00:00Z"
    responded_at : str
        ISO 8601 UTC timestamp when the approver responded,
        e.g. "2024-03-16T11:30:00Z"
    sla_minutes : int
        SLA threshold in minutes. Default 2880 (48 hours).

    Returns
    -------
    dict with keys:
        response_time_minutes : int
            How many minutes elapsed between sent and responded (rounded down)
        sla_breached : bool
            True if response_time_minutes > sla_minutes
        breach_by_minutes : int
            How many minutes over the SLA (0 if not breached)

    Example
    -------
    >>> result = calculate_sla_result(
    ...     "2024-03-15T09:00:00Z",
    ...     "2024-03-16T11:30:00Z"
    ... )
    >>> result["response_time_minutes"]
    1590  # 26.5 hours = 1590 minutes
    >>> result["sla_breached"]
    False
    """
    # TODO: Parse the timestamps, calculate elapsed minutes, and return the dict
    # Hint: use datetime.fromisoformat() after replacing trailing 'Z' with '+00:00'
    return {}


# Self-check tests for Exercise 5
def test_exercise_5():
    """Verify SLA calculation for within-SLA and breached cases."""

    # Case 1: Response within SLA (26.5 hours = 1590 minutes, SLA = 2880)
    result1 = calculate_sla_result(
        "2024-03-15T09:00:00Z",
        "2024-03-16T11:30:00Z"
    )
    assert isinstance(result1, dict), \
        "Return value must be a dict with response_time_minutes, sla_breached, breach_by_minutes"

    assert result1.get("response_time_minutes") == 1590, (
        "26.5 hours = 1590 minutes. "
        f"Got: {result1.get('response_time_minutes')}"
    )
    assert result1.get("sla_breached") is False, (
        "1590 minutes < 2880 minutes SLA — should not be breached. "
        f"Got: {result1.get('sla_breached')}"
    )
    assert result1.get("breach_by_minutes") == 0, (
        "No breach — breach_by_minutes should be 0. "
        f"Got: {result1.get('breach_by_minutes')}"
    )

    # Case 2: Response breaches SLA (55 hours = 3300 minutes, SLA = 2880)
    result2 = calculate_sla_result(
        "2024-03-15T09:00:00Z",
        "2024-03-17T16:00:00Z"
    )
    assert result2.get("response_time_minutes") == 3300, (
        "55 hours = 3300 minutes. "
        f"Got: {result2.get('response_time_minutes')}"
    )
    assert result2.get("sla_breached") is True, (
        "3300 minutes > 2880 minutes SLA — should be breached. "
        f"Got: {result2.get('sla_breached')}"
    )
    assert result2.get("breach_by_minutes") == 420, (
        "3300 - 2880 = 420 minutes over SLA. "
        f"Got: {result2.get('breach_by_minutes')}"
    )

    # Case 3: Custom SLA of 60 minutes, response in 45 minutes
    result3 = calculate_sla_result(
        "2024-03-15T14:00:00Z",
        "2024-03-15T14:45:00Z",
        sla_minutes=60
    )
    assert result3.get("response_time_minutes") == 45
    assert result3.get("sla_breached") is False
    assert result3.get("breach_by_minutes") == 0

    print("[PASS] Exercise 5: SLA calculation is correct for all test cases")


# =============================================================================
# Run all exercises
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Module 06: Approval Flows — Self-Check Exercises")
    print("=" * 60)
    print()

    exercises = [
        ("Exercise 1: Approval type selection", test_exercise_1),
        ("Exercise 2: Purchase order card design", test_exercise_2),
        ("Exercise 3: Component mapping", test_exercise_3),
        ("Exercise 4: Pipeline outcome logic", test_exercise_4),
        ("Exercise 5: SLA calculation", test_exercise_5),
    ]

    passed = 0
    failed = 0

    for name, test_fn in exercises:
        print(f"--- {name} ---")
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] Unexpected error: {type(e).__name__}: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed}/{passed + failed} exercises passed")
    if failed == 0:
        print("All exercises complete. You are ready for Module 07.")
    else:
        print(f"{failed} exercise(s) need attention. Review the error messages above.")
    print("=" * 60)
