"""
Module 09 — Copilot Agents: Agent Design Self-Check Exercises

These exercises build the design thinking skills needed to architect Copilot agents
before opening Copilot Studio. Given a business scenario, you will:

  1. Identify the required topics
  2. Map each topic to its backing Power Automate flow
  3. Identify security requirements
  4. Design the conversation flow for specific scenarios

Run with:
    python 01_agent_design_exercise.py

All exercises are self-grading. Each one prints PASS or FAIL with an explanation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# =============================================================================
# Data Structures Used Across All Exercises
# =============================================================================

@dataclass
class TopicDesign:
    """
    Design specification for a single Copilot agent topic.

    Attributes
    ----------
    name : str
        The topic name as it would appear in Copilot Studio.
    trigger_phrases : list[str]
        Representative phrases that should activate this topic (min 3).
    collected_variables : list[str]
        Names of variables the topic collects via Question nodes.
    backing_flow_name : str
        The name of the Power Automate flow this topic calls as an action.
    flow_input_parameters : list[str]
        Parameter names passed from the topic to the flow.
    flow_output_parameters : list[str]
        Parameter names returned from the flow to the topic.
    """
    name: str
    trigger_phrases: list[str]
    collected_variables: list[str]
    backing_flow_name: str
    flow_input_parameters: list[str]
    flow_output_parameters: list[str]


@dataclass
class AgentDesign:
    """
    Complete design specification for a Copilot agent.

    Attributes
    ----------
    agent_name : str
        Display name for the agent.
    agent_description : str
        One-sentence description of the agent's purpose.
    topics : list[TopicDesign]
        All custom topics in the agent.
    requires_authentication : bool
        True if the agent requires authenticated users (e.g., Teams channel with SSO).
    service_account_needed : bool
        True if a dedicated service account is required for flow connections.
    knowledge_sources : list[str]
        SharePoint sites or document URLs to use for Generative Answers.
    security_notes : str
        Any additional security considerations specific to this scenario.
    """
    agent_name: str
    agent_description: str
    topics: list[TopicDesign]
    requires_authentication: bool
    service_account_needed: bool
    knowledge_sources: list[str] = field(default_factory=list)
    security_notes: str = ""


# =============================================================================
# Exercise 1: HR Leave Request Agent
# =============================================================================
# Scenario:
# Contoso HR wants a Copilot agent deployed to Microsoft Teams that allows
# employees to:
#   - Check their remaining annual leave balance (read from HR system via HTTP)
#   - Submit a leave request (write to a SharePoint list for HR review)
#   - Check the status of a pending leave request
#
# The agent must know WHO is asking (for balance lookup and submission).
# All leave records contain sensitive personal data — DLP must restrict
# connectors to only HR-approved ones.
#
# Task:
# Fill in the `hr_leave_agent` variable with an AgentDesign that correctly
# models the scenario described above.
#
# Rules:
#   - Topics must have at least 3 trigger phrases each
#   - Flow input parameters must include the user's email where relevant
#   - Authentication must be correctly set
#   - security_notes must mention DLP

hr_leave_agent: AgentDesign | None = None  # Replace with your AgentDesign

# Example of a correctly structured design for a DIFFERENT scenario (expense agent):
# hr_leave_agent = AgentDesign(
#     agent_name="HR Leave Assistant",
#     agent_description="...",
#     topics=[
#         TopicDesign(
#             name="Check Leave Balance",
#             trigger_phrases=["how much leave", "check my balance", "remaining days"],
#             collected_variables=[],  # No collection needed -- uses System.User.Email
#             backing_flow_name="HR - Get Leave Balance",
#             flow_input_parameters=["EmployeeEmail"],
#             flow_output_parameters=["AnnualLeaveRemaining", "SickLeaveRemaining"],
#         ),
#         ...
#     ],
#     requires_authentication=True,
#     service_account_needed=True,
#     knowledge_sources=["https://contoso.sharepoint.com/sites/HR/Policies"],
#     security_notes="DLP must restrict connectors to SharePoint and HTTP only.",
# )


def check_exercise_1() -> None:
    """Validate the hr_leave_agent design against the scenario requirements."""
    print("=" * 60)
    print("Exercise 1: HR Leave Request Agent Design")
    print("=" * 60)

    errors = []

    if hr_leave_agent is None:
        errors.append(
            "hr_leave_agent is None. Assign an AgentDesign instance."
        )
        _report_result(errors, exercise_number=1)
        return

    if not isinstance(hr_leave_agent, AgentDesign):
        errors.append(
            f"Expected AgentDesign, got {type(hr_leave_agent).__name__}."
        )
        _report_result(errors, exercise_number=1)
        return

    # Must have at least 3 topics (balance, submit, status)
    if len(hr_leave_agent.topics) < 3:
        errors.append(
            f"Expected at least 3 topics (balance, submit, status). "
            f"Got {len(hr_leave_agent.topics)}."
        )

    # Each topic must have at least 3 trigger phrases
    for topic in hr_leave_agent.topics:
        if len(topic.trigger_phrases) < 3:
            errors.append(
                f"Topic '{topic.name}' needs at least 3 trigger phrases. "
                f"Got {len(topic.trigger_phrases)}."
            )

    # Authentication must be True — agent must know who is asking
    if not hr_leave_agent.requires_authentication:
        errors.append(
            "requires_authentication must be True. "
            "The agent needs to know WHO is asking to look up their balance."
        )

    # Service account needed — leave data is sensitive and should not use personal connections
    if not hr_leave_agent.service_account_needed:
        errors.append(
            "service_account_needed must be True. "
            "Flows accessing HR data must use a dedicated service account, "
            "not a personal connection."
        )

    # At least one topic must pass EmployeeEmail or SubmitterEmail to its flow
    email_passed = any(
        any("email" in param.lower() for param in topic.flow_input_parameters)
        for topic in hr_leave_agent.topics
    )
    if not email_passed:
        errors.append(
            "At least one topic must pass the user's email to its flow. "
            "Use System.User.Email in the topic and map it to a flow input parameter."
        )

    # Security notes must mention DLP
    if "dlp" not in hr_leave_agent.security_notes.lower():
        errors.append(
            "security_notes must mention DLP. "
            "HR data is sensitive — DLP policies restrict which connectors are allowed."
        )

    _report_result(errors, exercise_number=1)


# =============================================================================
# Exercise 2: Topic-to-Flow Parameter Mapping
# =============================================================================
# Scenario:
# You are building the "Search Knowledge Base" topic for the IT Helpdesk agent
# from Guide 02.
#
# Task:
# Fill in the `search_kb_topic` variable with a TopicDesign that correctly
# describes the Search Knowledge Base topic.
#
# Requirements:
#   - Topic collects one variable: the user's search query
#   - The flow receives that variable as an input
#   - The flow returns at least 3 output values (found flag + at least two content fields)
#   - Must have at least 5 trigger phrases (knowledge search is triggered many ways)

search_kb_topic: TopicDesign | None = None  # Replace with your TopicDesign


def check_exercise_2() -> None:
    """Validate the search_kb_topic design."""
    print("\n" + "=" * 60)
    print("Exercise 2: Search KB Topic Parameter Mapping")
    print("=" * 60)

    errors = []

    if search_kb_topic is None:
        errors.append(
            "search_kb_topic is None. Assign a TopicDesign instance."
        )
        _report_result(errors, exercise_number=2)
        return

    if not isinstance(search_kb_topic, TopicDesign):
        errors.append(
            f"Expected TopicDesign, got {type(search_kb_topic).__name__}."
        )
        _report_result(errors, exercise_number=2)
        return

    # Must collect the search query
    if len(search_kb_topic.collected_variables) < 1:
        errors.append(
            "collected_variables must include at least one variable "
            "(the user's search query). Got empty list."
        )

    # Flow input must include the collected search variable
    if search_kb_topic.collected_variables:
        collected_lower = [v.lower() for v in search_kb_topic.collected_variables]
        inputs_lower = [p.lower() for p in search_kb_topic.flow_input_parameters]
        if not any(var in " ".join(inputs_lower) for var in collected_lower):
            errors.append(
                f"flow_input_parameters should include the collected variable "
                f"'{search_kb_topic.collected_variables[0]}'. "
                f"The topic must pass what the user typed to the flow."
            )

    # Flow must return at least 3 outputs (found flag + title + one more)
    if len(search_kb_topic.flow_output_parameters) < 3:
        errors.append(
            f"flow_output_parameters needs at least 3 values: a found/not-found flag, "
            f"the article title, and at least one more (summary or URL). "
            f"Got {len(search_kb_topic.flow_output_parameters)}."
        )

    # Must have at least 5 trigger phrases
    if len(search_kb_topic.trigger_phrases) < 5:
        errors.append(
            f"Need at least 5 trigger phrases. Users say 'how do I', "
            f"'search KB', 'find an article', 'look up', and many other variations. "
            f"Got {len(search_kb_topic.trigger_phrases)}."
        )

    # Flow name should reference search or KB in some way
    flow_lower = search_kb_topic.backing_flow_name.lower()
    if not any(word in flow_lower for word in ["search", "kb", "knowledge", "article"]):
        errors.append(
            f"backing_flow_name '{search_kb_topic.backing_flow_name}' is unclear. "
            "Name the flow to describe what it does, e.g., 'Helpdesk - Search KB Articles'."
        )

    _report_result(errors, exercise_number=2)


# =============================================================================
# Exercise 3: Security Requirements Analysis
# =============================================================================
# Scenario:
# Your team is deploying a Copilot agent that:
#   - Is published to a public-facing website (unauthenticated web channel)
#   - Can answer general company FAQ questions using Generative Answers
#   - Can submit a contact form request (name, email, message) to a SharePoint list
#   - CANNOT access any internal employee data
#   - CANNOT trigger approval flows or financial operations
#
# Task:
# Fill in the `web_agent_security` dictionary with the correct values for each
# security requirement key.
#
# Keys and expected types:
#   "requires_authentication"     : bool
#   "service_account_needed"      : bool
#   "expose_employee_data"        : bool  — should internal employee data be accessible?
#   "safe_for_financial_actions"  : bool  — should financial operations be callable?
#   "dlp_connectors_allowed"      : list[str]  — which connectors should DLP permit?
#   "channels"                    : list[str]  — which publication channels?
#   "security_concern"            : str  — one-sentence description of the primary risk

web_agent_security: dict | None = None  # Replace with your dictionary


def check_exercise_3() -> None:
    """Validate the web_agent_security configuration."""
    print("\n" + "=" * 60)
    print("Exercise 3: Security Requirements for Public Web Agent")
    print("=" * 60)

    errors = []

    if web_agent_security is None:
        errors.append(
            "web_agent_security is None. Assign a dictionary with the required keys."
        )
        _report_result(errors, exercise_number=3)
        return

    required_keys = [
        "requires_authentication",
        "service_account_needed",
        "expose_employee_data",
        "safe_for_financial_actions",
        "dlp_connectors_allowed",
        "channels",
        "security_concern",
    ]

    for key in required_keys:
        if key not in web_agent_security:
            errors.append(f"Missing key: '{key}'. All keys are required.")

    if errors:
        _report_result(errors, exercise_number=3)
        return

    # Public web channel — no authentication
    if web_agent_security["requires_authentication"] is True:
        errors.append(
            "requires_authentication should be False for a public web channel. "
            "The web channel does not authenticate users."
        )

    # Service account still needed — flow writes to SharePoint
    if web_agent_security["service_account_needed"] is False:
        errors.append(
            "service_account_needed should be True. "
            "Even without user authentication, flows need a service account "
            "connection to write to SharePoint."
        )

    # Must NOT expose employee data on a public channel
    if web_agent_security["expose_employee_data"] is True:
        errors.append(
            "expose_employee_data must be False. "
            "A public unauthenticated channel must never surface internal employee data."
        )

    # Must NOT allow financial actions on a public channel
    if web_agent_security["safe_for_financial_actions"] is True:
        errors.append(
            "safe_for_financial_actions must be False. "
            "Financial operations require authenticated users with appropriate roles."
        )

    # DLP should only allow SharePoint (for contact form) — not Approvals, not Graph
    allowed = [c.lower() for c in web_agent_security.get("dlp_connectors_allowed", [])]
    if not any("sharepoint" in c for c in allowed):
        errors.append(
            "dlp_connectors_allowed should include 'SharePoint'. "
            "The contact form write operation requires the SharePoint connector."
        )
    dangerous_connectors = ["approvals", "azure ad", "graph", "dataverse"]
    for dangerous in dangerous_connectors:
        if any(dangerous in c for c in allowed):
            errors.append(
                f"dlp_connectors_allowed should NOT include '{dangerous}'. "
                "This connector provides access to internal data or sensitive operations "
                "inappropriate for a public channel."
            )

    # Channel must include web
    channels_lower = [c.lower() for c in web_agent_security.get("channels", [])]
    if not any("web" in c for c in channels_lower):
        errors.append(
            "channels must include 'Web' or 'Web channel'. "
            "The scenario specifies a public-facing website deployment."
        )

    # Security concern should be non-empty
    if not web_agent_security.get("security_concern", "").strip():
        errors.append(
            "security_concern must be a non-empty string describing the primary risk."
        )

    _report_result(errors, exercise_number=3)


# =============================================================================
# Exercise 4: Conversation Flow Design
# =============================================================================
# Scenario:
# Design the conversation flow for a "Book a Meeting Room" topic in a Facilities
# Management Copilot agent.
#
# The topic must:
#   1. Confirm whether the user wants to book a room (they might have triggered the
#      topic with an ambiguous phrase like "I need a room")
#   2. Collect: the date, the start time, the end time, and the room size preference
#      (Small: 1-4 people, Medium: 5-10 people, Large: 11+ people)
#   3. Pass all collected values + the user's email to the backing flow
#   4. Return a booking confirmation number and the room name that was assigned
#
# Task:
# Fill in `room_booking_topic` with a TopicDesign that correctly models this.
#
# Additional requirement: the collected_variables list must be in conversation ORDER
# (the order questions are asked), starting with date.

room_booking_topic: TopicDesign | None = None  # Replace with your TopicDesign


def check_exercise_4() -> None:
    """Validate the room_booking_topic design."""
    print("\n" + "=" * 60)
    print("Exercise 4: Room Booking Conversation Flow Design")
    print("=" * 60)

    errors = []

    if room_booking_topic is None:
        errors.append(
            "room_booking_topic is None. Assign a TopicDesign instance."
        )
        _report_result(errors, exercise_number=4)
        return

    if not isinstance(room_booking_topic, TopicDesign):
        errors.append(
            f"Expected TopicDesign, got {type(room_booking_topic).__name__}."
        )
        _report_result(errors, exercise_number=4)
        return

    # Must collect: date, start time, end time, room size (at minimum 4 variables)
    if len(room_booking_topic.collected_variables) < 4:
        errors.append(
            f"collected_variables needs at least 4 items: date, start time, end time, "
            f"and room size preference. Got {len(room_booking_topic.collected_variables)}."
        )
    else:
        vars_lower = [v.lower() for v in room_booking_topic.collected_variables]
        # Check for date
        if not any("date" in v for v in vars_lower):
            errors.append(
                "collected_variables must include a date variable. "
                "Name it 'bookingDate', 'meetingDate', or similar."
            )
        # Check for time (start or end)
        if not any("time" in v or "start" in v for v in vars_lower):
            errors.append(
                "collected_variables must include a start time variable. "
                "Name it 'startTime', 'meetingStart', or similar."
            )
        if not any("end" in v for v in vars_lower):
            errors.append(
                "collected_variables must include an end time variable. "
                "Name it 'endTime', 'meetingEnd', or similar."
            )
        # Check for room size
        if not any("size" in v or "room" in v or "capacity" in v for v in vars_lower):
            errors.append(
                "collected_variables must include a room size/capacity variable. "
                "Name it 'roomSize', 'capacity', or similar."
            )

        # Date must come first in the sequence
        if vars_lower and "date" not in vars_lower[0]:
            errors.append(
                f"collected_variables should start with the date. "
                f"Got first variable: '{room_booking_topic.collected_variables[0]}'. "
                "Ask for the date before times — you can't know what days are available "
                "until you know which date the user wants."
            )

    # Flow inputs must include user email + all collected variables
    inputs_lower = [p.lower() for p in room_booking_topic.flow_input_parameters]
    if not any("email" in p for p in inputs_lower):
        errors.append(
            "flow_input_parameters must include the user's email. "
            "Map System.User.Email to a parameter like 'BookerEmail'."
        )
    if len(room_booking_topic.flow_input_parameters) < 5:  # 4 collected + email
        errors.append(
            f"flow_input_parameters should have at least 5 items: "
            f"the 4 collected variables plus the user's email. "
            f"Got {len(room_booking_topic.flow_input_parameters)}."
        )

    # Flow outputs must include a confirmation number and the assigned room name
    outputs_lower = [p.lower() for p in room_booking_topic.flow_output_parameters]
    if not any("confirm" in p or "booking" in p or "reference" in p for p in outputs_lower):
        errors.append(
            "flow_output_parameters must include a booking/confirmation number. "
            "Name it 'ConfirmationNumber', 'BookingReference', or similar."
        )
    if not any("room" in p or "name" in p for p in outputs_lower):
        errors.append(
            "flow_output_parameters must include the assigned room name. "
            "Name it 'AssignedRoom', 'RoomName', or similar."
        )

    _report_result(errors, exercise_number=4)


# =============================================================================
# Helper Functions
# =============================================================================

def _report_result(errors: list[str], exercise_number: int) -> None:
    """Print pass/fail report for an exercise."""
    if not errors:
        print(f"[PASS] Exercise {exercise_number} passed.")
    else:
        print(f"[FAIL] Exercise {exercise_number} failed with {len(errors)} issue(s):")
        for i, error in enumerate(errors, start=1):
            print(f"       {i}. {error}")


# =============================================================================
# Run All Exercises
# =============================================================================

if __name__ == "__main__":
    print()
    print("Module 09 — Copilot Agent Design Self-Check")
    print("Fill in the variables above each check_ function, then run this file.")
    print()

    check_exercise_1()
    check_exercise_2()
    check_exercise_3()
    check_exercise_4()

    print()
    print("Done. Fix any FAIL messages above and re-run to verify your designs.")
