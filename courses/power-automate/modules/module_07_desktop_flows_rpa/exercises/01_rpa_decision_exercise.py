"""
Module 07 — Exercise 01: RPA Decision Framework
================================================

Self-check exercise. No submission required.

Work through the 10 scenarios below. For each scenario:
  1. Read the description.
  2. Fill in your answers in the ANSWER section (replace the None values).
  3. Run the file to check your answers immediately.

Each scenario tests one of these three skills:
  A. Choosing the right flow type (cloud, desktop, or hybrid)
  B. Matching an automation task to the correct PAD action group
  C. Identifying attended vs unattended execution mode

Run with:
    python 01_rpa_decision_exercise.py

All feedback prints to the console.
"""

from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------

FlowType = Literal["cloud", "desktop", "hybrid"]
ActionGroup = Literal["ui_automation", "web", "excel", "file", "email", "system"]
ExecutionMode = Literal["attended", "unattended"]


# ---------------------------------------------------------------------------
# PART A: Flow Type Selection
# Choose whether each scenario calls for a cloud flow, desktop flow, or hybrid.
#
# Definitions:
#   cloud    — entire automation runs in the Microsoft cloud via connectors
#   desktop  — automation requires interacting with a local Windows app or UI
#   hybrid   — cloud flow orchestrates a desktop flow sub-component
# ---------------------------------------------------------------------------

@dataclass
class FlowTypeScenario:
    scenario_id: str
    description: str
    your_answer: FlowType  # Replace None with "cloud", "desktop", or "hybrid"
    correct_answer: FlowType
    explanation: str


FLOW_TYPE_SCENARIOS: list[FlowTypeScenario] = [
    FlowTypeScenario(
        scenario_id="A1",
        description=(
            "When a new row is added to a SharePoint list, automatically send a "
            "Teams message to the #operations channel with the row's details."
        ),
        your_answer=None,  # TODO: Replace with "cloud", "desktop", or "hybrid"
        correct_answer="cloud",
        explanation=(
            "SharePoint and Teams both have Power Automate connectors. "
            "No local application interaction is needed. A single Automated cloud flow "
            "handles the trigger (SharePoint) and the action (Teams) entirely in the cloud."
        ),
    ),
    FlowTypeScenario(
        scenario_id="A2",
        description=(
            "Every weekday at 7 AM, open a locally-installed inventory management "
            "application (a Windows thick-client with no REST API), extract the "
            "day's low-stock items from its reports screen, and save them to a CSV file."
        ),
        your_answer=None,  # TODO: Replace with "cloud", "desktop", or "hybrid"
        correct_answer="desktop",
        explanation=(
            "The target is a thick-client Windows application with no API. "
            "Only desktop flows (via PAD UI Automation actions) can interact with it. "
            "The schedule trigger would be a cloud flow calling the desktop flow, "
            "making this technically hybrid — but the core value is the desktop interaction. "
            "If the intent is purely local execution triggered manually or by schedule, "
            "a standalone desktop flow with a scheduled cloud trigger qualifies as hybrid."
            # Note to grader: "hybrid" is also accepted for A2 — see check_answers().
        ),
    ),
    FlowTypeScenario(
        scenario_id="A3",
        description=(
            "A customer submits a form in Microsoft Forms. A cloud flow extracts the "
            "responses, then calls a desktop flow to enter the customer data into a "
            "legacy CRM application. After the desktop flow confirms the entry, "
            "the cloud flow sends a confirmation email to the customer."
        ),
        your_answer=None,  # TODO: Replace with "cloud", "desktop", or "hybrid"
        correct_answer="hybrid",
        explanation=(
            "The trigger (Forms) and follow-up action (email) are cloud connectors. "
            "The CRM entry requires a desktop flow because the CRM has no API. "
            "Cloud flow orchestrates the whole process; desktop flow handles the UI step. "
            "This is the canonical hybrid pattern."
        ),
    ),
    FlowTypeScenario(
        scenario_id="A4",
        description=(
            "An operations analyst clicks a button in a Power Apps mobile app. "
            "This triggers a workflow that creates a Dataverse record, sends a "
            "Teams notification, and updates a SharePoint list."
        ),
        your_answer=None,  # TODO: Replace with "cloud", "desktop", or "hybrid"
        correct_answer="cloud",
        explanation=(
            "Power Apps, Dataverse, Teams, and SharePoint all have connectors. "
            "An Instant cloud flow triggered by the Power Apps button handles all steps "
            "entirely in the cloud. No local machine interaction is required."
        ),
    ),
]


# ---------------------------------------------------------------------------
# PART B: PAD Action Group Matching
# Match each automation task to the correct Power Automate Desktop action group.
#
# Action groups:
#   ui_automation — interact with Windows desktop application elements
#   web           — control a web browser (Edge or Chrome) by element
#   excel         — read/write Excel workbooks via COM automation
#   file          — file and folder operations on the local file system
#   email         — send or retrieve email via Outlook or SMTP/IMAP
#   system        — run applications, read environment variables, manage processes
# ---------------------------------------------------------------------------

@dataclass
class ActionGroupScenario:
    scenario_id: str
    task: str
    your_answer: ActionGroup  # Replace None with the action group name
    correct_answer: ActionGroup
    explanation: str


ACTION_GROUP_SCENARIOS: list[ActionGroupScenario] = [
    ActionGroupScenario(
        scenario_id="B1",
        task=(
            "Click the 'Submit Order' button inside a locally-installed "
            "order management application built in Delphi."
        ),
        your_answer=None,  # TODO: "ui_automation", "web", "excel", "file", "email", or "system"
        correct_answer="ui_automation",
        explanation=(
            "Delphi is a thick-client Windows application. "
            "PAD's UI Automation actions interact with Windows application elements "
            "via the UI Automation accessibility tree. "
            "The Web action group is for browser content only."
        ),
    ),
    ActionGroupScenario(
        scenario_id="B2",
        task=(
            "Read all values from the range A1:D100 in a locally-saved "
            "Excel workbook named 'Q1_Sales.xlsx' and store them as a data table."
        ),
        your_answer=None,  # TODO: Replace
        correct_answer="excel",
        explanation=(
            "PAD's Excel action group reads and writes Excel workbooks directly "
            "via COM automation — no browser or UI screenshot needed. "
            "The 'Read from Excel worksheet' action with 'All values from range' "
            "returns a data table variable."
        ),
    ),
    ActionGroupScenario(
        scenario_id="B3",
        task=(
            "Fill in the username and password fields on a corporate login page "
            "in Microsoft Edge, then click the Sign In button."
        ),
        your_answer=None,  # TODO: Replace
        correct_answer="web",
        explanation=(
            "This targets a web page rendered in a browser. "
            "PAD's Web action group ('Fill text field on web page', "
            "'Click link on web page') uses CSS/XPath selectors to interact "
            "with browser DOM elements. UI Automation targets Windows app controls, "
            "not browser content."
        ),
    ),
    ActionGroupScenario(
        scenario_id="B4",
        task=(
            "Get a list of all PDF files in the folder C:\\Reports\\Monthly "
            "that were modified today, then move them to C:\\Reports\\Archive."
        ),
        your_answer=None,  # TODO: Replace
        correct_answer="file",
        explanation=(
            "File and folder operations (list files, filter by date, move) "
            "are in PAD's File action group. "
            "'Get files in folder' with a date filter returns a list; "
            "'Move file' processes each item in the list."
        ),
    ),
    ActionGroupScenario(
        scenario_id="B5",
        task=(
            "Send an email with the subject 'Daily Report Complete' and attach "
            "the file C:\\Reports\\daily_summary.xlsx using the locally-installed "
            "Microsoft Outlook desktop application."
        ),
        your_answer=None,  # TODO: Replace
        correct_answer="email",
        explanation=(
            "PAD's Email action group includes an Outlook subgroup that uses "
            "the locally-installed Outlook desktop application to send messages "
            "with attachments. This is distinct from the Office 365 Outlook connector "
            "in cloud flows, which calls the Graph API."
        ),
    ),
    ActionGroupScenario(
        scenario_id="B6",
        task=(
            "Launch the executable 'InvoiceProcessor.exe' located at "
            "C:\\Applications\\InvoiceProcessor.exe and wait for it to finish loading "
            "before proceeding to the next step."
        ),
        your_answer=None,  # TODO: Replace
        correct_answer="system",
        explanation=(
            "PAD's System action group includes 'Run application' to launch "
            "an executable and 'Wait for process' to pause until a process is running. "
            "This is the correct group for OS-level operations like launching applications."
        ),
    ),
]


# ---------------------------------------------------------------------------
# PART C: Attended vs Unattended
# For each scenario, determine the correct execution mode.
#
# Attended:   user must be logged in; PAD must be running; flow visible on screen
# Unattended: no user required; service account; runs in background; overnight capable
# ---------------------------------------------------------------------------

@dataclass
class ExecutionModeScenario:
    scenario_id: str
    description: str
    your_answer: ExecutionMode  # Replace None with "attended" or "unattended"
    correct_answer: ExecutionMode
    explanation: str


EXECUTION_MODE_SCENARIOS: list[ExecutionModeScenario] = [
    ExecutionModeScenario(
        scenario_id="C1",
        description=(
            "A finance analyst triggers a desktop flow from a cloud flow to look up "
            "an invoice in the legacy billing system. The flow runs during business hours "
            "on the analyst's own Windows laptop. Volume: 20 lookups per day."
        ),
        your_answer=None,  # TODO: "attended" or "unattended"
        correct_answer="attended",
        explanation=(
            "The analyst is present and the machine is their own laptop. "
            "Attended mode is appropriate: lower cost (per-user license), "
            "no service account required, and the analyst can intervene "
            "if the flow encounters an unexpected dialog."
        ),
    ),
    ExecutionModeScenario(
        scenario_id="C2",
        description=(
            "A cloud flow runs every night at 2 AM to process 800 employee "
            "timesheet records by entering each one into a legacy HR system. "
            "No employees are present in the office at 2 AM. "
            "The process must complete before the 6 AM payroll run."
        ),
        your_answer=None,  # TODO: "attended" or "unattended"
        correct_answer="unattended",
        explanation=(
            "No user is present at 2 AM, so Attended mode cannot work. "
            "Unattended mode runs the desktop flow in a background session using "
            "a service account. An unattended RPA add-on license per machine is required. "
            "The service account must have rights to the legacy HR system."
        ),
    ),
    ExecutionModeScenario(
        scenario_id="C3",
        description=(
            "During a live client meeting, a sales manager clicks a Power Apps button "
            "that triggers a desktop flow to look up the client's account history "
            "in the company's on-premises CRM. The result is displayed in Power Apps. "
            "The manager's laptop must be unlocked and PAD must be running."
        ),
        your_answer=None,  # TODO: "attended" or "unattended"
        correct_answer="attended",
        explanation=(
            "The sales manager is present and the lookup is user-initiated in real time. "
            "Attended mode is the only option here — the manager's session provides "
            "access to the on-premises CRM with their own credentials. "
            "Unattended would require a service account with CRM access and would not "
            "benefit from the always-present user."
        ),
    ),
    ExecutionModeScenario(
        scenario_id="C4",
        description=(
            "A dedicated Windows Server VM processes purchase order confirmations "
            "from an email inbox by extracting data and entering it into an ERP system. "
            "The VM runs 24/7 with no interactive user session. "
            "Volume: up to 300 orders per hour during peak periods."
        ),
        your_answer=None,  # TODO: "attended" or "unattended"
        correct_answer="unattended",
        explanation=(
            "A server VM with no interactive user session requires Unattended mode. "
            "At 300 orders per hour, a machine group with multiple machines "
            "and parallel unattended runs may be needed depending on processing time. "
            "The service account must have rights to the email inbox and ERP system."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Answer checking logic
# ---------------------------------------------------------------------------

def check_answers() -> None:
    """Run all scenarios and print pass/fail feedback with explanations."""
    total = 0
    passed = 0

    print("=" * 70)
    print("MODULE 07 — EXERCISE 01: RPA Decision Framework Self-Check")
    print("=" * 70)

    # Part A: Flow Type
    print("\nPART A: Flow Type Selection")
    print("-" * 70)

    for s in FLOW_TYPE_SCENARIOS:
        total += 1
        if s.your_answer is None:
            print(f"  [{s.scenario_id}] NOT ANSWERED — fill in your_answer for this scenario.")
            continue

        # A2 accepts both "desktop" and "hybrid" as valid answers
        # because a scheduled desktop flow is technically triggered by a cloud flow.
        valid_answers = {s.correct_answer}
        if s.scenario_id == "A2":
            valid_answers = {"desktop", "hybrid"}

        if s.your_answer.lower() in valid_answers:
            passed += 1
            print(f"  [{s.scenario_id}] CORRECT ({s.your_answer})")
            print(f"         {s.description[:60]}...")
        else:
            print(f"  [{s.scenario_id}] INCORRECT")
            print(f"         Scenario: {s.description[:60]}...")
            print(f"         Your answer:    {s.your_answer}")
            print(f"         Correct answer: {s.correct_answer}")
            print(f"         Why: {s.explanation}")
        print()

    # Part B: Action Group
    print("PART B: PAD Action Group Matching")
    print("-" * 70)

    for s in ACTION_GROUP_SCENARIOS:
        total += 1
        if s.your_answer is None:
            print(f"  [{s.scenario_id}] NOT ANSWERED — fill in your_answer for this scenario.")
            continue

        if s.your_answer.lower() == s.correct_answer:
            passed += 1
            print(f"  [{s.scenario_id}] CORRECT ({s.your_answer})")
            print(f"         {s.task[:60]}...")
        else:
            print(f"  [{s.scenario_id}] INCORRECT")
            print(f"         Task: {s.task[:60]}...")
            print(f"         Your answer:    {s.your_answer}")
            print(f"         Correct answer: {s.correct_answer}")
            print(f"         Why: {s.explanation}")
        print()

    # Part C: Execution Mode
    print("PART C: Attended vs Unattended")
    print("-" * 70)

    for s in EXECUTION_MODE_SCENARIOS:
        total += 1
        if s.your_answer is None:
            print(f"  [{s.scenario_id}] NOT ANSWERED — fill in your_answer for this scenario.")
            continue

        if s.your_answer.lower() == s.correct_answer:
            passed += 1
            print(f"  [{s.scenario_id}] CORRECT ({s.your_answer})")
            print(f"         {s.description[:60]}...")
        else:
            print(f"  [{s.scenario_id}] INCORRECT")
            print(f"         Scenario: {s.description[:60]}...")
            print(f"         Your answer:    {s.your_answer}")
            print(f"         Correct answer: {s.correct_answer}")
            print(f"         Why: {s.explanation}")
        print()

    # Score summary
    answered = sum(
        1
        for collection in [FLOW_TYPE_SCENARIOS, ACTION_GROUP_SCENARIOS, EXECUTION_MODE_SCENARIOS]
        for s in collection
        if s.your_answer is not None
    )

    print("=" * 70)
    print(f"SCORE: {passed}/{answered} answered correctly ({total} total scenarios)")
    if answered < total:
        print(f"       {total - answered} scenario(s) not yet answered.")
    if passed == answered == total:
        print("All scenarios answered correctly. Well done.")
    elif passed >= answered * 0.8:
        print("Strong result. Review the incorrect scenarios and their explanations.")
    else:
        print(
            "Review Guides 01 and 02, then revisit the incorrect scenarios. "
            "Focus on the explanations — understanding the 'why' matters more than memorizing."
        )
    print("=" * 70)


# ---------------------------------------------------------------------------
# BONUS: Design challenge (no automated check — read and reflect)
# ---------------------------------------------------------------------------

BONUS_SCENARIO = """
BONUS DESIGN CHALLENGE (no automated check):

Scenario: A mid-size manufacturer receives 50-100 PDF purchase orders per day
via email. Each PDF is unstructured (different formats from different suppliers).
The ERP system where orders must be entered is a 15-year-old Windows desktop
application with no API. The operations team currently has one person spending
4 hours/day on this manual entry.

Design the automation architecture. Consider:

1. Which components handle the email → PDF extraction?
   (Hint: does AI Builder or Power Automate's email connector help here?)

2. Which component reads the PDF content?
   (Hint: can this be a cloud action or does it require a desktop flow?)

3. How does order data get into the Windows ERP application?
   (Hint: which PAD action group?)

4. Should this run attended or unattended? Why?

5. What error handling is needed if:
   a. The PDF is unreadable or in an unexpected format?
   b. The ERP application shows an error dialog?
   c. A supplier sends a duplicate order?

6. How would you use a machine group if volume spikes to 300 orders/day?

Write your design in comments below this docstring and discuss with a colleague
or in the course community forum.
"""

print(BONUS_SCENARIO)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    check_answers()
