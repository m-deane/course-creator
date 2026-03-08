"""
Module 00 Self-Check: Power Automate Platform Fundamentals
==========================================================

This is an ungraded self-check exercise. Run the script, answer each question
by setting the variable to your chosen option, then run again to see results.

How to use:
    1. Read each question carefully
    2. Set the answer variable to the string that matches your choice
       (e.g., answer_1 = "B")
    3. Run the script: python 01_platform_quiz.py
    4. Read the explanations for any questions you got wrong
    5. Review the corresponding guide section before moving on

There are 15 questions covering:
    - Power Platform components
    - Core terminology (flow, trigger, action, connector, environment)
    - Flow types and when to use each
    - Connector tiers and licensing
    - Portal navigation
"""

# ============================================================
# ANSWER BLOCK — Fill in your answers here before running
# Set each variable to the letter of your chosen answer: "A", "B", "C", or "D"
# ============================================================

answer_1 = ""   # Which product in the Power Platform is used to AUTOMATE workflows?
answer_2 = ""   # What is a "connector" in Power Automate?
answer_3 = ""   # A flow that runs every Monday at 9 AM is which flow type?
answer_4 = ""   # Which is NOT a valid trigger category?
answer_5 = ""   # What is the minimum number of triggers a flow can have?
answer_6 = ""   # A Diamond icon on a connector tile indicates...
answer_7 = ""   # You need to automate a legacy Windows app with no API. Which flow type?
answer_8 = ""   # An "environment" in Power Automate is best described as...
answer_9 = ""   # Where in the portal do you go first when a flow stops working?
answer_10 = ""  # SharePoint, Teams, and Outlook connectors are in which tier?
answer_11 = ""  # What does the SQL Server connector require (beyond a Microsoft 365 license)?
answer_12 = ""  # A "Business Process Flow" requires which service to be enabled?
answer_13 = ""  # Which portal section is the CORRECT way to promote flows to production?
answer_14 = ""  # A connection has "Error" status. What does this mean?
answer_15 = ""  # What is the shared data layer across all Power Platform products?

# ============================================================
# QUIZ ENGINE — Do not modify below this line
# ============================================================

QUESTIONS = [
    {
        "number": 1,
        "question": "Which product in the Microsoft Power Platform is the primary tool for AUTOMATING workflows and processes?",
        "options": {
            "A": "Power Apps",
            "B": "Power Automate",
            "C": "Power BI",
            "D": "Copilot Studio",
        },
        "answer": "B",
        "explanation": (
            "Power Automate is the automation layer of the Power Platform. "
            "Power Apps builds custom applications, Power BI visualizes data, "
            "and Copilot Studio creates conversational AI agents. "
            "Power Automate's job is to connect services and automate the steps between them."
        ),
        "guide_ref": "Guide 01, Section: Where Power Automate Fits",
    },
    {
        "number": 2,
        "question": "What is a 'connector' in Power Automate?",
        "options": {
            "A": "A saved username/password combination for an external service",
            "B": "A prebuilt adapter that wraps an external service's API and exposes its triggers and actions",
            "C": "A type of flow that runs automatically",
            "D": "The link between two actions inside a flow",
        },
        "answer": "B",
        "explanation": (
            "A connector is a prebuilt wrapper around an external API. It handles authentication, "
            "rate limiting, and API versioning, and exposes named triggers and actions to the flow designer. "
            "Option A describes a 'connection' (an authenticated instance of a connector), not the connector itself. "
            "Connectors and connections are distinct: a connector is the template, a connection is the credential."
        ),
        "guide_ref": "Guide 01, Section: Core Terminology — Connector",
    },
    {
        "number": 3,
        "question": "A flow that runs automatically every Monday at 9 AM is which flow type?",
        "options": {
            "A": "Automated cloud flow",
            "B": "Instant cloud flow",
            "C": "Scheduled cloud flow",
            "D": "Business process flow",
        },
        "answer": "C",
        "explanation": (
            "Scheduled cloud flows are triggered on a time-based recurrence — you define the frequency, "
            "start time, and time zone. Automated flows respond to events (something happening in a connected service). "
            "Instant flows are triggered manually by a user. Business process flows are human-guided stage-based workflows."
        ),
        "guide_ref": "Guide 01, Section: Flow Types — Scheduled Cloud Flow",
    },
    {
        "number": 4,
        "question": "Which of the following is NOT a valid trigger category in Power Automate?",
        "options": {
            "A": "Event trigger (fires when something happens)",
            "B": "Schedule trigger (fires on a timer)",
            "C": "Condition trigger (fires when a formula evaluates to true)",
            "D": "Manual trigger (fires when a user presses a button)",
        },
        "answer": "C",
        "explanation": (
            "There is no 'condition trigger' in Power Automate. Triggers are either event-based "
            "(something happens in a connected service), schedule-based (timer), or manual (user action). "
            "Conditions are control flow structures that run AFTER a trigger has fired — they branch "
            "the flow's logic but do not start execution."
        ),
        "guide_ref": "Guide 01, Section: Core Terminology — Trigger",
    },
    {
        "number": 5,
        "question": "What is the minimum and maximum number of triggers a single flow can have?",
        "options": {
            "A": "Minimum 0, maximum unlimited",
            "B": "Minimum 1, maximum unlimited",
            "C": "Minimum 1, maximum 1",
            "D": "Minimum 1, maximum 5",
        },
        "answer": "C",
        "explanation": (
            "Every flow has exactly one trigger — no more, no fewer. This is a fundamental constraint "
            "of the Power Automate execution model. If you need a flow to respond to multiple events, "
            "you create multiple flows (one per trigger) or use a parent flow that child flows call. "
            "Importantly: you cannot change a flow's trigger type after creation."
        ),
        "guide_ref": "Guide 01, Section: Core Terminology — Trigger",
    },
    {
        "number": 6,
        "question": "In the connector gallery at Data > Connectors, a diamond icon on a connector tile indicates...",
        "options": {
            "A": "The connector is certified by Microsoft",
            "B": "The connector is a Premium connector requiring an upgraded Power Automate license",
            "C": "The connector is deprecated and will be removed",
            "D": "The connector requires admin approval before use",
        },
        "answer": "B",
        "explanation": (
            "The diamond icon marks Premium connectors, which require a Power Automate Per User or "
            "Per Flow plan (not just a Microsoft 365 license). Common premium connectors include "
            "SQL Server, Salesforce, HTTP (custom REST calls), Dataverse, and ServiceNow. "
            "Attempting to use a Premium connector with only a Standard license prompts an upgrade notice."
        ),
        "guide_ref": "Guide 01, Section: Core Terminology — Connector (Connector Tiers table)",
    },
    {
        "number": 7,
        "question": "You need to automate data entry into a legacy Windows application that has no web API or modern integration. Which Power Automate flow type should you use?",
        "options": {
            "A": "Automated cloud flow",
            "B": "Scheduled cloud flow",
            "C": "Desktop flow (RPA)",
            "D": "Business process flow",
        },
        "answer": "C",
        "explanation": (
            "Desktop flows use Robotic Process Automation (RPA) to interact with desktop applications "
            "and websites by simulating mouse clicks and keystrokes — no API required. "
            "Cloud flows (Automated, Instant, Scheduled) can only reach services that have connectors "
            "or REST APIs. If the target system has no API, Desktop flow is the only option. "
            "Power Automate Desktop must be installed on the machine running the automation."
        ),
        "guide_ref": "Guide 01, Section: Flow Types — Desktop Flow (RPA)",
    },
    {
        "number": 8,
        "question": "An 'environment' in Power Automate is best described as...",
        "options": {
            "A": "The set of connectors available for use in flows",
            "B": "An isolated container that holds flows, apps, connections, and data, serving as a security and governance boundary",
            "C": "The browser tab where the Power Automate portal is open",
            "D": "A group of users who can access the same flows",
        },
        "answer": "B",
        "explanation": (
            "Environments are isolated administrative containers — the primary security and governance boundary "
            "in Power Automate. Each environment has its own set of flows, apps, connections, and optionally "
            "a Dataverse database. Flows in one environment cannot directly access flows or data in another. "
            "Best practice: use separate environments for Development, Test, and Production. "
            "The Default environment is auto-created and available to all licensed users."
        ),
        "guide_ref": "Guide 01, Section: Core Terminology — Environment",
    },
    {
        "number": 9,
        "question": "A flow that was working for months suddenly starts failing. What is the FIRST place to check in the Power Automate portal?",
        "options": {
            "A": "Home page — to see if there are any outage announcements",
            "B": "Data > Connections — to check if any connection has Error status (expired credentials)",
            "C": "Solutions — to see if the flow was accidentally moved",
            "D": "My Flows — to verify the flow is still turned on",
        },
        "answer": "B",
        "explanation": (
            "The most common cause of a previously-working flow suddenly failing is an expired or "
            "broken connection. OAuth tokens expire (often every 60-90 days for Microsoft connections, "
            "shorter for some third-party ones). Service account passwords change. API keys rotate. "
            "When this happens, all flows using that connection fail. Check Data > Connections first — "
            "a red 'Error' status on a connection is the immediate signal. Fix: click the connection and re-authenticate."
        ),
        "guide_ref": "Guide 02, Section: Data > Connections",
    },
    {
        "number": 10,
        "question": "SharePoint, Microsoft Teams, Outlook, and OneDrive connectors belong to which connector tier?",
        "options": {
            "A": "Premium — they require a Power Automate Per User license",
            "B": "Custom — they were built by Microsoft using the Custom Connector framework",
            "C": "Standard — they are included with most Microsoft 365 subscriptions",
            "D": "Enterprise — they require a Microsoft 365 E3 or higher license",
        },
        "answer": "C",
        "explanation": (
            "SharePoint, Teams, Outlook, OneDrive, Excel Online, Microsoft Forms, and most other "
            "core Microsoft 365 services are Standard connectors. They are included in most Microsoft 365 "
            "subscription plans and do not require an additional Power Automate license beyond what M365 includes. "
            "Premium connectors (SQL Server, Salesforce, HTTP, Dataverse, ServiceNow) require Per User or Per Flow plans."
        ),
        "guide_ref": "Guide 01, Section: Core Terminology — Connector Tiers table",
    },
    {
        "number": 11,
        "question": "A flow maker wants to query a SQL Server database from a Power Automate cloud flow. What license is required beyond a standard Microsoft 365 plan?",
        "options": {
            "A": "No additional license — SQL Server is a Standard connector",
            "B": "A Power Automate Per User or Per Flow plan — SQL Server is a Premium connector",
            "C": "A Microsoft Azure subscription — SQL Server is an Azure-only connector",
            "D": "A Power Apps license — SQL Server is only accessible through Power Apps",
        },
        "answer": "B",
        "explanation": (
            "The SQL Server connector is Premium. Using it requires a Power Automate Per User plan "
            "(approx. $15/user/month) or a Per Flow plan (approx. $100/flow/month). "
            "The M365 bundled Power Automate access only includes Standard connectors. "
            "The HTTP connector is also Premium — so any custom REST API call has the same licensing requirement. "
            "Always check the diamond icon in the connector gallery before designing a flow for a budget-constrained organization."
        ),
        "guide_ref": "Guide 02, Section: Licensing Tiers",
    },
    {
        "number": 12,
        "question": "A 'Business Process Flow' in Power Automate requires which service to be enabled in the environment?",
        "options": {
            "A": "Azure Logic Apps",
            "B": "Microsoft Dataverse",
            "C": "SharePoint Online",
            "D": "Power Apps Premium",
        },
        "answer": "B",
        "explanation": (
            "Business Process Flows are a Dataverse-native feature. They create a stage-based UI overlay "
            "on model-driven Power Apps forms and are stored as Dataverse entities. "
            "An environment must have a Dataverse database provisioned to use Business Process Flows. "
            "The Default environment has Dataverse, but custom environments must be created with Dataverse enabled. "
            "This is why the portal shows 'requires a Dataverse environment' when you select this flow type."
        ),
        "guide_ref": "Guide 01, Section: Flow Types — Business Process Flow",
    },
    {
        "number": 13,
        "question": "What is the RECOMMENDED mechanism for promoting flows from a Development environment to Production?",
        "options": {
            "A": "Recreate the flow manually in the Production environment",
            "B": "Use 'Save as' to clone the flow and share it with the production environment",
            "C": "Export the flow as a managed Solution and import it into the Production environment",
            "D": "Copy the flow's JSON definition and paste it in the Production environment's flow designer",
        },
        "answer": "C",
        "explanation": (
            "Solutions are the Microsoft-recommended ALM (Application Lifecycle Management) mechanism for "
            "deploying flows across environments. Build inside an unmanaged solution in Development, "
            "export as a managed solution, import into Test, validate, then promote to Production. "
            "Managed solutions are read-only in the target environment — protecting production from accidental edits. "
            "Manual recreation (A) is error-prone and does not scale. 'Save as' (B) does not cross environments. "
            "Copying JSON (D) is fragile and loses connection references."
        ),
        "guide_ref": "Guide 02, Section: Solutions",
    },
    {
        "number": 14,
        "question": "In the Data > Connections page, a connection shows 'Error' status (red). What does this indicate?",
        "options": {
            "A": "The connector for this connection has been deprecated by Microsoft",
            "B": "The credentials for this connection have expired or become invalid, causing all flows using it to fail",
            "C": "The connection was accidentally deleted and needs to be recreated from scratch",
            "D": "The connected service (e.g., SharePoint) is currently experiencing an outage",
        },
        "answer": "B",
        "explanation": (
            "Error status on a connection means the stored credentials are no longer valid — "
            "the OAuth token expired, a password changed, or an API key was rotated. "
            "Every flow that uses this connection will fail until the connection is re-authenticated. "
            "Fix: click the connection row, then click the 'Fix connection' or 'Edit' button and sign in again. "
            "The connection itself does not need to be deleted and recreated — just re-authenticated. "
            "Service outages produce different error patterns (connection is 'Connected' but flow actions fail)."
        ),
        "guide_ref": "Guide 02, Section: Data > Connections",
    },
    {
        "number": 15,
        "question": "Which service acts as the shared data layer across Power Apps, Power Automate, Power BI, and Copilot Studio?",
        "options": {
            "A": "Azure Blob Storage",
            "B": "SharePoint Online",
            "C": "Microsoft Dataverse",
            "D": "SQL Server on Azure",
        },
        "answer": "C",
        "explanation": (
            "Microsoft Dataverse is the low-code data platform built into the Power Platform. "
            "It provides a relational schema, role-based security, audit logging, and native integration "
            "with all Power Platform products. Power Apps model-driven apps are built on Dataverse. "
            "Dataverse connectors (Premium) give Power Automate direct, governed access to this data. "
            "Power BI has native Dataverse connectors. Copilot Studio agents store conversation data in Dataverse. "
            "SharePoint and SQL Server are external systems that Power Platform connects TO — they are not the shared layer."
        ),
        "guide_ref": "Guide 01, Section: Where Power Automate Fits — The Power Platform Ecosystem",
    },
]


# ============================================================
# Scoring logic
# ============================================================

def _letter_to_answer(letter: str, question_number: int) -> str:
    """Normalize the submitted answer letter."""
    return letter.strip().upper() if letter else ""


def run_quiz() -> None:
    """Evaluate all answers and print a results report."""
    submitted = {
        1: answer_1,
        2: answer_2,
        3: answer_3,
        4: answer_4,
        5: answer_5,
        6: answer_6,
        7: answer_7,
        8: answer_8,
        9: answer_9,
        10: answer_10,
        11: answer_11,
        12: answer_12,
        13: answer_13,
        14: answer_14,
        15: answer_15,
    }

    correct_count = 0
    unanswered_count = 0
    results = []

    for q in QUESTIONS:
        n = q["number"]
        raw = submitted.get(n, "")
        given = _letter_to_answer(raw, n)
        correct = q["answer"]

        if not given:
            unanswered_count += 1
            results.append((n, "UNANSWERED", False, q))
        elif given == correct:
            correct_count += 1
            results.append((n, given, True, q))
        else:
            results.append((n, given, False, q))

    total_answered = len(QUESTIONS) - unanswered_count

    # Print report
    print("=" * 70)
    print("  MODULE 00 SELF-CHECK RESULTS")
    print("=" * 70)
    print()

    for (n, given, is_correct, q) in results:
        if given == "UNANSWERED":
            status = "[ -- ] UNANSWERED"
        elif is_correct:
            status = "[  OK  ] CORRECT"
        else:
            status = "[ MISS ] INCORRECT"

        print(f"Q{n:02d}: {status}")
        print(f"      {q['question'][:72]}")

        if not is_correct and given != "UNANSWERED":
            print(f"      You answered:  {given} — {q['options'].get(given, '(invalid option)')}")
            print(f"      Correct answer: {q['answer']} — {q['options'][q['answer']]}")

        if given == "UNANSWERED":
            print(f"      Set answer_{n} = 'A', 'B', 'C', or 'D' to answer this question.")

        if not is_correct or given == "UNANSWERED":
            print(f"      Explanation: {q['explanation']}")
            print(f"      Review: {q['guide_ref']}")

        print()

    # Summary
    print("=" * 70)
    if unanswered_count > 0:
        print(f"  Answered: {total_answered} / {len(QUESTIONS)}")
        print(f"  Unanswered: {unanswered_count} — set the answer variables and re-run.")
    else:
        pct = int(correct_count / len(QUESTIONS) * 100)
        print(f"  Score: {correct_count} / {len(QUESTIONS)}  ({pct}%)")
        print()
        if pct == 100:
            print("  All correct. You are ready for Module 01.")
        elif pct >= 80:
            print("  Strong foundation. Review the missed questions, then proceed to Module 01.")
        elif pct >= 60:
            print("  Good start. Re-read the guide sections referenced above before proceeding.")
        else:
            print("  Work through Guide 01 and Guide 02 again, focusing on the missed topics.")
            print("  Key areas to review: flow types, connector tiers, and core terminology.")
    print("=" * 70)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    run_quiz()
