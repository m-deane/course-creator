"""
Module 02 — Exercise 01: Connector and Trigger Matching

Self-check exercise. Run the file with `python 01_connector_matching.py`
and follow the prompts. Type your answer when asked and press Enter.
The script explains the correct answer whether you get it right or wrong.

No imports required beyond the standard library.
Estimated time: 10 minutes.
"""

import sys


# ---------------------------------------------------------------------------
# Question data
# Each question is a dict with:
#   prompt      - the scenario text shown to the learner
#   options     - lettered answer choices (list of strings)
#   answer      - the correct letter (str)
#   explanation - full explanation shown after every attempt
# ---------------------------------------------------------------------------

QUESTIONS = [
    # ── Section 1: Match scenario to trigger family ────────────────────────
    {
        "section": "Section 1: Trigger Family Identification",
        "prompt": (
            "Scenario 1-A\n"
            "Your manager needs to kick off a monthly supplier payment approval\n"
            "process by clicking a button in the Power Automate mobile app.\n"
            "Which trigger FAMILY should this flow use?"
        ),
        "options": [
            "A. Automated (event-driven)",
            "B. Instant (on-demand)",
            "C. Scheduled (recurrence)",
            "D. Sliding Window",
        ],
        "answer": "B",
        "explanation": (
            "Correct answer: B — Instant trigger.\n"
            "\n"
            "An Instant trigger fires when a human explicitly initiates it —\n"
            "in this case, the manager pressing a button in the mobile app.\n"
            "Use 'Manually trigger a flow' and the Run button appears in the\n"
            "Power Automate mobile app and browser portal.\n"
            "\n"
            "Why not the others?\n"
            "  A (Automated): No external event fires this — a person does.\n"
            "  C (Scheduled): Would run on a clock regardless of manager intent.\n"
            "  D (Sliding Window): A type of scheduled trigger, same issue as C."
        ),
    },
    {
        "section": "Section 1: Trigger Family Identification",
        "prompt": (
            "Scenario 1-B\n"
            "The finance team receives invoice PDFs via email. Every time\n"
            "a new email with subject line starting with 'INVOICE:' arrives\n"
            "in the Accounts Payable shared mailbox, a flow should extract\n"
            "the attachment and save it to SharePoint.\n"
            "Which trigger FAMILY is correct?"
        ),
        "options": [
            "A. Automated (event-driven)",
            "B. Instant (on-demand)",
            "C. Scheduled (recurrence)",
            "D. None — this cannot be automated",
        ],
        "answer": "A",
        "explanation": (
            "Correct answer: A — Automated trigger.\n"
            "\n"
            "An email arriving is an external event. The 'When a new email\n"
            "arrives (V3)' trigger in the Office 365 Outlook connector is a\n"
            "WEBHOOK trigger — it fires near-instantly when the email lands.\n"
            "\n"
            "Add a Subject Filter of 'INVOICE:' in the trigger's advanced\n"
            "options (or as a trigger condition) so only invoice emails\n"
            "start the flow — not every email in the mailbox.\n"
            "\n"
            "The shared mailbox consideration: create the flow's Outlook\n"
            "connection using a service account that has access to the\n"
            "Accounts Payable mailbox, and set the Folder field to the\n"
            "shared mailbox's Inbox path."
        ),
    },
    {
        "section": "Section 1: Trigger Family Identification",
        "prompt": (
            "Scenario 1-C\n"
            "Every business day at 6:00 AM, a flow must query a SQL database\n"
            "for overnight orders and email a summary to the operations team.\n"
            "Which trigger type handles this best?"
        ),
        "options": [
            "A. Automated — SQL Server 'When an item is created'",
            "B. Instant — Manually trigger a flow",
            "C. Scheduled — Recurrence (weekdays at 06:00, with time zone set)",
            "D. Scheduled — Sliding Window",
        ],
        "answer": "C",
        "explanation": (
            "Correct answer: C — Scheduled Recurrence.\n"
            "\n"
            "This is a periodic batch task with no reactive event — it runs\n"
            "on the clock. Recurrence is correct. Configuration:\n"
            "  Frequency: Week\n"
            "  Interval:  1\n"
            "  On these days: Monday, Tuesday, Wednesday, Thursday, Friday\n"
            "  At these hours: 6\n"
            "  Time zone: (your organisation's time zone)\n"
            "\n"
            "Why not D (Sliding Window)?\n"
            "Sliding Window is correct IF missing a run matters — for example,\n"
            "if skipping a day would leave a gap in a financial audit trail.\n"
            "For an operations summary email, a missed morning is acceptable;\n"
            "Recurrence is simpler and sufficient.\n"
            "\n"
            "Why not A?\n"
            "SQL Server's polling trigger fires on new rows, not on a schedule.\n"
            "It would fire for every new order as it is inserted — not once\n"
            "per morning as a summary."
        ),
    },
    # ── Section 2: Match scenario to specific connector + trigger name ─────
    {
        "section": "Section 2: Connector and Trigger Selection",
        "prompt": (
            "Scenario 2-A\n"
            "A flow must start immediately when a customer submits a response\n"
            "to a Microsoft Forms survey. Which connector and trigger are correct?"
        ),
        "options": [
            "A. SharePoint — When an item is created",
            "B. Microsoft Forms — When a new response is submitted",
            "C. Office 365 Outlook — When a new email arrives (V3)",
            "D. HTTP — GET the Forms API on a schedule",
        ],
        "answer": "B",
        "explanation": (
            "Correct answer: B — Microsoft Forms, 'When a new response is submitted'.\n"
            "\n"
            "Microsoft Forms has a dedicated connector with a webhook trigger\n"
            "that fires the moment a respondent submits the form. This is the\n"
            "purpose-built integration — no polling, no workarounds.\n"
            "\n"
            "The trigger is a Standard connector (no premium license needed)\n"
            "and returns the response ID as dynamic content. Use a 'Get\n"
            "response details' action immediately after the trigger to retrieve\n"
            "the individual answer fields.\n"
            "\n"
            "Why not A? SharePoint does not know about Forms submissions\n"
            "unless you configure Forms to save responses to SharePoint —\n"
            "and even then, you would use the Forms trigger, not SharePoint's.\n"
            "\n"
            "Why not D? Polling the API with HTTP adds latency and complexity.\n"
            "The webhook trigger is instant and simpler."
        ),
    },
    {
        "section": "Section 2: Connector and Trigger Selection",
        "prompt": (
            "Scenario 2-B\n"
            "A Teams user right-clicks a message posted in a support channel\n"
            "and selects 'Create support ticket' from the message actions menu.\n"
            "This should create a ticket in the organisation's helpdesk system\n"
            "via its REST API.\n"
            "Which trigger is used to receive the Teams action?"
        ),
        "options": [
            "A. Microsoft Teams — When a new channel message is added",
            "B. Microsoft Teams — When a Teams message action is triggered",
            "C. HTTP — Webhook endpoint registered in Teams admin",
            "D. Instant — Manually trigger a flow",
        ],
        "answer": "B",
        "explanation": (
            "Correct answer: B — 'When a Teams message action is triggered'.\n"
            "\n"
            "This trigger registers the flow as a right-click context action\n"
            "on Teams messages. When a user selects the flow from the\n"
            "'More actions' menu on a message, the flow fires with the\n"
            "message content as dynamic content:\n"
            "  - messageBody: the full message text\n"
            "  - messageSender: email of the person who sent the message\n"
            "  - teamId, channelId: where the message lives\n"
            "  - messageLink: a deep link to the message\n"
            "\n"
            "The flow can then use the HTTP connector (Premium) to POST\n"
            "to the helpdesk REST API to create the ticket, passing the\n"
            "messageBody as the ticket description.\n"
            "\n"
            "Why not A? 'When a new channel message is added' is a polling\n"
            "trigger that fires on every message — not on a user action.\n"
            "\n"
            "Why not D? Manually trigger requires navigating to the PA portal\n"
            "or mobile app — it is not accessible from within Teams natively."
        ),
    },
    # ── Section 3: Connector tier classification ───────────────────────────
    {
        "section": "Section 3: Standard vs Premium Classification",
        "prompt": (
            "Classify each connector as Standard (S) or Premium (P).\n"
            "Which answer lists them correctly?\n"
            "\n"
            "  1. Office 365 Outlook\n"
            "  2. SQL Server\n"
            "  3. Microsoft Teams\n"
            "  4. HTTP\n"
            "  5. SharePoint\n"
            "  6. Salesforce\n"
        ),
        "options": [
            "A. S, P, S, P, S, P",
            "B. P, P, S, P, S, P",
            "C. S, S, S, S, S, S",
            "D. S, P, S, S, P, P",
        ],
        "answer": "A",
        "explanation": (
            "Correct answer: A — S, P, S, P, S, P\n"
            "\n"
            "  1. Office 365 Outlook   → STANDARD  (included in M365)\n"
            "  2. SQL Server           → PREMIUM   (requires Power Automate plan)\n"
            "  3. Microsoft Teams      → STANDARD  (included in M365)\n"
            "  4. HTTP                 → PREMIUM   (requires Power Automate plan)\n"
            "  5. SharePoint           → STANDARD  (included in M365)\n"
            "  6. Salesforce           → PREMIUM   (third-party SaaS connector)\n"
            "\n"
            "Rule of thumb:\n"
            "  Standard: Core Microsoft 365 productivity services\n"
            "  Premium:  Databases, external SaaS platforms, raw HTTP calls,\n"
            "            and Azure infrastructure services\n"
            "\n"
            "A flow mixing standard and premium connectors requires ALL users\n"
            "who run the flow to have a Power Automate per-user plan — or\n"
            "use a per-flow plan license assigned to the flow itself."
        ),
    },
    {
        "section": "Section 3: Standard vs Premium Classification",
        "prompt": (
            "Scenario 3-B\n"
            "You are designing a flow for a team of 200 users. The flow:\n"
            "  - Triggers from a SharePoint list change (Standard)\n"
            "  - Queries an Azure SQL database (Premium)\n"
            "  - Sends an Outlook email (Standard)\n"
            "\n"
            "What is the MOST COST-EFFECTIVE licensing approach?"
        ),
        "options": [
            "A. 200 Microsoft 365 licenses — standard connectors only cover it",
            "B. 200 Power Automate per-user licenses — each user needs their own",
            "C. 1 Power Automate per-flow license — assigned to this flow, covers all users",
            "D. No license needed — SQL Server is free in Azure",
        ],
        "answer": "C",
        "explanation": (
            "Correct answer: C — 1 Power Automate per-flow license.\n"
            "\n"
            "The per-flow plan (~$100/month) is assigned to one specific flow.\n"
            "Any number of users can trigger and run that flow regardless of\n"
            "their individual license tier. This is dramatically cheaper than\n"
            "200 per-user licenses when:\n"
            "  - The flow is shared across many users\n"
            "  - Each user runs it infrequently\n"
            "\n"
            "Per-user plan is better when one user needs many different flows\n"
            "with premium connectors — the per-user plan covers unlimited flows\n"
            "for that one user.\n"
            "\n"
            "Why not A? M365 licenses do not cover premium connectors (SQL Server,\n"
            "HTTP). The flow would be saved but disabled.\n"
            "\n"
            "Why not D? The Azure SQL license is separate from the Power Automate\n"
            "connector license. SQL Server connector is Premium regardless of\n"
            "how the database itself is licensed."
        ),
    },
    # ── Section 4: Polling vs webhook ──────────────────────────────────────
    {
        "section": "Section 4: Polling vs Webhook Triggers",
        "prompt": (
            "Scenario 4-A\n"
            "A security team's flow must fire within 5 seconds of a suspicious\n"
            "login event being written to an Azure AD audit log.\n"
            "Which trigger mechanism is required to meet this latency requirement?"
        ),
        "options": [
            "A. Polling trigger — checks every 3 minutes",
            "B. Polling trigger — checks every 1 minute (premium plan)",
            "C. Webhook trigger — source service pushes event to Power Automate",
            "D. Recurrence trigger — checks every 30 seconds",
        ],
        "answer": "C",
        "explanation": (
            "Correct answer: C — Webhook trigger.\n"
            "\n"
            "Polling triggers check for new events on an interval — the\n"
            "minimum polling interval on a paid plan is approximately 1 minute.\n"
            "A 5-second latency requirement cannot be met by polling.\n"
            "\n"
            "Webhook triggers (push model) fire the instant the source service\n"
            "delivers the event payload to Power Automate's webhook endpoint.\n"
            "Latency is typically 1-3 seconds end-to-end.\n"
            "\n"
            "For Azure AD sign-in events, consider:\n"
            "  - Azure Event Grid + HTTP Webhook trigger in Power Automate\n"
            "  - Microsoft Sentinel alert → Logic Apps or Power Automate\n"
            "  - Microsoft Defender for Identity connector (Premium, webhook)\n"
            "\n"
            "Why not D? Recurrence is not a reactive trigger at all — it runs\n"
            "at fixed times regardless of whether any event occurred, and\n"
            "30-second recurrence is not supported by Power Automate."
        ),
    },
    {
        "section": "Section 4: Polling vs Webhook Triggers",
        "prompt": (
            "Scenario 4-B\n"
            "Your flow uses the SharePoint 'When an item is created' trigger.\n"
            "A new list item is created at 2:00 PM. At what time is the flow\n"
            "MOST LIKELY to start running on a standard M365 plan?"
        ),
        "options": [
            "A. 2:00:01 PM — webhook delivers instantly",
            "B. 2:00:30 PM — 30-second polling interval",
            "C. 2:03 PM — up to 3-minute polling interval",
            "D. 2:15 PM — 15-minute polling interval on free plan",
        ],
        "answer": "C",
        "explanation": (
            "Correct answer: C — approximately 2:03 PM.\n"
            "\n"
            "SharePoint triggers are POLLING triggers. Power Automate calls\n"
            "the SharePoint REST API every 3 minutes (on paid plans) to check\n"
            "for new or modified items since the last check.\n"
            "\n"
            "If the item is created immediately after a poll, the next poll\n"
            "will detect it up to 3 minutes later. On average, the lag is\n"
            "1.5 minutes; worst case is 3 minutes.\n"
            "\n"
            "On a free Microsoft 365 trial plan (no Power Automate add-on),\n"
            "the interval can be up to 15 minutes — that would be answer D.\n"
            "\n"
            "This is why answer A is wrong: SharePoint does NOT use webhooks\n"
            "for Power Automate triggers despite SharePoint supporting webhooks\n"
            "in general. The Power Automate SharePoint connector is polling-based.\n"
            "\n"
            "For near-instant SharePoint triggering, use an Azure Event Grid\n"
            "subscription on SharePoint events + HTTP webhook trigger in Power\n"
            "Automate — but this requires significant Azure infrastructure setup."
        ),
    },
    # ── Section 5: Trigger conditions ─────────────────────────────────────
    {
        "section": "Section 5: Trigger Conditions",
        "prompt": (
            "Scenario 5-A\n"
            "A SharePoint list has 500 items updated per day. Your flow should\n"
            "only process updates where the 'Status' column equals 'Approved'.\n"
            "Approximately 10 items per day meet this criterion.\n"
            "\n"
            "Which approach MINIMISES unnecessary flow run consumption?"
        ),
        "options": [
            "A. Use a Condition action inside the flow body to check Status.\n"
            "   The flow runs 500 times/day, skips 490 via Condition.",
            "B. Use a trigger condition expression to filter Status = 'Approved'.\n"
            "   The flow starts only 10 times/day.",
            "C. Use an OData filter on a separate 'Get items' action to fetch\n"
            "   only Approved items after the trigger fires.",
            "D. Build two separate flows — one for Approved, one for the rest.",
        ],
        "answer": "B",
        "explanation": (
            "Correct answer: B — trigger condition expression.\n"
            "\n"
            "Trigger conditions are evaluated BEFORE the flow run starts.\n"
            "If the condition evaluates to false, the event is discarded and\n"
            "NO flow run is created — no run credits are consumed.\n"
            "\n"
            "Expression to add in the trigger's Settings → Trigger Conditions:\n"
            "  @equals(triggerBody()?['Status']?['Value'], 'Approved')\n"
            "\n"
            "This reduces 500 run starts/day to 10. At scale (a million events),\n"
            "the difference between options A and B is hundreds of thousands of\n"
            "run credits — the difference between being within quota or not.\n"
            "\n"
            "Why option A is less efficient:\n"
            "The flow STARTS 500 times. Each start consumes one flow run from\n"
            "the daily API request quota, even if the Condition action\n"
            "immediately terminates the run. Run history also fills up with\n"
            "490 'skipped' runs per day, making debugging harder.\n"
            "\n"
            "Option C is a different technique (server-side OData filtering\n"
            "on Get items results) — useful WITHIN a flow body, not for\n"
            "controlling when the trigger fires.\n"
            "\n"
            "Option D adds maintenance overhead and doubles connections."
        ),
    },
    {
        "section": "Section 5: Trigger Conditions",
        "prompt": (
            "Scenario 5-B\n"
            "Write the correct trigger condition expression to filter\n"
            "Outlook emails where the sender is NOT from your domain\n"
            "(contoso.com). Only emails from external senders should\n"
            "trigger the flow.\n"
            "\n"
            "Which expression is correct?"
        ),
        "options": [
            "A. @equals(triggerBody()?['From'], 'external')",
            "B. @not(endsWith(triggerBody()?['From'], '@contoso.com'))",
            "C. @contains(triggerBody()?['From'], 'external')",
            "D. @triggerBody()?['From'] != '@contoso.com'",
        ],
        "answer": "B",
        "explanation": (
            "Correct answer: B — @not(endsWith(triggerBody()?['From'], '@contoso.com'))\n"
            "\n"
            "Breakdown of the expression:\n"
            "  triggerBody()?['From']\n"
            "    → The From address string from the email trigger payload\n"
            "       e.g. 'alice@contoso.com' or 'vendor@external.com'\n"
            "\n"
            "  endsWith(..., '@contoso.com')\n"
            "    → true if the address ends with your domain (internal sender)\n"
            "\n"
            "  @not(...)\n"
            "    → inverts the result: true when the sender is NOT internal\n"
            "       = external sender → trigger fires\n"
            "\n"
            "Why not A?\n"
            "  The From field contains an email address, not the literal\n"
            "  string 'external'. This would never match.\n"
            "\n"
            "Why not C?\n"
            "  'external' is not a reliable substring — external domains\n"
            "  could be anything (vendor.com, partner.org, etc.)\n"
            "\n"
            "Why not D?\n"
            "  Power Automate expression language uses function syntax,\n"
            "  not comparison operators (!=, ==). The correct comparison\n"
            "  function is @not(equals(...)) or @not(endsWith(...))."
        ),
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_quiz(questions: list[dict]) -> None:
    """
    Run the interactive self-check quiz.

    Displays each question, collects the learner's answer, and shows
    the explanation regardless of whether the answer is correct.
    Tracks score across sections and prints a final summary.
    """
    score = 0
    total = len(questions)
    current_section = None

    print("=" * 65)
    print("  Module 02 — Connector and Trigger Matching Self-Check")
    print("=" * 65)
    print("\nType the letter of your answer (A, B, C, or D) and press Enter.")
    print("The explanation is shown after every question.\n")

    for i, q in enumerate(questions, 1):
        # Print section header when section changes
        if q.get("section") != current_section:
            current_section = q["section"]
            print(f"\n{'─' * 65}")
            print(f"  {current_section}")
            print(f"{'─' * 65}\n")

        print(f"Question {i} of {total}")
        print()
        print(q["prompt"])
        print()
        for option in q["options"]:
            print(f"  {option}")
        print()

        # Collect and validate input
        while True:
            raw = input("Your answer: ").strip().upper()
            if raw in ("A", "B", "C", "D"):
                break
            print("  Please enter A, B, C, or D.")

        # Evaluate
        correct = raw == q["answer"].upper()
        if correct:
            score += 1
            print("\n  Correct.\n")
        else:
            print(f"\n  Incorrect. The correct answer is {q['answer']}.\n")

        # Always show explanation
        print("  Explanation:")
        for line in q["explanation"].split("\n"):
            print(f"  {line}")

        print()
        if i < total:
            input("  Press Enter to continue...\n")

    # Final summary
    print("=" * 65)
    print(f"  Final score: {score} / {total}")
    pct = score / total * 100
    print(f"  ({pct:.0f}%)")
    print()

    if pct == 100:
        print("  Perfect score. You have a solid grasp of Power Automate")
        print("  connector and trigger selection.")
    elif pct >= 70:
        print("  Good result. Review the explanations for questions you missed")
        print("  and re-read the relevant sections in the guides.")
    else:
        print("  Work through the guide again, focusing on:")
        print("    - Guide 01: Trigger Types (trigger families, polling vs webhook,")
        print("                trigger conditions)")
        print("    - Guide 02: Connectors Deep Dive (tier classification, auth types)")
        print("  Then re-run this exercise.")

    print("=" * 65)


if __name__ == "__main__":
    try:
        run_quiz(QUESTIONS)
    except KeyboardInterrupt:
        print("\n\nExited early. Run the script again to restart from the beginning.")
        sys.exit(0)
