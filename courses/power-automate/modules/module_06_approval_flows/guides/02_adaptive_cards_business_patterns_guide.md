# Adaptive Cards and Business Process Patterns

> **Reading time:** ~22 min | **Module:** 6 — Approval Flows | **Prerequisites:** Module 4

## In Brief

Adaptive Cards are JSON-defined UI components that render natively inside Microsoft Teams and Outlook. When you use them in approval flows, approvers get a rich, interactive experience—formatted data, action buttons, and inline response forms—directly in their Teams channel or chat, without leaving the application.

<div class="callout-insight">

<strong>Insight:</strong> A standard approval email is a notification. An Adaptive Card in Teams is an interactive application. Approvers can read context, review data, type comments, and submit their decision without opening a browser or navigating to another system.

</div>

<div class="callout-key">

<strong>Key Concept:</strong> Adaptive Cards are JSON-defined UI components that render natively inside Microsoft Teams and Outlook. When you use them in approval flows, approvers get a rich, interactive experience—formatted data, action buttons, and inline response forms—directly in their Teams channel or chat, without leaving the application.

</div>


---

## What Are Adaptive Cards?

Adaptive Cards are a platform-agnostic UI framework. You write a JSON payload describing the card's structure, and the host application (Teams, Outlook, Windows Notifications) renders it using that platform's native UI components. The same JSON renders differently on desktop and mobile—the platform adapts it.

<div class="callout-insight">

<strong>Insight:</strong> Adaptive Cards are a platform-agnostic UI framework.

</div>


For Power Automate approval flows, Adaptive Cards appear in:

| Host | How cards appear | User action |
|------|-----------------|-------------|
| Microsoft Teams | Posted in a chat or channel | Click button, fill form inline |
| Outlook | Actionable Message in email body | Click button in email |
| Teams Approvals App | Structured approval card | Approve/Reject with comments |

The **Post Adaptive Card and Wait for Response** action in Power Automate handles the full loop: post the card to a Teams user or channel, pause the flow, and resume when the user interacts with the card.

---

## Adaptive Card JSON Structure

Every Adaptive Card is a JSON object with this top-level structure:

<div class="callout-key">

<strong>Key Point:</strong> Every Adaptive Card is a JSON object with this top-level structure:


example.json


The following implementation builds on the approach above:


### Body Elements

The `body` array contains the v...

</div>


The following implementation builds on the approach above:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```json
{
    "type": "AdaptiveCard",
    "version": "1.4",
    "body": [],
    "actions": []
}
```

</div>
</div>

### Body Elements

The `body` array contains the visible content of the card. Key element types:

**TextBlock** — Displays text with formatting control:


The following implementation builds on the approach above:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```json
{
    "type": "TextBlock",
    "text": "Expense Request: $450.00",
    "size": "Large",
    "weight": "Bolder",
    "color": "Accent"
}
```

</div>
</div>

**FactSet** — Renders a label-value pair list, ideal for structured data:


The following implementation builds on the approach above:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```json
{
    "type": "FactSet",
    "facts": [
        { "title": "Submitted by:", "value": "Priya Patel" },
        { "title": "Department:", "value": "Engineering" },
        { "title": "Amount:", "value": "$450.00" },
        { "title": "Category:", "value": "Software License" }
    ]
}
```

</div>
</div>

**Container** — Groups elements with optional background color and padding:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.json</span>
</div>

```json
{
    "type": "Container",
    "style": "emphasis",
    "items": [
        { "type": "TextBlock", "text": "Justification" },
        { "type": "TextBlock", "text": "Annual JetBrains renewal", "wrap": true }
    ]
}
```

</div>
</div>

**Input.Text** — Collects text input inline:

```json
{
    "type": "Input.Text",
    "id": "approverComments",
    "label": "Comments (optional)",
    "isMultiline": true,
    "placeholder": "Add context for the requestor..."
}
```

**Input.ChoiceSet** — Dropdown or radio buttons:

```json
{
    "type": "Input.ChoiceSet",
    "id": "decision",
    "style": "expanded",
    "choices": [
        { "title": "Approve", "value": "approve" },
        { "title": "Reject", "value": "reject" },
        { "title": "Request More Information", "value": "more_info" }
    ]
}
```

### Actions Array

The `actions` array defines the buttons at the bottom of the card:

```json
{
    "actions": [
        {
            "type": "Action.Submit",
            "title": "Submit Decision",
            "data": { "cardAction": "submit" }
        }
    ]
}
```

When an `Action.Submit` is clicked, the card submits all input values plus any static data in the `data` property back to Power Automate.

---

## Complete Approval Card Example

This card collects a decision and optional comments for an expense request:

```json
{
    "type": "AdaptiveCard",
    "version": "1.4",
    "body": [
        {
            "type": "TextBlock",
            "text": "Expense Approval Required",
            "size": "Large",
            "weight": "Bolder"
        },
        {
            "type": "TextBlock",
            "text": "Please review the following expense request and submit your decision.",
            "wrap": true,
            "color": "Default",
            "spacing": "Small"
        },
        {
            "type": "FactSet",
            "facts": [
                { "title": "Requested by:", "value": "Priya Patel" },
                { "title": "Amount:", "value": "$450.00" },
                { "title": "Category:", "value": "Software License" },
                { "title": "Submitted:", "value": "2024-03-15" }
            ]
        },
        {
            "type": "Container",
            "style": "emphasis",
            "items": [
                {
                    "type": "TextBlock",
                    "text": "Business Justification",
                    "weight": "Bolder"
                },
                {
                    "type": "TextBlock",
                    "text": "Annual renewal for JetBrains IDEs used by the backend team.",
                    "wrap": true
                }
            ]
        },
        {
            "type": "Input.ChoiceSet",
            "id": "decision",
            "label": "Your Decision",
            "isRequired": true,
            "style": "expanded",
            "choices": [
                { "title": "Approve", "value": "approve" },
                { "title": "Reject", "value": "reject" }
            ]
        },
        {
            "type": "Input.Text",
            "id": "comments",
            "label": "Comments",
            "isMultiline": true,
            "placeholder": "Optional: add comments for the requestor"
        }
    ],
    "actions": [
        {
            "type": "Action.Submit",
            "title": "Submit Decision",
            "style": "positive"
        }
    ]
}
```

> **On screen:** You can preview this card at [adaptivecards.io/designer](https://adaptivecards.io/designer). Paste the JSON into the Card Payload Editor on the left, select "Microsoft Teams" as the host, and see the rendered result on the right.

---

## Post Adaptive Card and Wait for Response in Teams

This Power Automate action combines posting and waiting in a single step.

**Action name:** Post adaptive card and wait for a response (Microsoft Teams)

**Required fields:**

| Field | What to enter |
|-------|--------------|
| Post as | Flow bot (or a specific user) |
| Post in | Chat with Flow bot, Channel, or Group chat |
| Recipient / Team / Channel | Dynamic value for who receives the card |
| Message | Your Adaptive Card JSON |
| Update message | Text shown on the card after the user submits |

> **On screen:** When you add this action, the **Message** field accepts raw JSON. You can use the **Adaptive Card** option from the panel on the right, but pasting your JSON directly gives you full control over the card structure.

### Dynamic Data Binding

The card's static text becomes dynamic when you inject Power Automate expressions into the JSON string. Use string interpolation to insert values from previous action outputs:

```text
"value": "@{triggerOutputs()?['body/Amount']}"
```

In the Power Automate designer, you build this by typing the JSON structure and clicking **Dynamic content** when you reach a value that should come from the flow. The designer inserts the expression token for you.

### Reading the Response

After the user submits the card, the action outputs a **User input** object. This is a JSON string containing all the input field values, keyed by their `id` properties.

To access the decision from the card above:

```text
outputs('Post_adaptive_card_and_wait_for_a_response')?['body/data/decision']
```

To access the comments:

```text
outputs('Post_adaptive_card_and_wait_for_a_response')?['body/data/comments']
```

---

## Multi-Stage Approval Pipeline

Complex business processes require multiple sequential approvals. The pattern: each stage completes before the next begins, and rejection at any stage terminates the pipeline.

### Architecture: Request → Manager → Finance → Confirmation

```text
Stage 1: Manager Approval
  ↓ Approved
Stage 2: Finance Director Approval
  ↓ Approved
Stage 3: Notify all parties, update records
  ↓
Complete
```

**If rejected at any stage:** Send rejection notification with comments, update record status, stop the pipeline.

### Implementation Structure

```text
[Trigger: New expense request]
    |
    ▼
[Stage 1: Post card to Manager in Teams]
    |
    ├── Decision == "reject" → Update status = "Rejected by Manager", send notification → End
    |
    ▼ (Decision == "approve")
[Update SharePoint: ManagerApproved = true, ManagerComments = ...]
    |
    ▼
[Stage 2: Post card to Finance Director in Teams]
    |
    ├── Decision == "reject" → Update status = "Rejected by Finance", send notification → End
    |
    ▼ (Decision == "approve")
[Update SharePoint: FinanceApproved = true, FinanceComments = ...]
    |
    ▼
[Send confirmation to all parties]
[Update status = "Fully Approved"]
```

> **On screen:** In the Power Automate designer, multi-stage flows appear as a vertical chain of actions with Condition branches after each approval step. Collapse completed stages to keep the canvas readable.

### SharePoint Tracking Schema

Use a SharePoint list to track approval history across stages:

| Column | Type | Purpose |
|--------|------|---------|
| Status | Choice | Current pipeline state |
| Stage1ApproverEmail | Single line | Who was asked |
| Stage1Decision | Single line | approve / reject |
| Stage1Comments | Multiline | Approver's notes |
| Stage1CompletedAt | Date/Time | When they responded |
| Stage2ApproverEmail | Single line | Finance director |
| Stage2Decision | Single line | approve / reject |
| Stage2Comments | Multiline | Finance notes |
| Stage2CompletedAt | Date/Time | When they responded |

Update these columns after each stage using **Update item** (SharePoint). This creates a full audit trail visible directly in SharePoint without querying Dataverse.

### Tracking in Dataverse

For higher-volume or more complex pipelines, store approval history in a Dataverse table instead. Create a custom table **Approval History** with columns:

- RequestId (Lookup to the main request table)
- Stage (Whole number: 1, 2, 3)
- ApproverEmail (Email)
- Decision (Choice: Approved, Rejected, Escalated)
- Comments (Multiline text)
- ResponseTimestamp (Date and Time)

Use **Add a new row** (Dataverse) to write each stage's outcome. Use **List rows** (Dataverse) to retrieve the full history for reporting.

---

## Business Process Patterns

### Pattern 1: Escalation

When an approver does not respond within a defined SLA (e.g., 48 hours), automatically escalate to their manager or a backup approver.

**Implementation with Parallel Branches:**

```text
[Start and wait for an approval]
    |
    ▼ (use Create an approval, not Start and wait)

Parallel branch A:              Parallel branch B:
[Wait for an approval]          [Delay: 48 hours]
    |                               |
    ▼ (response received)           ▼ (timeout)
[Cancel the approval]           [Cancel the approval]
[Act on response normally]      [Create new approval → escalation manager]
```

> **Key implementation note:** Use the **Create an approval** (non-blocking) action followed by a **Delay** action and a **Wait for an approval** action in parallel branches. This requires the non-blocking pattern because you need to coordinate two concurrent branches.

**Practical approach:** The Microsoft-documented escalation pattern uses a Do Until loop with a deadline check rather than parallel branches, because parallel branches in Power Automate cannot easily share state. Use a SharePoint column or Dataverse field to signal whether a response arrived, and check it inside the timer branch.

### Pattern 2: Delegation

Allow approvers to reassign an approval to another person. Implement this by:

1. Adding a "Delegate" option to the Adaptive Card's choice set
2. Adding an **Input.Text** field: "Delegate to (email)"
3. In the flow: if decision == "delegate", create a new approval assigned to the delegate's email
4. Update the tracking record with the delegation chain

```text
Original Approver → clicks Delegate → enters delegate@your-org.com
    ↓
Flow reads delegate email
Flow creates new approval assigned to delegate
Flow logs: "Delegated by original@your-org.com to delegate@your-org.com"
Delegate receives approval notification
```

### Pattern 3: SLA Enforcement

Track how long approvals take and flag SLA breaches for reporting.

**Fields to track:**
- `RequestSubmittedAt` — when the flow triggered
- `ApprovalSentAt` — when the approval notification was sent
- `ApprovalRespondedAt` — when the approver responded
- `SLABreached` — boolean, set to true if response time > SLA threshold

**Calculation in Power Automate:**

```text
SLA threshold: 24 hours (in minutes: 1440)

Response time in minutes:
  dateDiff(ApprovalSentAt, ApprovalRespondedAt, 'minutes')

SLA breached:
  greater(responsetime, 1440)
```

Set `SLABreached = true` in your SharePoint or Dataverse record when the response time exceeds the threshold. Build a Power BI report or SharePoint view filtered by `SLABreached = true` to surface patterns.

### Pattern 4: Conditional Routing by Amount

Route approvals to different approvers based on the value of the request.

```text
Amount < $500       → Direct Manager
Amount $500–$5000   → Department Head
Amount > $5000      → VP Finance + CFO (everyone must approve)
```

Implementation: Use a **Switch** action (or nested Conditions) at the start of the flow to set an `ApproverEmail` variable and `ApprovalType` variable, then pass those variables into the approval action.

```text
[Initialize variable: ApproverEmail]
[Initialize variable: ApprovalType]

[Switch on Amount]
    Case < 500:
        Set ApproverEmail = Manager.Mail
        Set ApprovalType = "Approve/Reject - First to respond"
    Case < 5000:
        Set ApproverEmail = DeptHead.Mail
        Set ApprovalType = "Approve/Reject - First to respond"
    Default (≥ 5000):
        Set ApproverEmail = "vp@your-org.com;cfo@your-org.com"
        Set ApprovalType = "Approve/Reject - Everyone must approve"

[Start and wait for an approval]
    Approval type: ApprovalType variable
    Assigned to: ApproverEmail variable
```

---

## Step-by-Step: Building the Multi-Stage Pipeline

This walkthrough extends the expense approval from Guide 01 to add a Finance stage for requests over $500.

### Step 1: Add an Amount Check Before Stage 1

After the trigger, add a **Condition**:
- Amount (from trigger) is greater than 500

In the NO branch: proceed directly to standard manager approval (single stage).

In the YES branch: proceed to the two-stage pipeline built in the steps below.

### Step 2: Stage 1 — Manager Approval via Teams Card

Inside the YES branch:

1. Add **Get manager (V2)** using the submitter's email
2. Add **Post adaptive card and wait for a response** (Microsoft Teams)
   - Post as: Flow bot
   - Post in: Chat with Flow bot
   - Recipient: Manager's email (dynamic content from Get manager)
   - Message: Paste your approval card JSON, replacing static values with dynamic content
   - Update message: `Your decision has been recorded. Thank you.`

3. Add a **Condition**:
   - Value: `outputs('Post_adaptive_card')?['body/data/decision']`
   - is equal to: `approve`

4. In the NO branch: Update SharePoint item Status = "Rejected - Manager", send notification, end.

5. In the YES branch: Update SharePoint columns `Stage1Decision = approve`, `Stage1Comments = [comments input]`, `Stage1CompletedAt = utcNow()`.

> **On screen:** When building the Post adaptive card action, switch the designer to **Code view** for the Message field by clicking the **</>** icon. This lets you paste the full card JSON and insert dynamic content expressions directly.

### Step 3: Stage 2 — Finance Approval

After the YES branch from Stage 1:

1. Add **Get user profile (V2)** to look up the Finance Director's email (or use a static email stored in an environment variable)
2. Add another **Post adaptive card and wait for a response**
   - Recipient: Finance Director's email
   - Message: A finance-specific card showing the full request plus the manager's approval and comments

3. Add a **Condition** on the finance decision:
   - Approve → Update SharePoint: Stage2Decision, send full confirmation to all parties, set Status = "Fully Approved"
   - Reject → Update SharePoint, send rejection with Finance's comments, set Status = "Rejected - Finance"

### Step 4: Final Confirmation

In the fully approved branch, send a single email or Teams message to:
- The original requestor (approved, they can proceed)
- The manager (their approval is on record)
- Finance (their sign-off is confirmed)

Include a summary of both stages: who approved, when, and any comments.

---

## Common Pitfalls

<div class="callout-danger">

<strong>Danger:</strong> The pitfalls below are the most common mistakes practitioners make. Each one can silently degrade your results without obvious errors.

</div>

- **Adaptive Card JSON invalid**: The designer does not validate card JSON before runtime. Use the Adaptive Card Designer at adaptivecards.io to validate structure before pasting into Power Automate.
- **Input.ChoiceSet returns null**: If the user submits without selecting a choice and `isRequired` is not set to true, the value is null. Always check for null before using the decision value.
- **Card not rendering in Teams**: Verify the card version matches what Teams supports. Teams supports Adaptive Card schema 1.4 and below. Schema 1.5+ features may not render.
- **User input expressions break**: The full expression path for reading card responses is verbose. Save it as a variable immediately after the card action to avoid repeating it: `Set variable ResponseData = outputs('Post_adaptive_card')?['body/data']`
- **Parallel branch state sharing**: Power Automate parallel branches cannot write to the same variable. Use SharePoint or Dataverse columns as shared state for escalation timing patterns.

<div class="callout-warning">

<strong>Warning:</strong> - **Adaptive Card JSON invalid**: The designer does not validate card JSON before runtime.

</div>

---

## Connections


<div class="callout-info">

<strong>Info:</strong> This section maps how this guide connects to the broader course. Use these links to navigate related material.


- **Builds on:** Guide 01 — Approvals connector basics; Module 04 — Conditions and parallel branches
- **Leads to:** Module 07 — Desktop flows for document processing in approval pipelines
- **Related to:** Module 03 — Expressions for dynamic card content; Module 05 — SharePoint for approval tracking

---




## Practice Questions

**Question 1 — Conceptual:** Based on the concepts in this guide, explain in your own words why the core technique matters and when you would choose it over alternatives.

**Question 2 — Application:** Sketch out how you would apply the main concept from this guide to a real-world dataset or problem you have encountered. What would you need to watch out for?


## Further Reading

- [Adaptive Cards designer (interactive)](https://adaptivecards.io/designer/)
- [Adaptive Cards schema explorer](https://adaptivecards.io/explorer/)
- [Microsoft Docs: Post adaptive card and wait for response](https://learn.microsoft.com/en-us/power-automate/adaptive-cards/create-adaptive-cards)
- [Microsoft Docs: Approval escalation with parallel branches](https://learn.microsoft.com/en-us/power-automate/parallel-modern-approvals)


---

## Cross-References

<a class="link-card" href="./02_adaptive_cards_business_patterns_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Interactive slide deck covering the key concepts with visual examples.</div>
</a>

<a class="link-card" href="../notebooks/01_adaptive_card_designer.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises and real data.</div>
</a>
