# Approvals Connector: Automating Human Decision Points

> **Reading time:** ~14 min | **Module:** 6 — Approval Flows | **Prerequisites:** Module 4

## In Brief

The Approvals connector lets you embed structured human decisions into your flows. Instead of sending a generic email and waiting for a reply, the connector creates a trackable approval record, routes it to the right people, handles responses, and feeds the decision back into your automation—all without writing a line of custom email logic.

<div class="callout-insight">

<strong>Insight:</strong> Approvals are not just email notifications. They are persistent records with a status (Pending, Approved, Rejected) that Power Automate can query, update, and act on—even days or weeks after the request was created.

</div>

<div class="callout-key">

<strong>Key Concept:</strong> The Approvals connector lets you embed structured human decisions into your flows. Instead of sending a generic email and waiting for a reply, the connector creates a trackable approval record, routes it to the right people, handles responses, and feeds the decision back into your automation—all without writing a line of custom email logic.

</div>


---

## Why the Approvals Connector Exists

Before the Approvals connector, teams handled sign-off workflows by emailing someone a document, waiting for a reply, and manually updating a spreadsheet. Problems:

<div class="callout-insight">

<strong>Insight:</strong> Before the Approvals connector, teams handled sign-off workflows by emailing someone a document, waiting for a reply, and manually updating a spreadsheet.

</div>


- No audit trail linking the request to the decision
- Approver's reply can get buried in email
- Requestor has no visibility into status without following up
- Escalation requires someone to remember to check

The Approvals connector solves all four problems with a dedicated infrastructure layer: approval records stored in Dataverse, email/Teams notifications generated automatically, a unified Approvals Center in Teams for approvers, and response data flowing directly back into your flow logic.

---

## Approval Types

Power Automate offers four distinct approval patterns. Choosing the right one determines the behavior of routing, notification, and completion.

<div class="callout-key">

<strong>Key Point:</strong> Power Automate offers four distinct approval patterns.

</div>


### Approve/Reject — First to Respond

The most common pattern. One or more approvers are notified simultaneously. The approval completes the moment any single approver responds. Useful when any authorized person can make the call.

```
Approver A ──┐
Approver B ──┼──► First response wins ──► Flow continues
Approver C ──┘
```

**When to use:** Expense approvals where any manager in the group can authorize.

### Approve/Reject — Everyone Must Approve

All named approvers must individually respond before the approval completes. If anyone rejects, the outcome is Rejected regardless of the other responses. Useful for compliance scenarios where multiple sign-offs are required.

```
Approver A ──► Must respond ──┐
Approver B ──► Must respond ──┼──► All approved? ──► Flow continues
Approver C ──► Must respond ──┘
```

**When to use:** Contract approvals requiring legal, finance, and operations sign-off.

### Custom Responses — First to Respond

Instead of the binary Approve/Reject, you define your own response options (e.g., Approve, Reject, Request More Information, Delegate). First responder wins. Useful when a simple yes/no doesn't capture the decision space.

**When to use:** IT change requests that might be Approved, Rejected, or Deferred.

### Custom Responses — Everyone Must Respond

Every approver receives the custom response options and must individually respond before the flow continues. Useful for collecting structured feedback from multiple stakeholders simultaneously.

**When to use:** Pre-launch sign-off checklist where Security, Legal, and Marketing each confirm their area.

---

## Core Actions

### Start and Wait for an Approval (Blocking)

This is a single action that both creates the approval record and pauses the flow until a response arrives. The flow run stays in a "waiting" state—Power Automate does not poll or spin; it resumes the run when the approver responds.

```
[Trigger]
    |
    ▼
[Start and wait for an approval]   ← Flow pauses here
    |
    ▼ (approval response arrives)
[Condition: Outcome == 'Approve']
```

**Key fields:**

| Field | What it controls |
|-------|-----------------|
| Approval type | One of the four patterns above |
| Title | Subject line shown to approvers |
| Assigned to | Email addresses of approvers (semicolon-separated) |
| Details | Body of the approval request (supports HTML) |
| Item link | Optional URL to the item being approved |
| Item link description | Label for the link |
| Request date | Defaults to trigger time |
| Enable notifications | Whether to send email/Teams alerts |

> **On screen:** In the action panel, you will see a dropdown at the top labeled **Approval type**. Selecting "Approve/Reject - Everyone must approve" expands the **Assigned to** field with a note confirming all approvers must respond.

### Create an Approval (Non-Blocking)

Separates creating the approval record from waiting for the response. The flow continues immediately after creating the approval. You use a second action, **Wait for an approval**, later in the flow to retrieve the response when needed.

```
[Create an approval]     ← Creates record, flow continues
    |
    ▼
[Do other work in parallel]
    |
    ▼
[Wait for an approval]   ← Flow pauses here until response
    |
    ▼
[Act on response]
```

**When to use:** When you want to send an acknowledgment email to the requestor immediately, then check back for the response later—without holding up the confirmation.

---

## Handling Approval Responses with Conditions

The **Start and wait for an approval** action outputs a response object. The key output is `Outcome`, which contains the approver's decision as a string.

For standard Approve/Reject approvals, `Outcome` is either `"Approve"` or `"Reject"`.

Build a Condition action directly below the approval action:

```
Condition: outputs('Start_and_wait_for_an_approval')['body/outcome'] is equal to 'Approve'
    |
    ├── YES branch: Send approval confirmation email, update record status
    |
    └── NO branch: Send rejection notification, log reason
```

> **On screen:** After adding the Condition action, click the left value field and select **Dynamic content** from the panel on the right. Scroll to the "Start and wait for an approval" section and select **Outcome**.

For multi-approver flows (Everyone must approve), the `Outcome` field reflects the final outcome after all responses. Individual responses are available in the `Responses` array output, which contains each approver's email, response, and timestamp.

### Extracting Response Details

The `Responses` output is an array. Use an **Apply to each** loop to access individual responses:

```
[Apply to each] items('Apply_to_each') from Responses
    |
    ▼
[Append to string variable]
    Approver: items('Apply_to_each')?['responder']
    Decision: items('Apply_to_each')?['approvalResponse']
    Comments: items('Apply_to_each')?['comments']
```

---

## Approval Email Formatting and Customization

The **Details** field in the approval action supports a limited set of HTML. This is what appears in the body of the approval email that approvers receive.

### Basic HTML Structure

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.html</span>
</div>

The following implementation builds on the approach above:

```html
<b>Expense Request Details</b><br>
<br>
Submitted by: <b>Priya Patel</b><br>
Department: Engineering<br>
Amount: <b>$450.00</b><br>
Category: Software License<br>
Business justification: Annual renewal for JetBrains IDEs used by the backend team.<br>
<br>
<a href="https://your-org.sharepoint.com/sites/Finance/Expenses/item123">View Receipt</a>
```

</div>

### What Renders and What Does Not

| Element | Renders in email | Renders in Teams |
|---------|-----------------|-----------------|
| `<b>`, `<i>` | Yes | Yes |
| `<br>` | Yes | Yes |
| `<a href>` | Yes | Partial |
| Tables | No | No |
| Images | No | No |
| CSS styles | No | No |

> **On screen:** The Details field in the Power Automate designer accepts both plain text and HTML. Type your HTML directly into the field or use dynamic content expressions to insert values from trigger outputs.

### Item Link

Always populate **Item link** and **Item link description**. These create a prominent "View Item" button in both the email and the Teams approval card, giving approvers direct access to the supporting document or request form before they respond.

---

## Building a Basic Expense Approval Flow: Step by Step

This walkthrough builds a flow that triggers when a new item is added to a SharePoint list called **Expense Requests**, routes it to the submitter's manager for approval, and updates the list item with the decision.

### Prerequisites

- SharePoint list: **Expense Requests** with columns: Title, Amount, Category, Justification, Status (Choice: Pending/Approved/Rejected), ManagerDecision (single line text)
- Users are licensed for Power Automate (Per User or Per Flow)
- The approver's email is known (use Office 365 Users connector to look up manager dynamically)

### Step 1: Create the Flow

1. Go to **make.powerautomate.com**
2. Select **+ Create** from the left navigation
3. Choose **Automated cloud flow**
4. Name the flow: `Expense Approval - SharePoint Trigger`
5. Search for and select **When an item is created** (SharePoint trigger)
6. Select your site and list, then click **Create**

> **On screen:** The designer opens with the trigger already on the canvas. You will see two dropdown fields: Site Address and List Name. Both are required before the flow can run.

### Step 2: Get the Submitter's Manager

1. Click **+ New step**
2. Search for **Office 365 Users**
3. Select **Get manager (V2)**
4. In the **User (UPN)** field, select dynamic content **Created by Email** from the trigger

> **On screen:** The **Get manager (V2)** action returns a user profile object. You will use the **Mail** property from this action as the approver's email address in the next step.

### Step 3: Add the Approval Action

1. Click **+ New step**
2. Search for **Approvals**
3. Select **Start and wait for an approval**
4. Set **Approval type** to **Approve/Reject - First to respond**
5. Set **Title** to: `Expense Approval: ` followed by dynamic content **Title** from the trigger
6. Set **Assigned to** to dynamic content **Mail** from the Get manager action
7. In **Details**, enter:

```
Submitted by: [Created By DisplayName]
Amount: $[Amount]
Category: [Category]
Justification: [Justification]
```

Replace bracketed placeholders with the corresponding dynamic content fields.

8. Set **Item link** to the URL of the SharePoint item (use the `Link to item` dynamic content from the trigger)
9. Set **Item link description** to `View Expense Request`

> **On screen:** After selecting dynamic content for the Title field, you will see a token chip representing that value. The field will display: `Expense Approval: ` followed by a blue chip labeled **Title**.

### Step 4: Handle the Decision

1. Click **+ New step**
2. Select **Condition**
3. In the left value field, select dynamic content **Outcome** (from the approval action)
4. Set the operator to **is equal to**
5. In the right value field, type `Approve`

### Step 5: YES Branch — Approved

Inside the YES branch, add **Update item** (SharePoint):

- Site: your site
- List: Expense Requests
- ID: dynamic content **ID** from the trigger
- Status: `Approved`
- ManagerDecision: `Approved by [Approvers Name] on [Response date]`

Then add **Send an email (V2)** (Office 365 Outlook) to notify the submitter of approval.

### Step 6: NO Branch — Rejected

Inside the NO branch, add **Update item** (SharePoint):

- Status: `Rejected`
- ManagerDecision: `Rejected — [Comments from the approval responses]`

Then add **Send an email (V2)** to notify the submitter of rejection, including the comments.

### Step 7: Test the Flow

1. Click **Save**, then click **Test** in the top right
2. Select **Manually** and click **Test**
3. Create a new item in the Expense Requests SharePoint list
4. Check the approver's inbox for the approval email
5. Approve or reject the request
6. Verify the SharePoint item status updates correctly

> **On screen:** During the test run, the flow run detail page shows each action with a green checkmark when successful. The **Start and wait for an approval** action shows a clock icon while waiting for the approver to respond.

---

## Common Pitfalls

<div class="callout-danger">

<strong>Danger:</strong> The pitfalls below are the most common mistakes practitioners make. Each one can silently degrade your results without obvious errors.

</div>

- **Approval goes to wrong person**: The **Assigned to** field requires email addresses, not display names. Use the Office 365 Users connector to resolve names to emails when the approver is dynamic.
- **Flow times out**: By default, flows time out after 30 days. For approvals that might take longer, use the **Reminder** settings or build an escalation step using a parallel branch.
- **Outcome string mismatch**: The Outcome value is case-sensitive in some contexts. Use `"Approve"` (not `"Approved"`). Check the approval type documentation for exact output strings for custom responses.
- **HTML not rendering**: Check that the Details field contains valid HTML. Nested tags and unclosed elements break rendering silently.

<div class="callout-warning">

<strong>Warning:</strong> - **Approval goes to wrong person**: The **Assigned to** field requires email addresses, not display names.

</div>

---

## Connections


<div class="callout-info">

<strong>Info:</strong> This section maps how this guide connects to the broader course. Use these links to navigate related material.

</div>

- **Builds on:** Module 05 — SharePoint triggers and list operations
- **Leads to:** Guide 02 — Adaptive Cards and multi-stage approval pipelines
- **Related to:** Module 04 — Conditions and branching logic

---


## Practice Questions

**Question 1 — Conceptual:** Based on the concepts in this guide, explain in your own words why the core technique matters and when you would choose it over alternatives.

**Question 2 — Application:** Sketch out how you would apply the main concept from this guide to a real-world dataset or problem you have encountered. What would you need to watch out for?


## Further Reading

- [Microsoft Docs: Approvals connector reference](https://learn.microsoft.com/en-us/connectors/approvals/)
- [Approvals Center in Microsoft Teams](https://support.microsoft.com/en-us/office/what-is-approvals-in-teams-f618e52c-0e01-4af3-857f-08e8cde55949)
- [Power Automate: Manage long-running approvals](https://learn.microsoft.com/en-us/power-automate/approval-email-customization)


---

## Cross-References

<a class="link-card" href="./01_approval_connector_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Interactive slide deck covering the key concepts with visual examples.</div>
</a>

<a class="link-card" href="../notebooks/01_adaptive_card_designer.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises and real data.</div>
</a>
