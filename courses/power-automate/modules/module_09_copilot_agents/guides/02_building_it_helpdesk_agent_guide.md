# Building an IT Helpdesk Copilot Agent: End-to-End Project

> **Reading time:** ~27 min | **Module:** 9 — Copilot Agents | **Prerequisites:** Module 8

## In Brief

This guide builds a complete IT helpdesk Copilot agent from scratch. By the end you will have a working agent connected to SharePoint lists and Power Automate flows that employees can use to search KB articles, create support tickets, check ticket status, and escalate critical issues—all through a natural language conversation in Microsoft Teams.

<div class="callout-insight">

<strong>Insight:</strong> Build the data layer first, then the flows, then the agent. Each layer depends on the one below it. Trying to build the agent before the flows are ready forces you to return and rewire things repeatedly.

</div>

<div class="callout-key">

<strong>Key Concept:</strong> This guide builds a complete IT helpdesk Copilot agent from scratch. By the end you will have a working agent connected to SharePoint lists and Power Automate flows that employees can use to search KB articles, create support tickets, check ticket status, and escalate critical issues—all through a natural language conversation in Microsoft Teams.

</div>


---

## Agent Design

### Persona

<div class="callout-insight">

<strong>Insight:</strong> ### Persona

- **Name:** Helpdesk Bot
- **Tone:** Professional, concise, action-oriented
- **Scope:** IT issues only — hardware, software, network, accounts
- **Limitations:** Does not provide general...

</div>


- **Name:** Helpdesk Bot
- **Tone:** Professional, concise, action-oriented
- **Scope:** IT issues only — hardware, software, network, accounts
- **Limitations:** Does not provide general business advice; escalates unknown or security-critical issues to a human agent

### Topics and Their Flows

| Topic | Trigger Examples | Backing Flow |
|-------|-----------------|--------------|
| Search Knowledge Base | "how do I fix VPN", "find an article about", "search KB" | Search KB Articles |
| Create Support Ticket | "submit a ticket", "report an issue", "I need help with" | Create Support Ticket |
| Check Ticket Status | "check my ticket", "what is the status of INC-", "update on my case" | Get Ticket Status |
| Escalate Issue | "escalate", "this is critical", "I need a manager" | Escalation Approval |

### Conversation Flow Overview

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

The following implementation builds on the approach above:

```text
User message
     │
     ▼
Topic matched by trigger phrase
     │
     ├──► Search KB Topic
     │         Questions: search term
     │         Action: Search KB Articles flow → returns title, summary, URL
     │         Message: displays results or offers to create ticket
     │
     ├──► Create Ticket Topic
     │         Questions: category, description, priority
     │         Action: Create Support Ticket flow → returns ticket ID
     │         Message: confirms ticket number and expected response time
     │
     ├──► Check Status Topic
     │         Questions: ticket ID
     │         Action: Get Ticket Status flow → returns status, assignee, last update
     │         Message: displays current status details
     │
     └──► Escalate Issue Topic
               Questions: reason for escalation, ticket ID (optional)
               Action: Escalation Approval flow → notifies IT manager, returns confirmation
               Message: confirms escalation and provides reference number
```

</div>

---

## Step 1: Create SharePoint Lists

Both SharePoint lists must be created before building any flows or topics.

<div class="callout-key">

<strong>Key Point:</strong> Both SharePoint lists must be created before building any flows or topics.

</div>


### List 1: IT Knowledge Base Articles

1. Navigate to your SharePoint site
2. Click **+ New** → **List**
3. Name the list: `IT_KB_Articles`
4. Add the following columns:

| Column Name | Type | Notes |
|-------------|------|-------|
| Title | Single line of text | Built-in; use as article headline |
| Category | Choice | Hardware, Software, Network, Account, Security |
| Summary | Multiple lines of text | 2–3 sentence overview shown in search results |
| ArticleBody | Multiple lines of text (rich text) | Full article content |
| Keywords | Single line of text | Comma-separated search terms |
| ArticleURL | Hyperlink or Picture | Link to full article if hosted externally |
| IsActive | Yes/No | Filter inactive articles from search results |

> **On screen:** In the SharePoint list settings panel, click **+ Add column** then select the column type from the flyout. For Choice columns, type each choice value separated by a newline in the choices field.

5. Add 3–5 sample KB articles covering common IT issues (VPN, password reset, printer setup, Teams installation, MFA enrollment) so search results return meaningful data during testing.

### List 2: IT Support Tickets

1. In the same SharePoint site, click **+ New** → **List**
2. Name the list: `IT_Support_Tickets`
3. Add the following columns:

| Column Name | Type | Notes |
|-------------|------|-------|
| Title | Single line of text | Auto-generated: "INC-{ID}: {short description}" |
| TicketID | Single line of text | Formatted ticket number, e.g., INC-2094 |
| Category | Choice | Hardware, Software, Network, Account, Other |
| Priority | Choice | Low, Medium, High, Critical |
| Description | Multiple lines of text | User-provided description of the issue |
| SubmitterEmail | Single line of text | Email from System.User.Email |
| SubmitterName | Single line of text | Display name of the submitter |
| Status | Choice | New, Assigned, In Progress, Pending User, Resolved, Closed |
| AssignedTeam | Single line of text | Routing based on category |
| AssignedTo | Person or Group | Specific technician (set manually or via assignment flow) |
| TicketCreated | Date and Time | Timestamp when ticket was created |
| LastUpdated | Date and Time | Timestamp of most recent update |
| EscalationStatus | Choice | Not Escalated, Escalation Requested, Escalated |
| ResolutionNotes | Multiple lines of text | Technician resolution summary |

> **On screen:** For the TicketID column, use type "Single line of text" rather than auto-number. The Create Ticket flow generates the formatted ID using the SharePoint item's numeric ID after creation.

---

## Step 2: Build the Power Automate Flows

All four flows use the **"When a flow is run from Copilot"** trigger from the Microsoft Copilot Studio connector.

<div class="callout-info">

<strong>Info:</strong> All four flows use the **"When a flow is run from Copilot"** trigger from the Microsoft Copilot Studio connector.

</div>


### Flow 1: Search KB Articles

**Purpose:** Query the IT_KB_Articles SharePoint list using a text search and return the top matching article.

**Input parameters:**
- `SearchQuery` (Text) — the user's search term

**Output parameters:**
- `ArticleFound` (Text) — "true" or "false"
- `ArticleTitle` (Text) — title of the best match
- `ArticleSummary` (Text) — 2–3 sentence summary
- `ArticleCategory` (Text) — category of the article
- `ArticleURL` (Text) — link to the full article

**Build steps:**

1. Create a new Instant cloud flow
2. Trigger: **When a flow is run from Copilot** → add input: `SearchQuery` (Text)
3. Add action: **Get items** (SharePoint)
   - Site: your IT site
   - List: IT_KB_Articles
   - Filter Query: `IsActive eq 1 and (substringof(Title, '${SearchQuery}') or substringof(Keywords, '${SearchQuery}') or substringof(Category, '${SearchQuery}'))`

   > **Note:** SharePoint OData filter syntax requires the `substringof` function for text search. Build the filter expression using the expression editor: `concat("IsActive eq 1 and (substringof('", triggerBody()['text'], "',Title) or substringof('", triggerBody()['text'], "',Keywords))")`

4. Add **Condition** action: `length(outputs('Get_items')?['body/value'])` is greater than `0`

5. **YES branch** (article found):
   - Add **Compose** action, input: `first(outputs('Get_items')?['body/value'])`
   - Add **Return value(s) to Power Virtual Agents**:
     - `ArticleFound`: `true`
     - `ArticleTitle`: `outputs('Compose')?['Title']`
     - `ArticleSummary`: `outputs('Compose')?['Summary']`
     - `ArticleCategory`: `outputs('Compose')?['Category']`
     - `ArticleURL`: `outputs('Compose')?['ArticleURL']`

6. **NO branch** (no article found):
   - Add **Return value(s) to Power Virtual Agents**:
     - `ArticleFound`: `false`
     - `ArticleTitle`: ` ` (single space — empty string not accepted)
     - `ArticleSummary`: `No articles found for that search term.`
     - `ArticleCategory`: ` `
     - `ArticleURL`: ` `

7. Name the flow: `Helpdesk - Search KB Articles`
8. Save

> **On screen:** In the filter query field of the Get items action, switch to the expression editor to build the OData filter using `concat()`. The dynamic content panel does not support OData filter syntax construction — the expression editor is required here.

---

### Flow 2: Create Support Ticket

**Purpose:** Write a new row to the IT_Support_Tickets SharePoint list and return the formatted ticket ID.

**Input parameters:**
- `TicketTitle` (Text) — short description of the issue
- `Category` (Text) — Hardware, Software, Network, Account, or Other
- `Priority` (Text) — Low, Medium, High, or Critical
- `SubmitterEmail` (Text) — email address of the user creating the ticket
- `SubmitterName` (Text) — display name of the user

**Output parameters:**
- `TicketID` (Text) — formatted ticket number, e.g., INC-2094
- `AssignedTeam` (Text) — team the ticket was routed to
- `EstimatedResponse` (Text) — human-readable SLA estimate

**Build steps:**

1. Create a new Instant cloud flow
2. Trigger: **When a flow is run from Copilot** → add all five input parameters
3. Add **Initialize variable** action:
   - Name: `assignedTeam`
   - Type: String
   - Value: expression `if(equals(triggerBody()['text_2'], 'Hardware'), 'Hardware Team', if(equals(triggerBody()['text_2'], 'Software'), 'Software Team', if(equals(triggerBody()['text_2'], 'Network'), 'Network Team', if(equals(triggerBody()['text_2'], 'Account'), 'Account Team', 'General IT'))))`

   > **Note:** `text_2` is the internal parameter name for the second Text parameter in the trigger. In the expression editor, reference parameters by their index name or use the dynamic content panel to insert them.

4. Add **Initialize variable** action:
   - Name: `slaEstimate`
   - Type: String
   - Value: expression `if(equals(triggerBody()['text_3'], 'Critical'), '2 hours', if(equals(triggerBody()['text_3'], 'High'), '4 hours', if(equals(triggerBody()['text_3'], 'Medium'), '1 business day', '3 business days')))`

5. Add **Create item** (SharePoint):
   - Site: your IT site
   - List: IT_Support_Tickets
   - Title: `concat('INC-', string(rand(1000, 9999)), ': ', triggerBody()['text'])` — see note below
   - Category: dynamic content `Category`
   - Priority: dynamic content `Priority`
   - Description: dynamic content `TicketTitle`
   - SubmitterEmail: dynamic content `SubmitterEmail`
   - SubmitterName: dynamic content `SubmitterName`
   - Status: `New`
   - AssignedTeam: variable `assignedTeam`
   - TicketCreated: `utcNow()`
   - LastUpdated: `utcNow()`
   - EscalationStatus: `Not Escalated`

   > **Note:** For the TicketID, create the item first (to get the auto-incremented SharePoint ID), then use a second **Update item** action to set the `TicketID` column to `concat('INC-', string(outputs('Create_item')?['body/ID']))`. This produces sequential IDs like INC-2094 based on the SharePoint item's internal integer ID.

6. Add **Update item** (SharePoint) to set the TicketID:
   - ID: `outputs('Create_item')?['body/ID']`
   - TicketID: `concat('INC-', string(outputs('Create_item')?['body/ID']))`

7. Add **Return value(s) to Power Virtual Agents**:
   - `TicketID`: `concat('INC-', string(outputs('Create_item')?['body/ID']))`
   - `AssignedTeam`: variable `assignedTeam`
   - `EstimatedResponse`: variable `slaEstimate`

8. Name the flow: `Helpdesk - Create Support Ticket`
9. Save

---

### Flow 3: Escalation Approval Flow

**Purpose:** Send an approval request to the IT manager when a user escalates their issue, then update the ticket record with the escalation outcome.

**Input parameters:**
- `TicketID` (Text) — existing ticket to escalate, or empty if no ticket exists
- `EscalationReason` (Text) — user's stated reason for escalation
- `SubmitterEmail` (Text) — user's email
- `SubmitterName` (Text) — user's display name

**Output parameters:**
- `EscalationConfirmation` (Text) — confirmation message for the user
- `EscalationReference` (Text) — reference number for the escalation request

**Build steps:**

1. Create a new Instant cloud flow
2. Trigger: **When a flow is run from Copilot** → add four input parameters
3. Add **Start and wait for an approval** (Approvals connector):
   - Approval type: Custom Responses — First to respond
   - Response options: Accept Escalation, Reject Escalation, Request More Information
   - Title: `concat('Escalation Request - ', triggerBody()['text'], ': ', triggerBody()['text_2'])`
   - Assigned to: your IT manager's email address (or use an environment variable)
   - Details: `concat('User: ', triggerBody()['text_4'], '\nEmail: ', triggerBody()['text_3'], '\nReason: ', triggerBody()['text_2'])`
   - Item link: if `TicketID` is not empty, construct the link to the SharePoint ticket item

4. Add **Condition** on the approval outcome:
   - `outputs('Start_and_wait_for_an_approval')?['body/outcome']` is equal to `Accept Escalation`

5. **YES branch** (accepted):
   - If TicketID is not empty, add **Update item** (SharePoint) to set `EscalationStatus` to `Escalated`
   - Add **Return value(s)**:
     - `EscalationConfirmation`: `Your escalation has been accepted. An IT manager will contact you within 30 minutes.`
     - `EscalationReference`: `concat('ESC-', utcNow('yyyyMMddHHmm'))`

6. **NO branch** (rejected or more info requested):
   - Add **Return value(s)**:
     - `EscalationConfirmation`: `concat('Your escalation request was reviewed. Response: ', outputs('Start_and_wait_for_an_approval')?['body/outcome'], '. Manager notes: ', first(outputs('Start_and_wait_for_an_approval')?['body/responses'])?['comments'])`
     - `EscalationReference`: `concat('ESC-', utcNow('yyyyMMddHHmm'))`

7. Name the flow: `Helpdesk - Escalation Approval`
8. Save

> **On screen:** The Approvals "Start and wait for an approval" action will pause the flow. In the agent conversation, the user will receive the confirmation message only after the IT manager responds. Set appropriate expectations in the message node: "I've submitted your escalation request. You'll hear back within 30 minutes once reviewed."

---

### Flow 4: Get Ticket Status

**Purpose:** Look up a ticket in the IT_Support_Tickets list by ticket ID and return its current status details.

**Input parameters:**
- `TicketID` (Text) — formatted ticket number, e.g., INC-2094

**Output parameters:**
- `TicketFound` (Text) — "true" or "false"
- `CurrentStatus` (Text) — current status value
- `AssignedTeam` (Text) — team handling the ticket
- `LastUpdated` (Text) — formatted date/time of last update
- `ResolutionNotes` (Text) — any resolution notes from the technician

**Build steps:**

1. Create a new Instant cloud flow
2. Trigger: **When a flow is run from Copilot** → add input: `TicketID` (Text)
3. Add **Get items** (SharePoint):
   - List: IT_Support_Tickets
   - Filter Query: `TicketID eq '${TicketID}'`
   - Top Count: 1

4. Add **Condition**: `length(outputs('Get_items')?['body/value'])` greater than `0`

5. **YES branch**:
   - **Compose** action: `first(outputs('Get_items')?['body/value'])`
   - **Return value(s)**:
     - `TicketFound`: `true`
     - `CurrentStatus`: `outputs('Compose')?['Status']`
     - `AssignedTeam`: `outputs('Compose')?['AssignedTeam']`
     - `LastUpdated`: `formatDateTime(outputs('Compose')?['LastUpdated'], 'MMMM d, yyyy h:mm tt')`
     - `ResolutionNotes`: `coalesce(outputs('Compose')?['ResolutionNotes'], 'No resolution notes yet.')`

6. **NO branch**:
   - **Return value(s)**:
     - `TicketFound`: `false`
     - `CurrentStatus`: `Ticket not found`
     - `AssignedTeam`: ` `
     - `LastUpdated`: ` `
     - `ResolutionNotes`: `No ticket found with ID: ${TicketID}. Please verify the ticket number.`

7. Name the flow: `Helpdesk - Get Ticket Status`
8. Save

---

## Step 3: Create the Copilot Agent in Copilot Studio

### Initial Setup

1. Navigate to **copilotstudio.microsoft.com**
2. Confirm you are in the correct environment (same as your Power Automate flows)
3. Click **+ New agent** → **Skip to configure**
4. Configure:
   - **Name:** `IT Helpdesk Assistant`
   - **Description:** `Your self-service IT support assistant. Search knowledge base articles, submit and track support tickets, and escalate critical issues.`
   - **Instructions:** `You are a professional IT helpdesk assistant for your organization's employees. Be concise and action-oriented. Only handle IT-related requests: hardware, software, network, and account issues. Always confirm before creating tickets. Do not provide general business advice. If a request is outside IT scope, politely redirect to the appropriate department.`
5. Click **Create**

> **On screen:** The agent canvas loads with the Topics panel on the left. You see four pre-existing system topics: Conversation Start, End of Conversation, Confirmed Success, Confirmed Failure, and Fallback. Do not delete these — they handle edge cases automatically.

### Configure Generative Answers

Before building topics, connect a knowledge source for generative fallback answers.

1. In the left panel, click **Knowledge**
2. Click **+ Add knowledge**
3. Select **SharePoint**
4. Enter the URL of your SharePoint site containing the IT_KB_Articles list
5. Click **Add**
6. Under **Generative answers** settings, set: when no topic is matched, search this knowledge source before showing the Fallback topic

> **On screen:** The Knowledge section shows a list of connected sources. A green checkmark appears next to the SharePoint source once indexing completes (typically 5–10 minutes). Until indexing is complete, generative answers will not draw from this source.

---

### Topic 1: Search Knowledge Base

1. Click **Topics** → **+ New topic** → **From blank**
2. Rename to: `Search Knowledge Base`
3. In the **Trigger** node, add trigger phrases:
   - `search knowledge base`
   - `find an article`
   - `how do I fix`
   - `look up`
   - `IT documentation`
   - `knowledge base`
   - `find help`
4. Add a **Message** node: `I can search our IT knowledge base for you.`
5. Add a **Question** node:
   - Question: `What topic or issue are you looking for?`
   - Identify: **User's entire response**
   - Save response as variable: `searchQuery`
6. Add a **Call an action** node → **Power Automate flows** → select `Helpdesk - Search KB Articles`
   - Map input: `SearchQuery` → `searchQuery`
   - Output variables are created automatically: `ArticleFound`, `ArticleTitle`, `ArticleSummary`, `ArticleCategory`, `ArticleURL`
7. Add a **Condition** node:
   - Condition: `Topic.ArticleFound` is equal to `true`
8. In the **True** branch, add a **Message** node:
   ```
   Here's what I found:

   **{Topic.ArticleTitle}**
   Category: {Topic.ArticleCategory}

   {Topic.ArticleSummary}

   [Read the full article]({Topic.ArticleURL})
   ```
9. Below the message, add a **Question** node: `Was this helpful, or would you like to create a support ticket?`
   - Quick replies: `That helped` / `Create a ticket`
   - Save response as variable: `helpfulResponse`
10. Add a **Condition** on `helpfulResponse` equals `Create a ticket` → redirect to the **Create Ticket** topic using a **Redirect** node
11. In the **False** branch, add a **Message** node: `I couldn't find any articles matching "{Topic.searchQuery}". Would you like me to create a support ticket instead?`
12. Add a **Question** node: quick replies `Yes, create a ticket` / `No thanks`
13. Condition on response → redirect to Create Ticket topic if yes
14. Click **Save**

> **On screen:** The Redirect node appears as a circular arrow icon. When you add one, a dropdown shows all other topics in the agent. Selecting a topic here passes control to that topic and carries over any variables that match by name.

---

### Topic 2: Create Support Ticket

1. Click **Topics** → **+ New topic** → **From blank**
2. Rename to: `Create Support Ticket`
3. Trigger phrases:
   - `submit a ticket`
   - `report an issue`
   - `I need help with`
   - `create a ticket`
   - `log an issue`
   - `something is broken`
4. Add a **Message** node: `I'll help you create a support ticket.`
5. Add a **Question** node:
   - Question: `What category best describes your issue?`
   - Identify: **User's choice of options**
   - Options: Hardware, Software, Network, Account, Other
   - Save as variable: `issueCategory`
6. Add a **Question** node:
   - Question: `Briefly describe the issue:`
   - Identify: **User's entire response**
   - Save as variable: `issueDescription`
7. Add a **Question** node:
   - Question: `What is the priority?`
   - Options: Low, Medium, High, Critical
   - Save as variable: `issuePriority`
8. Add a **Message** node to confirm before submitting:
   ```
   I'll create a ticket with:
   - Category: {Topic.issueCategory}
   - Priority: {Topic.issuePriority}
   - Description: {Topic.issueDescription}

   Shall I proceed?
   ```
9. Add a **Question** node: quick replies `Yes, submit` / `No, let me change something`
   - Condition on `No` → loop back to the category question (use a Redirect to this same topic or a Go to step node)
10. Add a **Call an action** node → `Helpdesk - Create Support Ticket`
    - Map: `TicketTitle` → `issueDescription`, `Category` → `issueCategory`, `Priority` → `issuePriority`, `SubmitterEmail` → `System.User.Email`, `SubmitterName` → `System.User.DisplayName`
11. Add a **Message** node:
    ```
    Your ticket has been created.

    Ticket ID: **{Topic.TicketID}**
    Assigned to: {Topic.AssignedTeam}
    Expected response: {Topic.EstimatedResponse}

    You'll receive email updates as your ticket progresses.
    ```
12. Click **Save**

> **On screen:** `System.User.Email` and `System.User.DisplayName` are system variables populated automatically when the agent runs in a channel with authentication (Teams). They appear in the variable picker under the **System** section.

---

### Topic 3: Check Ticket Status

1. Click **Topics** → **+ New topic** → **From blank**
2. Rename to: `Check Ticket Status`
3. Trigger phrases:
   - `check my ticket`
   - `ticket status`
   - `what is the status`
   - `update on my case`
   - `INC-`
   - `follow up on ticket`
4. Add a **Question** node:
   - Question: `Please enter your ticket ID (for example: INC-2094):`
   - Identify: **User's entire response**
   - Save as variable: `ticketIDInput`
5. Add a **Call an action** → `Helpdesk - Get Ticket Status`
   - Map: `TicketID` → `ticketIDInput`
6. Add a **Condition**: `Topic.TicketFound` equals `true`
7. **True branch** — Message node:
   ```
   Status for ticket **{Topic.ticketIDInput}**:

   Status: {Topic.CurrentStatus}
   Assigned team: {Topic.AssignedTeam}
   Last updated: {Topic.LastUpdated}

   {Topic.ResolutionNotes}
   ```
8. Add a follow-up **Question** node in the True branch: `Is there anything else I can help you with?`
   - Quick replies: `Escalate this ticket` / `No, I'm good`
   - Condition on `Escalate this ticket` → Redirect to Escalate Issue topic
9. **False branch** — Message node: `I couldn't find ticket "{Topic.ticketIDInput}". Please double-check the ID and try again. Ticket IDs start with INC- followed by numbers.`
10. Click **Save**

---

### Topic 4: Escalate Issue

1. Click **Topics** → **+ New topic** → **From blank**
2. Rename to: `Escalate Issue`
3. Trigger phrases:
   - `escalate`
   - `I need a manager`
   - `this is critical`
   - `urgent help needed`
   - `speak to someone`
   - `escalate my ticket`
4. Add a **Message** node: `I can escalate your issue to an IT manager. They will review and contact you directly.`
5. Add a **Question** node:
   - Question: `Do you have an existing ticket ID for this issue?`
   - Quick replies: `Yes` / `No`
   - Save as variable: `hasTicketID`
6. Add a **Condition** on `hasTicketID` equals `Yes`
7. In the **Yes** branch, add a **Question** node:
   - Question: `Please enter your ticket ID:`
   - Save as variable: `escalationTicketID`
8. In the **No** branch, set variable `escalationTicketID` to ` ` (blank) using a **Set variable** node
9. After the condition, add a **Question** node:
   - Question: `Briefly explain why you need to escalate:`
   - Identify: **User's entire response**
   - Save as variable: `escalationReason`
10. Add a **Message** node: `Submitting your escalation request now. Please wait a moment while an IT manager reviews it.`
11. Add a **Call an action** → `Helpdesk - Escalation Approval`
    - Map: `TicketID` → `escalationTicketID`, `EscalationReason` → `escalationReason`, `SubmitterEmail` → `System.User.Email`, `SubmitterName` → `System.User.DisplayName`
12. Add a **Message** node:
    ```
    {Topic.EscalationConfirmation}

    Reference number: {Topic.EscalationReference}
    ```
13. Click **Save**

---

## Step 4: Test the Agent in the Test Canvas

### Testing Each Topic

After saving all four topics, test each one in the Test canvas on the right side of the Copilot Studio screen.

1. Click **Test your agent** if the pane is collapsed
2. Click the refresh icon at the top of the test pane to start a new conversation

**Test Search Knowledge Base:**
- Type: `how do I connect to VPN`
- Verify the agent asks for a search query
- Type a term that matches a KB article keyword
- Verify the article title and summary appear in the response

**Test Create Support Ticket:**
- Type: `I need to submit a ticket`
- Walk through all prompts (category, description, priority)
- Verify confirmation message appears with a ticket ID
- Check the IT_Support_Tickets SharePoint list to confirm the row was created

**Test Check Ticket Status:**
- Type: `check my ticket status`
- Enter the ticket ID created in the previous test
- Verify the status details appear correctly

**Test Escalate Issue:**
- Type: `I need to escalate`
- Complete the escalation flow
- Check the approver's email or Teams Approvals Center for the approval request
- Respond to the approval and verify the agent delivers the confirmation

> **On screen:** In the test canvas, each agent response shows a small label below it indicating which topic handled the turn. If the wrong topic fires, click the label to open that topic on the canvas and review its trigger phrases for conflicts with other topics.

### Fixing Topic Conflicts

If the wrong topic activates during testing:

1. Click **Topics** in the left panel to see the topics list
2. Look for trigger phrase overlap between topics
3. The topic with more specific or longer trigger phrases takes precedence
4. Remove ambiguous phrases from the lower-priority topic
5. Retrain the agent by clicking **Save** on the modified topic

---

## Step 5: Publish to Teams and Microsoft 365 Copilot

### Publish to Microsoft Teams

1. In the top right of Copilot Studio, click **Publish**
2. Click **Publish** in the confirmation dialog
3. After publishing completes, click **Go to channels**
4. Select **Microsoft Teams**
5. Click **Turn on Teams**
6. Click **Open bot in Teams** to test the agent directly in Teams
7. To make the agent available to others: go to **Teams admin center** → **Teams apps** → **Manage apps** → find your agent → change availability to your target audience

> **On screen:** The Channels page shows all available publication targets. Each channel has an independent status: the agent can be published to Teams but not yet to Web. The green dot next to a channel means it is live.

### Publish to Microsoft 365 Copilot

1. On the Channels page, select **Microsoft Copilot (M365)**
2. Follow the authentication setup wizard — this requires admin consent for the Microsoft 365 Copilot integration
3. Once connected, the agent appears as a plugin in Microsoft 365 Copilot chat

> **On screen:** The M365 Copilot channel requires that both the agent and the Microsoft 365 Copilot integration are in the same tenant and that the Copilot Studio agent is enabled as an M365 plugin in the Microsoft admin center.

---

## DLP Policies and Environment Variables

### Environment Variables for This Agent

Create environment variables in the Power Platform admin center (or within a solution) to avoid hardcoding values in flow actions.

| Variable Name | Type | Example Value |
|---------------|------|---------------|
| `SP_IT_SITE_URL` | String | `https://your-org.sharepoint.com/sites/IT` |
| `SP_KB_LIST_NAME` | String | `IT_KB_Articles` |
| `SP_TICKETS_LIST_NAME` | String | `IT_Support_Tickets` |
| `IT_MANAGER_EMAIL` | String | `itmanager@your-org.com` |
| `ESCALATION_SLA_CRITICAL` | String | `2 hours` |

Use these environment variables inside flow actions by referencing them with the **Get environment variable** action from the Environment Variables connector, or by building the flow inside a solution where environment variables are accessible as first-class values.

### DLP Policy Recommendations

Work with your Power Platform administrator to ensure the DLP policy for the environment allows:
- SharePoint connector (required for all four flows)
- Approvals connector (required for escalation flow)
- Microsoft Copilot Studio connector (required for all flows)
- Office 365 Users connector (optional — for manager lookup)

Connectors not on the allowed list will cause flows to fail silently from the agent's perspective — the Call Action node will return empty outputs without an obvious error.

---

## Monitoring and Analytics

### Built-in Copilot Studio Analytics

In Copilot Studio, click **Analytics** in the left navigation to access:

- **Overview:** Total sessions, engaged sessions (user sent more than one message), resolution rate
- **Customer satisfaction:** If you configure the end-of-conversation feedback prompt, CSAT scores appear here
- **Sessions:** Full conversation transcripts — click any session to replay it
- **Topic usage:** Which topics are triggered most frequently, escalation rates per topic, abandonment rates

**Key metrics to track weekly:**
- Escalation rate per topic (high rate suggests the flow returns poor quality results)
- Abandonment rate (users who stop mid-conversation — indicates confusing prompts or slow responses)
- Session resolution rate (target >60% for a mature agent)

### Flow Run History

In Power Automate, click **My flows** and select each helpdesk flow to see:
- Run history with success/failure per run
- Individual run details showing which action failed and the error message
- Performance metrics (execution duration)

For production monitoring, set up a separate monitoring flow that queries the flow run history via the Power Automate Management connector and sends a daily digest to the IT operations team.

---

## Common Pitfalls

<div class="callout-danger">

<strong>Danger:</strong> The pitfalls below are the most common mistakes practitioners make. Each one can silently degrade your results without obvious errors.

</div>

- **Flow not appearing in Copilot Studio action picker:** Confirm the flow uses the "When a flow is run from Copilot" trigger and is saved without errors. Check that you are in the same environment in both tools.
- **Output variables are blank after the action node:** The flow's "Return value(s) to Power Virtual Agents" action must be reached. Check the flow run history to verify the return action executed.
- **SharePoint filter query returning no results:** OData filter syntax is strict — spaces in field names must be encoded, and the `substringof` function requires the field name as the second argument. Test queries in the SharePoint REST API browser before using in flows.
- **Escalation approval blocking the agent indefinitely:** The "Start and wait for an approval" action pauses the flow until the approver responds. Set a reasonable timeout by wrapping the approval in a **Do until** loop with a time-based exit condition.
- **System.User.Email is empty in test canvas:** The test canvas does not authenticate as a real user. Hard-code a test email in the mapping while testing, then switch back to the system variable before publishing.

<div class="callout-warning">

<strong>Warning:</strong> - **Flow not appearing in Copilot Studio action picker:** Confirm the flow uses the "When a flow is run from Copilot" trigger and is saved without errors.

</div>

---

## Connections


<div class="callout-info">

<strong>Info:</strong> This section maps how this guide connects to the broader course. Use these links to navigate related material.

</div>

- **Builds on:** Guide 01 — Copilot agents overview and architecture
- **Builds on:** Module 06 — Approvals connector (used in Flow 3)
- **Builds on:** Module 05 — SharePoint list operations (used in all four flows)
- **Related to:** Module 04 — Conditions and branching (topic condition nodes)

---


## Practice Questions

**Question 1 — Conceptual:** Based on the concepts in this guide, explain in your own words why the core technique matters and when you would choose it over alternatives.

**Question 2 — Application:** Sketch out how you would apply the main concept from this guide to a real-world dataset or problem you have encountered. What would you need to watch out for?


## Further Reading

- [Microsoft Copilot Studio: Create a topic](https://learn.microsoft.com/en-us/microsoft-copilot-studio/authoring-create-edit-topics)
- [Call Power Automate flows as actions](https://learn.microsoft.com/en-us/microsoft-copilot-studio/advanced-flow)
- [Copilot Studio analytics overview](https://learn.microsoft.com/en-us/microsoft-copilot-studio/analytics-overview)
- [Publish to Microsoft Teams](https://learn.microsoft.com/en-us/microsoft-copilot-studio/publication-add-bot-to-microsoft-teams)
- [Power Platform environment variables](https://learn.microsoft.com/en-us/power-apps/maker/data-platform/environmentvariables)


---

## Cross-References

<a class="link-card" href="./02_building_it_helpdesk_agent_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Interactive slide deck covering the key concepts with visual examples.</div>
</a>

<a class="link-card" href="../notebooks/01_copilot_agent_api.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises and real data.</div>
</a>
