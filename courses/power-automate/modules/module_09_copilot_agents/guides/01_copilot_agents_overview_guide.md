# Copilot Agents Overview: Conversational Automation with Copilot Studio

## In Brief

Copilot agents are conversational AI assistants built in Microsoft Copilot Studio. They understand natural language, maintain multi-turn conversations, and execute actions—including Power Automate flows—on behalf of users. Where a standalone flow responds to events, a Copilot agent responds to people.

> **Key Insight:** A Copilot agent is not a chatbot that talks. It is an automation layer with a conversational interface. Every topic in the agent is a structured decision tree that can call Power Automate flows, query data sources, and return formatted answers—all within a natural dialogue.

---

## Why Copilot Agents Exist

Standalone flows are triggered by events: a file is created, a form is submitted, a schedule fires. They are invisible to end users. Copilot agents fill the gap where users need to initiate automation themselves, but the request is unpredictable in wording and context.

Before Copilot Studio, teams built these interfaces with:
- Custom bot frameworks requiring pro-developer skills
- Email-based request systems (no consistency, no audit trail)
- SharePoint forms (rigid, no NLU, no conversational branching)

Copilot Studio (formerly Power Virtual Agents) provides a low-code environment where a maker can define conversational topics, wire in Power Automate flows as backend actions, and publish the agent to Teams, web, or Microsoft 365 Copilot—all without writing NLU training pipelines.

---

## Copilot Agent vs. Standalone Flow: When to Use Which

Use this decision matrix to choose the right tool for a given scenario.

```
Trigger type         ──► Event (file, schedule, form)  ──► Standalone flow
                     ──► User initiating on demand     ──► Copilot agent

Request structure    ──► Always the same fields         ──► Standalone flow
                     ──► Variable wording, context      ──► Copilot agent

User interaction     ──► None needed                    ──► Standalone flow
                     ──► Clarifying questions needed    ──► Copilot agent

Output               ──► System action (write, notify)  ──► Both
                     ──► Conversational answer to user  ──► Copilot agent
```

**Use a standalone flow when:** An invoice arrives by email and must be processed automatically, no human drives the trigger.

**Use a Copilot agent when:** A user types "I need to submit an IT ticket for a broken printer" and the agent needs to ask clarifying questions, search for related KB articles, and then create the ticket—all in one conversation.

**Use both together:** The agent handles the conversation; Power Automate flows handle the backend work. This is the most common production pattern.

---

## Copilot Studio Overview

Copilot Studio is a standalone web application at [copilotstudio.microsoft.com](https://copilotstudio.microsoft.com). It provides the following building blocks.

### Topics

A topic is a conversation unit. It defines:
- **Trigger phrases:** Natural language examples that activate the topic
- **Conversation nodes:** Questions the agent asks, messages it sends, conditions it evaluates, actions it calls
- **Variables:** Values collected during the conversation that are passed to actions

Every agent starts with built-in system topics (Greeting, Fallback, Escalate to Agent) and you add custom topics for each use case.

### Entities

Entities are named categories of information that the agent extracts from user input. Examples:
- **Prebuilt:** Date/time, number, email address, phone number, country/region
- **Custom:** Ticket priority levels (Low, Medium, High, Critical), IT categories (Hardware, Software, Network, Account)

When a user says "My laptop screen is broken and it's urgent", the agent extracts the entity values `category=Hardware` and `priority=Critical` automatically if you configure those custom entities.

### Actions

Actions connect the agent to external systems. In the context of this course:
- **Power Automate flows** — the primary backend mechanism
- **HTTP requests** — direct REST API calls
- **Connector actions** — Dataverse queries, SharePoint queries, Microsoft Graph calls

When an agent topic reaches a node configured as a Power Automate action, the flow runs in real time, returns output values, and the conversation continues with those values available as variables.

### Generative Answers

Copilot Studio integrates with Azure OpenAI to provide answers from documents and web content without requiring explicit topic authoring. When a user asks a question that no topic matches, the agent can query a connected knowledge source (SharePoint site, uploaded documents, public URLs) and return a synthesized answer.

This is particularly powerful for IT helpdesk scenarios where a vast KB cannot be fully encoded into topics but can be indexed as a knowledge source.

---

## Architecture: Agent to Data

The full architecture from user input to data and back:

```
User
  │
  ▼
Copilot Agent (Copilot Studio)
  │  Natural language input
  ▼
Topic Matching
  │  Trigger phrase recognition → correct topic selected
  ▼
Conversation Nodes
  │  Questions, variable collection, condition evaluation
  ▼
Power Automate Flow Action
  │  Agent calls flow with input parameters
  ▼
Flow Steps
  │  SharePoint / SQL / HTTP / Graph / Approvals connectors
  ▼
Data Sources
  │  SharePoint lists, Dataverse tables, external APIs
  ▼
Flow Returns Output
  │  Structured response back to the agent
  ▼
Agent Formats Response
  │  Message node renders answer using output variables
  ▼
User receives answer
```

### Key Data Contract Between Agent and Flow

When a Power Automate flow is used as an agent action, it must declare its inputs and outputs explicitly using the **HTTP request trigger** or the **Run a flow from Copilot** trigger.

**Flow inputs** become the parameters the agent passes when calling the flow.
**Flow outputs** become the variables the agent receives and can use in subsequent conversation nodes.

This contract is set at design time in both Copilot Studio (action configuration) and Power Automate (trigger schema and return values).

---

## Agent Capabilities: Natural Language Understanding

Copilot Studio agents do not perform raw intent classification. They use a combination of:

1. **Exact match** — trigger phrases you write are matched verbatim
2. **Fuzzy match** — similar phrasing that shares key words triggers the topic
3. **Entity extraction** — within a matched topic, slot values (entities) are pulled from the user's message
4. **Generative AI fallback** — if no topic matches, generative answers search connected knowledge sources

This means the agent does not require exhaustive NLU training. You write representative trigger phrases (5–10 per topic is sufficient) and the platform generalises from them.

### Multi-Turn Conversations

A single topic can span many conversational turns. The flow of a multi-turn topic:

```
Agent: "I can help create an IT ticket. What type of issue are you experiencing?"
User:  "My email isn't syncing."

Agent: "How long has this been happening?"
User:  "Since this morning."

Agent: "What's your urgency level — Low, Medium, High, or Critical?"
User:  "High."

Agent: [calls Power Automate flow with category=Email, duration='since morning', priority=High]
Agent: "Your ticket INC-2094 has been created and assigned to the email team.
        Estimated response: 2 hours."
```

Variables collected across the conversation are in scope for the entire topic and are available to pass to flow actions.

---

## Connecting Power Automate Flows as Agent Actions

### Flow Requirements

For a flow to be callable from a Copilot agent it must:

1. Use the trigger **"When a flow is run from Copilot"** (under Microsoft Copilot Studio connector in Power Automate)
2. Declare typed input parameters in the trigger schema
3. Include a **"Return value(s) to Power Virtual Agents"** action (also in the Copilot Studio connector) at the end of the flow
4. Be published (not in draft)
5. Be owned by or shared with the service account running the agent

### Step-by-Step: Adding a Flow Action to a Topic

**In Power Automate:**

1. Create a new Instant cloud flow
2. For the trigger, search **Microsoft Copilot Studio** and select **When a flow is run from Copilot**
3. In the trigger, add input parameters: define each parameter with a name and type (Text, Number, Boolean, or Table)
4. Build your flow logic (SharePoint queries, approvals, etc.)
5. Add the action **Return value(s) to Power Virtual Agents** at the end
6. Define output parameters and map them to flow output values
7. Save and name the flow clearly (the name appears in Copilot Studio)

> **On screen:** When you select the "When a flow is run from Copilot" trigger, the action card shows a section called **Input**. Click **+ Add an input** to define each parameter. For a "Create ticket" flow you might add: `TicketTitle` (Text), `Category` (Text), `Priority` (Text), `SubmitterEmail` (Text).

**In Copilot Studio:**

1. Open the topic where the flow should be called
2. Add a node by clicking **+** and selecting **Call an action**
3. Select **Flow** and find your flow by name
4. Map agent topic variables to flow input parameters
5. Map flow output values to new or existing topic variables
6. Use those output variables in subsequent message nodes

> **On screen:** When you select **Call an action → Flow**, Copilot Studio displays a panel showing the flow's declared input parameters on the left and output parameters on the right. Map each input by selecting the variable that was collected earlier in the conversation.

---

## Step-by-Step: Creating a Basic Copilot Agent

### Step 1: Navigate to Copilot Studio

1. Open a browser and go to **copilotstudio.microsoft.com**
2. Sign in with your Microsoft 365 work account
3. Select your **Environment** (top right corner) — use the same environment as your Power Automate flows
4. You land on the Home page showing your existing agents

> **On screen:** The home page shows a grid of existing agents with their names, last-modified dates, and channel publication status. A blue **+ New agent** button appears in the top right.

### Step 2: Create a New Agent

1. Click **+ New agent**
2. You are presented with two options: **Create with Copilot** (AI-assisted) and **Skip to configure**
3. Select **Skip to configure** for full manual control
4. Fill in:
   - **Name:** `IT Helpdesk Assistant`
   - **Description:** `Helps employees search IT knowledge base articles, create support tickets, and check ticket status`
   - **Instructions:** `You are an IT support assistant. Be concise and professional. Always confirm before creating tickets. Route hardware issues to the hardware team, software issues to the software team.`
5. Click **Create**

> **On screen:** After clicking Create, the agent canvas loads. You see the **Topics** panel on the left, a **Test your agent** pane on the right, and a toolbar at the top with Save, Publish, and Settings buttons.

### Step 3: Add a Custom Topic

1. In the left panel, click **Topics**
2. Click **+ New topic** → **From blank**
3. Click the topic title at the top and rename it to **Search Knowledge Base**

> **On screen:** The topic editor shows a canvas with a single starting node labeled **Trigger** containing a text field for trigger phrases.

4. In the **Trigger** node, add these trigger phrases:
   - `search knowledge base`
   - `find an article`
   - `how do I fix`
   - `look up IT documentation`
   - `knowledge base`
5. Add a **Message** node: type `I'll search the knowledge base for you. What topic are you looking for?`
6. Add a **Question** node to collect the search term:
   - Question text: `What topic would you like to search for?`
   - Save response to variable: `searchQuery` (type: Text)
7. Add a **Call an action** node → **Flow** → select your `Search KB Articles` flow
8. Map: `SearchQuery` input → `searchQuery` variable
9. Add a **Message** node using the flow's output: `Here's what I found: {Topic 1: outputVar1, Summary: outputVar2}`
10. Click **Save**

> **On screen:** Each node on the canvas is connected by a vertical line with arrow. Conditions branch into parallel paths shown side-by-side. The Test pane on the right lets you simulate a conversation immediately without publishing.

### Step 4: Test in the Test Canvas

1. In the **Test your agent** pane on the right, type: `I need to find an article`
2. The agent responds with the message from the Trigger path
3. Type a search query, e.g., `VPN connection issues`
4. The agent calls the flow and returns results
5. Verify the output matches expected KB article content

> **On screen:** The test canvas shows each conversation turn with the agent's responses in speech bubbles. Underneath each agent turn, a small icon labeled "Topic: Search Knowledge Base" shows which topic handled the turn. Clicking the icon opens the topic canvas with the active node highlighted.

---

## Authentication and Security Considerations

### Service Account and Permissions

Copilot agents run flow actions under the credentials of the connection configured in the flow, not the end user's credentials. This means:

- The connection account must have permissions to the SharePoint sites, Dataverse tables, or other resources the flow accesses
- Use a dedicated **service account** (a non-personal Azure AD account) for flow connections in production
- Document which service account is used for audit purposes

### Channel Authentication

When publishing the agent to Teams:
- Users authenticate via Azure AD SSO — the agent knows who is talking
- User identity (email, name) is available as system variables: `System.User.Email`, `System.User.DisplayName`
- Pass these to flows to stamp tickets with the actual user's identity

When publishing to an unauthenticated web channel:
- No identity is known unless the user provides it conversationally
- Do not expose sensitive data or perform destructive actions in unauthenticated channels

### DLP and Environment Variables

- Apply **Data Loss Prevention (DLP) policies** in the Power Platform admin center to control which connectors the flows (and therefore the agent) can use
- Use **environment variables** in Power Automate for all configurable values (SharePoint site URLs, list names, API endpoints) so they are easy to change across environments without editing flow definitions
- Never hardcode site URLs or sensitive values inside flow actions

### Scope of Access

Flows called from Copilot agents should operate with **least-privilege connections**. A flow that reads KB articles does not need a connection with write access to the same list. Create separate connections with appropriate permission scopes for read-only and write operations.

---

## Common Pitfalls

- **Flow trigger mismatch:** If the flow does not use the **"When a flow is run from Copilot"** trigger, it will not appear in the Copilot Studio action picker.
- **Environment mismatch:** The agent and its flows must be in the same Power Platform environment. Cross-environment calls are not supported.
- **Output parameters not mapped:** If the flow's "Return value(s)" action does not declare output parameters, no data returns to the agent and the output variables will be blank.
- **Unpublished flows:** Flows must be saved and the connection must be healthy (no broken credentials) for the agent to call them reliably.
- **Topic trigger phrase overlap:** If two topics share identical or very similar trigger phrases, the agent may route to the wrong topic. Review all topics for phrase conflicts in the Topics overview.

---

## Connections

- **Builds on:** Module 08 — Copilot in Power Automate (AI-assisted flow building)
- **Leads to:** Guide 02 — Building an end-to-end IT Helpdesk Agent
- **Related to:** Module 06 — Approval flows (used inside agent action flows)

---

## Further Reading

- [Microsoft Copilot Studio documentation](https://learn.microsoft.com/en-us/microsoft-copilot-studio/)
- [Use a flow as an action in Copilot Studio](https://learn.microsoft.com/en-us/microsoft-copilot-studio/advanced-flow)
- [Publish a Copilot agent to Microsoft Teams](https://learn.microsoft.com/en-us/microsoft-copilot-studio/publication-add-bot-to-microsoft-teams)
- [Authentication in Copilot Studio](https://learn.microsoft.com/en-us/microsoft-copilot-studio/configuration-end-user-authentication)
