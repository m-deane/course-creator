# Copilot Agent Quick Reference

This reference covers the key terminology, API contracts, channel comparisons, security practices, and troubleshooting patterns for Copilot Studio agents connected to Power Automate flows.

---

## Copilot Studio Terminology

| Term | Definition |
|------|-----------|
| **Agent** | The top-level conversational AI application built in Copilot Studio. Formerly called a "bot" in Power Virtual Agents. |
| **Topic** | A conversation unit representing a single user intent. Contains trigger phrases, questions, conditions, actions, and messages. |
| **Trigger phrase** | A sample utterance that activates a topic. 5–10 representative phrases per topic are sufficient. |
| **Entity** | A named category of information extracted from user input. Can be prebuilt (date, email, number) or custom (priority levels, categories). |
| **Action** | A node in a topic that calls an external system. For Power Automate integration, the action type is "Flow". |
| **Variable** | A named value stored during a conversation. Topic variables are scoped to the current topic. Global variables persist across topics in the same conversation. |
| **System variable** | Built-in variables populated by Copilot Studio. Key ones: `System.User.Email`, `System.User.DisplayName`, `System.Conversation.Id`. |
| **Generative Answers** | Feature that lets the agent answer questions from connected knowledge sources (SharePoint, documents, web URLs) without explicit topic authoring. |
| **Knowledge source** | A connected data source used for Generative Answers. Common types: SharePoint site, uploaded document, public URL. |
| **Slot filling** | When an entity value is already present in the user's initial message, the agent skips the Question node for that entity and uses the extracted value directly. |
| **Redirect** | A topic node that passes control to another topic, optionally carrying variables from the current topic. |
| **Fallback topic** | The system topic that fires when no custom topic matches the user's input. Configure this to deliver a helpful message and offer Generative Answers. |
| **Test canvas** | The built-in test pane in Copilot Studio that simulates conversations without publishing. Does not provide authenticated system variables. |
| **Channel** | A publication target. Agents can publish to multiple channels simultaneously: Teams, Web, Microsoft 365 Copilot, custom app. |
| **Environment** | A Power Platform container for resources. Agents and flows must be in the same environment to connect to each other. |

---

## Power Automate Flow Requirements for Agent Actions

For a flow to appear in the Copilot Studio action picker and exchange data with the agent, it must meet all five of these requirements.

### Requirement 1: Correct Trigger

The flow must use the trigger **"When a flow is run from Copilot"** from the **Microsoft Copilot Studio** connector.

- Search for this trigger by name: type "Copilot" in the trigger search box
- Do not use "Manually trigger a flow" — that trigger type is not visible to Copilot Studio
- The flow must be in the same environment as the agent

### Requirement 2: Declared Input Parameters

In the trigger configuration, declare each input parameter the flow expects to receive from the agent:

| Parameter Type | Notes |
|----------------|-------|
| Text | Use for strings, email addresses, formatted IDs |
| Number | Use for numeric values (quantity, count, amount) |
| Boolean | Use for yes/no flags |
| Table | Use for passing arrays of structured data |

Parameter names must match exactly how they appear in Copilot Studio's action configuration. Names are case-sensitive.

### Requirement 3: Return Action at the End

The flow must include the action **"Return value(s) to Power Virtual Agents"** (also in the Microsoft Copilot Studio connector) as the final step.

- This action must be reached for the agent to receive any output
- If the flow errors before this action, the agent receives blank output variables
- Both the accepted AND rejected branches of conditional logic must reach a return action

### Requirement 4: Published and Healthy Connection

- The flow must be saved (not in draft with errors)
- All connection references in the flow must be valid — broken connections cause silent failures
- If you use a service account connection, the service account must retain access to all referenced resources

### Requirement 5: Same Environment

The agent and all flows it calls must be in the same Power Platform environment. Cross-environment flow calls are not supported.

---

## Input/Output Parameter Naming Convention

Use descriptive PascalCase names for parameters. The names appear in Copilot Studio's mapping UI — vague names like `Text1` make mapping error-prone.

**Good parameter names:**
- Inputs: `TicketTitle`, `Category`, `Priority`, `SubmitterEmail`, `SearchQuery`
- Outputs: `TicketID`, `AssignedTeam`, `EstimatedResponse`, `ArticleFound`, `ArticleTitle`

**Poor parameter names:**
- Inputs: `text`, `text_1`, `input1`, `value`
- Outputs: `result`, `output`, `data`

---

## Publishing Channels Comparison

| Channel | Authentication | User Identity Available | Recommended For |
|---------|---------------|------------------------|-----------------|
| **Microsoft Teams** | Azure AD SSO | Yes — `System.User.Email` and `System.User.DisplayName` populated | Internal employees, approval workflows, sensitive data |
| **Web channel (embedded)** | None by default | No — unless user provides it | Public FAQs, contact forms, external users |
| **Microsoft 365 Copilot** | Azure AD SSO | Yes | High-discovery internal use, M365 power users |
| **SharePoint** | Inherits site authentication | Partial — depends on site auth settings | Intranet portals, department sites |
| **Custom app (Direct Line)** | Custom — you control it | Configurable | ISV apps, custom portals, API testing |
| **Skype (deprecated)** | — | — | Not recommended for new deployments |

### Teams Channel Specifics

- Requires admin approval in the Teams admin center for org-wide deployment
- The agent appears as a Teams app that users can install or receive via policy
- Approval notifications from flows (Module 06) appear in the user's Teams Approvals Center
- `System.User.Email` is available as soon as the user opens the chat — no question needed

### Web Channel Specifics

- A script snippet is generated in Copilot Studio — embed in any HTML page
- No user authentication by default — treat all inputs as untrusted
- Do not expose employee data, financial data, or write operations that could be abused
- Optional: implement token-based authentication by passing a JWT in the embed script

### Microsoft 365 Copilot Specifics

- Requires the agent to be published as an M365 plugin
- Requires admin consent in the Microsoft admin center
- The agent appears as a plugin option in the M365 Copilot chat interface
- Users invoke it by typing "@YourAgentName" followed by their query

---

## Security Best Practices

### Service Account Pattern

Create a dedicated Azure AD account (non-personal, non-MFA-blocked) for all Power Automate connections used by agent flows.

- Name it: `svc-pa-helpdesk@contoso.com` (descriptive, clearly a service account)
- Assign it only the permissions required: SharePoint site member, Approvals user — not global admin
- Store its credentials in Azure Key Vault or the Power Platform admin center's secure connection store
- Document which service account is used for each flow in your team's connection registry
- Rotate the password annually and update the connection immediately after rotation

### Least Privilege for Connections

Create separate connections for read-only and write operations on the same data source:
- A flow that only reads KB articles should use a read-only SharePoint connection
- A flow that creates tickets needs a write-capable connection
- Never create a single high-privilege connection and use it for all flows

### DLP Policy Configuration

In the Power Platform admin center → Data Policies:
- Block connectors that are not needed by any agent flow in the environment
- Use the "Business" and "Non-Business" classification to separate trusted connectors
- Test DLP policies in a dev environment before applying to production
- Document the approved connector list and review it quarterly

### Environment Variables

Store all configurable values as environment variables, never as hardcoded strings inside flow actions.

| What to store | Why |
|---------------|-----|
| SharePoint site URLs | Change when moving between dev/test/prod environments |
| SharePoint list names | Change during naming refactors without editing flows |
| Approver email addresses | Change when staff turn over |
| API endpoints | Change when upstream services update |
| SLA values | Change when service level agreements are renegotiated |

Reference environment variables in flow actions by opening the expression editor and selecting them from the **Environment variables** section of dynamic content.

---

## Troubleshooting Common Agent Issues

### Agent Calls the Wrong Topic

**Symptom:** A message triggers a different topic than expected.

**Diagnosis:** Open Copilot Studio → Topics overview → check trigger phrases for overlap between topics.

**Fix:** Remove ambiguous phrases from the lower-priority topic. Longer, more specific phrases take precedence over shorter ones. Phrases that appear in multiple topics route to whichever topic was trained most recently with that phrase.

---

### Flow Does Not Appear in Action Picker

**Symptom:** When configuring a Call Action → Flow node in Copilot Studio, the flow is not in the list.

**Checklist:**
1. The flow uses "When a flow is run from Copilot" as its trigger (not any other trigger type)
2. The flow is saved with no errors (red exclamation icon = not saved correctly)
3. The flow is in the same Power Platform environment as the agent
4. The connection in the flow is valid and not broken (check the flow's connections page)
5. Your Copilot Studio account has access to view the flow (check flow sharing settings)

---

### Output Variables Are Blank After Flow Runs

**Symptom:** The agent calls the flow but all output variable values are empty.

**Diagnosis:** The flow ran but the "Return value(s) to Power Virtual Agents" action was not reached, or the output parameters were not mapped.

**Fix checklist:**
1. Open the flow run history in Power Automate → check if the return action executed
2. Verify both branches of any Condition action include a return action
3. Open the flow's return action — confirm each output parameter has a value mapped (not blank)
4. Check for runtime errors in the flow that caused it to terminate before the return action

---

### "Sorry, I didn't understand" Fires Unexpectedly

**Symptom:** The Fallback topic fires even though you have a topic that should match.

**Diagnosis:** The user's phrasing shares no words with any trigger phrase in any topic.

**Fix:**
- Add more diverse trigger phrases including common misspellings, abbreviations, and different word orders
- Enable Generative Answers as the fallback behavior (instead of the static Fallback topic) to handle phrasing diversity automatically
- Review the test canvas session transcript to see exactly what the user typed and compare with trigger phrases

---

### Flow Times Out During Agent Conversation

**Symptom:** The agent appears to hang for a long time (30+ seconds) after calling a flow action. Users see a "I'm having trouble connecting" message.

**Causes:**
- The flow calls a slow external API (>30 second response time)
- The flow waits for an approval that takes minutes/hours
- The flow is blocked by a SharePoint throttling limit

**Fixes:**
- For approvals: warn users in the message node before the action that the response may take time
- Set expectations: "I've submitted your request. You'll receive a Teams message when approved."
- For slow APIs: add a timeout on the HTTP connector (default is 60 seconds; reduce to 20 for better UX)
- For SharePoint throttling: add retry logic or spread load across time windows

---

### System.User.Email Is Empty

**Symptom:** The agent passes a blank email to flows, causing them to fail or create records with no submitter.

**Cause:** The channel does not provide authentication, or the test canvas is being used (which does not authenticate).

**Fix:**
- For production: ensure the agent is published to Teams (authenticated) not Web (unauthenticated) for flows that require user identity
- For testing: substitute a hardcoded test email in the topic variable mapping while testing in the test canvas, then switch back to `System.User.Email` before publishing
- For web channel: add a Question node that asks for the user's email address and save it to a variable; map that variable to the flow input instead of `System.User.Email`

---

### Topic Fails to Redirect to Another Topic

**Symptom:** The Redirect node fires but the target topic does not activate.

**Fix checklist:**
1. The target topic must be active (not disabled) — check the topic list for a toggle
2. Verify the redirect node is pointing to the correct topic by name
3. Variables only carry over if they share the same name between source and target topic — rename if needed
4. If the redirect is inside a Condition branch, check that the condition evaluates correctly before the redirect

---

## Official Microsoft Documentation Links

- [Copilot Studio documentation home](https://learn.microsoft.com/en-us/microsoft-copilot-studio/)
- [Create topics](https://learn.microsoft.com/en-us/microsoft-copilot-studio/authoring-create-edit-topics)
- [Use Power Automate flows as actions](https://learn.microsoft.com/en-us/microsoft-copilot-studio/advanced-flow)
- [Configure authentication](https://learn.microsoft.com/en-us/microsoft-copilot-studio/configuration-end-user-authentication)
- [Publish to Teams](https://learn.microsoft.com/en-us/microsoft-copilot-studio/publication-add-bot-to-microsoft-teams)
- [Publish to Microsoft 365 Copilot](https://learn.microsoft.com/en-us/microsoft-copilot-studio/publication-add-bot-to-microsoft-copilot)
- [Analytics overview](https://learn.microsoft.com/en-us/microsoft-copilot-studio/analytics-overview)
- [Direct Line API 3.0](https://learn.microsoft.com/en-us/azure/bot-service/rest-api/bot-framework-rest-direct-line-3-0-concepts)
- [Power Platform DLP policies](https://learn.microsoft.com/en-us/power-platform/admin/wp-data-loss-prevention)
- [Power Platform environment variables](https://learn.microsoft.com/en-us/power-apps/maker/data-platform/environmentvariables)
- [Copilot Studio known limitations](https://learn.microsoft.com/en-us/microsoft-copilot-studio/known-limitations)
