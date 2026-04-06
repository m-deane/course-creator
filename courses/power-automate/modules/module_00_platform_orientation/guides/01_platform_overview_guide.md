# Power Automate: Platform Overview

> **Reading time:** ~10 min | **Module:** 0 — Platform Orientation | **Prerequisites:** None

## In Brief

Power Automate is Microsoft's cloud-based workflow automation service that lets you create automated processes — called **flows** — between applications and services without writing traditional code. It sits inside the broader **Microsoft Power Platform**, a suite of low-code tools designed to help organizations build solutions faster than conventional software development allows.

<div class="callout-insight">

<strong>Insight:</strong> Power Automate is not a single tool — it is a runtime, a connector hub, and a process orchestrator rolled into one. Understanding where it fits in the Power Platform ecosystem determines which automations are practical and which require a different tool.

</div>

<div class="callout-key">

<strong>Key Concept:</strong> Power Automate is Microsoft's cloud-based workflow automation service that lets you create automated processes — called **flows** — between applications and services without writing traditional code. It sits inside the broader **Microsoft Power Platform**, a suite of low-code tools designed to help organizations build solutions faster than conventional software development allows.

</div>


---

## Where Power Automate Fits: The Power Platform Ecosystem

Microsoft Power Platform has five primary products that share a common data layer:

<div class="callout-insight">

<strong>Insight:</strong> Microsoft Power Platform has five primary products that share a common data layer:

| Product | Primary Purpose |
|---|---|
| **Power Apps** | Build custom web and mobile applications |
| **Power Auto...

</div>


| Product | Primary Purpose |
|---|---|
| **Power Apps** | Build custom web and mobile applications |
| **Power Automate** | Automate workflows and processes |
| **Power BI** | Visualize and analyze data |
| **Copilot Studio** | Build AI-powered conversational agents |
| **Power Pages** | Create external-facing websites |

All five products connect to **Dataverse** — Microsoft's cloud-scale data platform — and share a common authentication model, environment structure, and connector library.

```
Power Platform
├── Power Apps         (build)
├── Power Automate     (automate)
├── Power BI           (analyze)
├── Copilot Studio     (converse)
├── Power Pages        (publish)
└── Dataverse          (store — shared by all)
```

Power Automate occupies the **automation layer**: when something happens in one system, Power Automate can react to it, process data, and take action in another system — all without human intervention.

### How the Products Interconnect

A common pattern in enterprise deployments:

1. A **Power App** presents a form to a user (e.g., a purchase request)
2. Submitting the form writes a record to **Dataverse**
3. A **Power Automate** flow detects the new record, routes an approval email, and updates a SharePoint list
4. **Power BI** reads the SharePoint list and updates an approval-rate dashboard
5. A **Copilot Studio** agent answers employee questions about approval status

Power Automate is often the connective tissue between tools that would otherwise be silos.

---

## Core Terminology

### Flow

<div class="callout-key">

<strong>Key Point:</strong> ### Flow

A **flow** is the fundamental unit in Power Automate.

</div>


A **flow** is the fundamental unit in Power Automate. It is a defined sequence of steps that runs automatically when certain conditions are met. Every flow has at minimum one trigger and one action.

Think of a flow like a recipe: the trigger is the event that starts cooking, and the actions are the steps that follow.

### Trigger

A **trigger** is the event that starts a flow. Every flow begins with exactly one trigger. Triggers fall into two broad categories:

- **Event triggers** — fire when something happens (a new email arrives, a row is added to a table, a button is pressed)
- **Schedule triggers** — fire on a time-based recurrence (every Monday at 9 AM, every 15 minutes)

Triggers determine the flow type. You cannot change a flow's trigger type after creation.

### Action

An **action** is a single operation the flow performs after the trigger fires. Flows can contain dozens of actions. Actions include:

- Sending an email
- Creating a file in SharePoint
- Updating a database row
- Calling an HTTP endpoint
- Posting a Teams message

Actions run sequentially by default. Control actions (conditions, loops, parallel branches) let you change execution order.

### Connector

A **connector** is a prebuilt adapter that lets Power Automate communicate with an external service. Each connector wraps an underlying API and exposes triggers and actions as named operations.

Microsoft ships over 1,000 connectors covering services from SharePoint and Excel to Salesforce, Slack, Twitter/X, and SAP. Connectors come in three tiers:

| Tier | Examples | Licensing |
|---|---|---|
| **Standard** | SharePoint, Outlook, Teams, OneDrive | Included with most Microsoft 365 plans |
| **Premium** | SQL Server, Salesforce, HTTP, Dataverse | Requires Power Automate Per User or Per Flow plan |
| **Custom** | Your own internal APIs | Built by your organization; requires Premium plan |

> **On screen:** In the Power Automate portal, navigate to `Data > Connectors` in the left sidebar. The connector gallery shows all available connectors. A diamond icon in the connector tile indicates it is a Premium connector.

### Connection

A **connection** is an authenticated instance of a connector. When you add a SharePoint action to a flow, you are prompted to create a connection — this is where you provide credentials (OAuth, API key, etc.) that the flow uses at runtime. Connections are stored at the user or service account level, not inside the flow definition itself.

### Environment

An **environment** is an isolated container that holds flows, apps, data, and connections. Environments serve as administrative and security boundaries.

- Every organization has a **Default environment** created automatically
- Administrators create additional environments for Dev, Test, and Production
- Flows in one environment cannot directly access flows or data in another environment
- Each environment can have its own Dataverse database

> **On screen:** The environment selector appears in the top-right corner of the Power Automate portal as a dropdown showing the current environment name. Clicking it lists all environments your account has access to.

---

## Flow Types

Power Automate supports five distinct flow types. Choosing the correct type is the first decision when designing any automation.

<div class="callout-info">

<strong>Info:</strong> Power Automate supports five distinct flow types.

</div>


### 1. Automated Cloud Flow

Triggered automatically when an event occurs in a connected service. The flow runs in the cloud with no user interaction required.

**Example triggers:** New email arrives, SharePoint list item created, HTTP request received, Teams message posted

**Best for:** Event-driven integrations, notifications, data synchronization

> **On screen:** `Home > + Create > Automated cloud flow`

### 2. Instant Cloud Flow (Manual / Button Flow)

Triggered manually by a user — by pressing a button in the Power Automate mobile app, in Teams, or from a Power Apps screen. Can accept user input at trigger time.

**Example:** A field technician taps a button on their phone to log a site visit and create a work order in Dynamics 365.

**Best for:** On-demand tasks that still benefit from automation, user-initiated processes

> **On screen:** `Home > + Create > Instant cloud flow`

### 3. Scheduled Cloud Flow

Triggered on a time-based recurrence. You define the start date, frequency, and time zone.

**Example:** Every weekday at 6 AM, query a SQL database for overnight exceptions and email a summary to the operations team.

**Best for:** Batch processing, reports, recurring maintenance tasks

> **On screen:** `Home > + Create > Scheduled cloud flow`

### 4. Desktop Flow (Robotic Process Automation)

Automates interactions with desktop applications and websites using a local agent called **Power Automate Desktop**. Desktop flows can record mouse clicks and keystrokes, enabling automation of legacy applications that expose no API.

Desktop flows run in two modes:
- **Attended:** Runs while a human user is logged in (user may see the automation)
- **Unattended:** Runs in the background on a dedicated machine (requires Unattended RPA add-on)

**Best for:** Legacy system integration, web scraping, automating software that has no API

> **On screen:** `Home > + Create > Desktop flow` — this opens Power Automate Desktop (a locally installed application)

### 5. Business Process Flow

A structured, stage-based workflow that guides users through a defined process via a UI embedded in a Dataiku or model-driven Power App. Unlike other flow types, Business Process Flows are user-facing rather than background processes.

**Example:** A multi-stage sales opportunity process with required fields at each stage (Qualify → Develop → Propose → Close).

**Best for:** Guided, human-driven processes with compliance or quality requirements

> **On screen:** `Home > + Create > Business process flow` — requires a Dataverse environment

---

## Common Use Cases

### Business Process Automation

<div class="callout-warning">

<strong>Warning:</strong> ### Business Process Automation

Replace manual, repetitive tasks with reliable automated sequences:

- Employee onboarding: auto-create Active Directory account, assign licenses, send welcome email
-...

</div>


Replace manual, repetitive tasks with reliable automated sequences:

- Employee onboarding: auto-create Active Directory account, assign licenses, send welcome email
- Invoice processing: extract data from email attachments, validate against purchase orders, route for approval
- Customer case routing: classify incoming support tickets and assign to the correct team queue

### Data Synchronization

Keep records consistent across systems that do not natively integrate:

- Sync new Salesforce contacts to a SharePoint directory
- Mirror Dynamics 365 order updates to a SQL reporting database
- Push IoT sensor readings from Azure Event Hubs to a Power BI streaming dataset

### Notifications and Alerts

Deliver timely information without requiring users to monitor dashboards:

- Alert on-call staff via Teams when a production monitoring alert fires
- Email weekly summaries of support ticket metrics to team leads
- Send mobile push notifications when a customer order ships

### Approval Workflows

Route decisions through a structured, trackable process:

- Document approval: request sign-off from multiple stakeholders in sequence or parallel
- Budget exception: escalate requests above a threshold to the next management level
- Content publishing: require editorial review before a SharePoint page goes live

Power Automate includes a built-in **Approvals connector** that handles the full lifecycle: request creation, email/Teams notification, response capture, and outcome routing.

---

## Portal Navigation Primer

The Power Automate portal lives at `make.powerautomate.com`. Key sections:

| Section | Path | Purpose |
|---|---|---|
| Home | `/` | Quick access to recent flows and templates |
| My Flows | `My flows` in left sidebar | Manage all flows you own or share |
| Create | `+ Create` button | Start a new flow |
| Templates | `Templates` in left sidebar | Pre-built flows for common scenarios |
| Connectors | `Data > Connectors` | Browse the connector catalog |
| Connections | `Data > Connections` | Manage your authenticated connections |
| Solutions | `Solutions` in left sidebar | Package flows for deployment |
| Monitor | `Monitor` section | View flow run history and errors |

> **On screen:** After signing in at `make.powerautomate.com`, the left navigation sidebar is always visible. The sidebar collapses to icons only on narrow screens — hover over any icon to see its label.

---

## Connections to Other Modules

- **Builds on:** Microsoft 365 familiarity (SharePoint, Outlook, Teams)
- **Leads to:** Module 01 — Building Your First Cloud Flow
- **Related to:** Power Apps (Module 05), Copilot integration (Module 08)

---


## Practice Questions

**Question 1 — Conceptual:** Based on the concepts in this guide, explain in your own words why the core technique matters and when you would choose it over alternatives.

**Question 2 — Application:** Sketch out how you would apply the main concept from this guide to a real-world dataset or problem you have encountered. What would you need to watch out for?


## Further Reading

- [Power Automate Documentation](https://learn.microsoft.com/en-us/power-automate/) — Official Microsoft docs; most reliable source for current feature availability
- [Power Platform Licensing Guide](https://go.microsoft.com/fwlink/?linkid=2085130) — PDF updated quarterly; defines exactly what each license tier includes
- [Connector Reference](https://learn.microsoft.com/en-us/connectors/connector-reference/) — Complete catalog of all connectors with trigger/action documentation


---

## Cross-References

<a class="link-card" href="./01_platform_overview_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Interactive slide deck covering the key concepts with visual examples.</div>
</a>

<a class="link-card" href="../notebooks/01_power_automate_overview.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises and real data.</div>
</a>
