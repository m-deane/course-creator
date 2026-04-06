# Navigating the Power Automate Portal

> **Reading time:** ~16 min | **Module:** 0 — Platform Orientation | **Prerequisites:** None

## In Brief

The Power Automate portal at `make.powerautomate.com` is the primary interface for creating, managing, monitoring, and deploying flows. This guide walks through every major section of the portal, explains what each section is for, and describes the UI in enough detail to navigate without screenshots.

<div class="callout-insight">

<strong>Insight:</strong> The portal UI updates frequently. Microsoft ships weekly releases. If a menu label or icon differs slightly from what is described here, check `learn.microsoft.com/power-automate` for the current UI reference. The underlying concepts and section purposes remain stable.

</div>

<div class="callout-key">

<strong>Key Concept:</strong> The Power Automate portal at `make.powerautomate.com` is the primary interface for creating, managing, monitoring, and deploying flows. This guide walks through every major section of the portal, explains what each section is for, and describes the UI in enough detail to navigate without screenshots.

</div>


---

## Signing In

Navigate to `make.powerautomate.com` in any modern browser. Sign in with your Microsoft organizational account (work or school account). Consumer Microsoft accounts (personal @outlook.com, @hotmail.com) have limited access and cannot connect to organizational systems.

<div class="callout-insight">

<strong>Insight:</strong> Navigate to `make.powerautomate.com` in any modern browser.

</div>


> **On screen:** The sign-in page shows the standard Microsoft identity prompt. If your organization uses single sign-on (SSO), you may be redirected to your organization's identity provider (Azure AD B2B, Okta, ADFS, etc.) before landing in the portal.

After sign-in, you land on the **Home** page inside the environment your account defaults to.

---

## Environment Selector

Before navigating any section, locate the **environment selector** — it appears in the **top-right area** of the portal header, showing the current environment name next to a small grid/globe icon.

<div class="callout-key">

<strong>Key Point:</strong> Before navigating any section, locate the **environment selector** — it appears in the **top-right area** of the portal header, showing the current environment name next to a small grid/globe icon.

</div>


> **On screen:** The top header bar contains (left to right): the Power Automate logo, a search bar, and on the far right, your current environment name, a question-mark help icon, a notification bell, and your profile avatar. Click the environment name to open a dropdown listing all environments your account has access to.

**Why this matters:** Everything you see in the portal — flows, connectors, connections, solutions — belongs to the currently selected environment. Switching environments changes the entire context. This is the first thing to check when a flow you expect to see is "missing."

Environment types you will encounter:

| Type | Description |
|---|---|
| **Default** | Auto-created for every tenant; named after your organization. Everyone with a license can create here. |
| **Sandbox** | Isolated environment for development or testing. Can be reset (all data deleted) without affecting production. |
| **Production** | Live operational environment. Changes should go through a deployment process, not be made directly. |
| **Developer** | Single-user environment included with the Power Apps/Power Automate Developer Plan (free). |
| **Trial** | 30-day temporary environment created when someone starts a trial license. |

---

## Left Navigation Sidebar

The left sidebar is the primary navigation structure. It is always visible on wide screens and collapses to icon-only on narrow viewports. Hover over any icon to reveal its label when collapsed.

The sidebar sections, from top to bottom:

### Home

**Path:** Click the house icon or the Power Automate logo

The Home page shows:
- **Quick access tiles** for recent flows and apps
- **Recommended templates** based on your connector usage
- **Learning resources** — links to tutorials and documentation
- **News and announcements** about new features

> **On screen:** The center of the Home page has a "Start with a template or create from blank" prompt with a search bar for templates. Below that, a "Recent" section shows flows you have opened or edited recently as clickable cards with the flow name, last modified date, and status indicator (On/Off).

Use Home as a launchpad, not a management surface. For serious flow management, use My Flows.

---

### My Flows

**Path:** `My flows` in the left sidebar

My Flows is split into two tabs:

**Cloud flows tab:**
Lists all automated, instant, and scheduled cloud flows you own or that have been shared with you. Each row shows:
- Flow name (clickable to open the flow detail page)
- Flow type icon (lightning bolt for automated, cursor for instant, clock for scheduled)
- Modified date
- Status toggle (On/Off) — click to enable or disable without deleting
- An ellipsis (`...`) menu for: Edit, Share, Save as, Turn off, Delete, Export

> **On screen:** The ellipsis menu on a flow card reveals options including "Save as" (clone the flow), "Export" (download as a .zip package for deployment), and "Send a copy" (share a template copy with another user). The "Share" option adds co-owners or run-only users.

**Desktop flows tab:**
Lists Power Automate Desktop flows registered to this environment. Desktop flows appear here only if the machine running Power Automate Desktop is registered to this environment.

**Shared with me tab:**
Flows others have shared with you as a co-owner or run-only user.

#### Flow Detail Page

Click any flow name to open its detail page. This page shows:
- **Flow diagram** — a read-only visual of all steps (click Edit to modify)
- **Run history** — a table of recent runs with start time, duration, and status (Succeeded/Failed/Running)
- **Connections** — the authenticated connections the flow uses
- **Owners** — who owns and can edit the flow
- **Properties** — description, creation date, environment

> **On screen:** The Run history table has a filter bar above it. Click any row to expand it and see each action's execution result — green check marks for success, red X for failure, with the error message inline. This is the primary debugging interface for failed flows.

---

### Create

**Path:** `+ Create` button in the left sidebar (often the most prominent element)

The Create page offers three starting points:

**Start from blank:**
- **Automated cloud flow** — choose a trigger from the connector library; flow runs when the event fires
- **Instant cloud flow** — choose a manual trigger; flow runs when a user invokes it
- **Scheduled cloud flow** — set a recurrence; flow runs on the timer
- **Desktop flow** — opens Power Automate Desktop (local app must be installed)
- **Business process flow** — requires Dataverse; creates a stage-gated user journey

> **On screen:** Each option shows as a card with an icon, a one-sentence description, and a "Create" button. The Automated cloud flow card has an additional search field where you can type a trigger keyword (e.g., "SharePoint") to filter available triggers before naming the flow.

**Start from a template:**
Selecting any start-from-blank option also offers a "Start from template" link. This opens the Templates gallery filtered to that flow type.

**Start from AI description (Copilot):**
A text field where you describe the automation in plain English. Copilot drafts a flow structure. Available in supported regions and license tiers.

> **On screen:** `Home > + Create` — the Create button appears both in the left sidebar and as a prominent call-to-action on the Home page.

---

### Templates

**Path:** `Templates` in the left sidebar

The Templates gallery contains thousands of pre-built flow templates contributed by Microsoft and community partners.

**Gallery layout:**
- A search bar at the top (search by keyword, connector name, or use case)
- Filter chips below the search bar for categories (Approval, Notifications, Productivity, etc.) and flow type
- Template cards showing the flow name, connector icons involved, and a "Use template" button

> **On screen:** Each template card shows the connector icons used (e.g., a SharePoint icon and an Outlook icon for "Send email when a SharePoint list item is modified"). Clicking a card opens a detail page showing the full flow structure and required connections before you commit to using it.

**When to use templates:** Templates are excellent starting points for common patterns. They are not finished solutions — expect to customize the trigger conditions, email addresses, list names, and any business logic. Treat templates as scaffolding, not a finished product.

---

### Data

**Path:** `Data` section in the left sidebar — expands to sub-items

The Data section has four sub-items:

#### Dataverse

Access to the Dataverse tables in the current environment. Shows table list, relationships, and row counts. From here you can browse table schemas, which is essential when building flows that read or write Dataverse records.

#### Connections


<div class="callout-info">

<strong>Info:</strong> This section maps how this guide connects to the broader course. Use these links to navigate related material.

</div>

**This is one of the most important management pages in the portal.**

A connection is an authenticated instance of a connector. Before a flow can use a connector, a connection must exist.

> **On screen:** `Data > Connections` shows a table of all connections in the current environment. Columns: Connection name, Connector name, Status, Created date, Shared with. A "New connection" button at the top opens the connector gallery filtered to connectors you can create connections for.

Connection statuses:
- **Connected** (green) — healthy, credentials are valid
- **Error** (red) — authentication has expired or credentials are invalid; click to re-authenticate
- **Warning** (yellow) — connected but the account has limited permissions

Broken connections are the most common cause of flow failures. Check `Data > Connections` first when flows start failing unexpectedly after previously working.

#### Custom Connectors

Lists connectors your organization has built using the Custom Connector wizard (based on OpenAPI specs). These wrap internal APIs and expose them to flows as if they were first-party connectors.

> **On screen:** `Data > Custom connectors` — each custom connector shows its name, the API host URL, and whether it is certified (Microsoft-reviewed). A "New custom connector" button opens the four-tab creation wizard: General, Security, Definition, Test.

#### Gateways

**On-premises data gateway** configurations. Gateways allow cloud flows to reach data sources inside your organization's network — SQL Server on a VM, an on-premises file share, a local Oracle database — without exposing them directly to the internet.

> **On screen:** `Data > Gateways` lists registered gateways with their status (Online/Offline), machine name, and version. If a gateway is offline, flows using it will fail. Gateways require the gateway software to be installed on a machine that has network access to the target data source.

---

### Monitor

**Path:** `Monitor` in the left sidebar — expands to sub-items

Monitor provides run history across flows — not just a single flow's history, but a cross-flow view.

#### Cloud Flow Activity

A table of all cloud flow runs across all flows you have access to, with filters for date range, flow name, status, and environment. Use this to spot patterns — if multiple flows started failing at the same time, a connection or service may be down.

> **On screen:** `Monitor > Cloud flow activity` — the table has sortable columns: Flow name, Run start, Duration, Status. Click any run row to jump to the flow's run detail page.

#### Desktop Flow Activity

Same concept for desktop flow runs. Shows machine name, run mode (Attended/Unattended), and any errors from the desktop automation agent.

#### Machines

Lists machines registered to run desktop flows in this environment. Each entry shows the machine name, status (Available/Offline/Busy), and the Power Automate Desktop version installed.

---

### AI Hub

**Path:** `AI Hub` or `AI Builder` in the left sidebar (label varies by tenant configuration)

AI Builder provides prebuilt and custom AI models that can be used as actions inside flows. Examples:
- **Document processing** — extract structured data from invoices, receipts, forms
- **Text classification** — categorize incoming text (e.g., support tickets)
- **Object detection** — identify objects in images
- **Sentiment analysis** — score the tone of text

> **On screen:** The AI Hub home page shows model categories on the left and model cards on the right. Each card shows accuracy metrics if the model has been trained on your data, or "Prebuilt" if it is a Microsoft-provided model requiring no training.

AI Builder actions appear in the flow designer under the "AI Builder" connector. Using AI Builder requires a Power Automate Premium license or AI Builder add-on credits.

---

### Solutions

**Path:** `Solutions` in the left sidebar

Solutions are containers that package flows (and apps, tables, custom connectors, and other components) for deployment across environments. They are the correct mechanism for moving flows from Development to Test to Production.

> **On screen:** `Solutions` shows a list of solutions in the current environment. Each row shows the solution name, publisher, version number, and whether it is managed (imported from another environment) or unmanaged (created locally). The "New solution" button opens a dialog for name, publisher, and version.

**Managed vs. Unmanaged:**
- **Unmanaged** solutions are editable — this is the development state
- **Managed** solutions are read-only — this is the production state; flows inside cannot be directly edited in the portal

**Why solutions matter:** Flows created outside a solution are called "unmanaged flows in the default layer." They cannot be cleanly deployed to other environments using Microsoft's recommended ALM (Application Lifecycle Management) practices. Start new flows inside a solution from day one.

---

### Process Mining

**Path:** `Process mining` in the left sidebar (may require separate license)

Process Mining analyzes event logs from business systems to visualize how processes actually run — including detours, rework, and bottlenecks — before you automate them.

> **On screen:** Process Mining shows a canvas with process flow diagrams built from imported event logs. It integrates with Power BI for drill-down analysis. Requires the Process Mining add-on license.

---

## Licensing Tiers

Power Automate licensing directly determines which connectors you can use and how flows can run. Choosing the wrong license tier is a common source of confusion when a flow that works in one environment fails in another.

<div class="callout-info">

<strong>Info:</strong> Power Automate licensing directly determines which connectors you can use and how flows can run.

</div>


The following implementation builds on the approach above:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```text
Licensing Tiers (simplified):
├── Microsoft 365 (included with M365 subscriptions)
│   └── Standard connectors only; limited run quota
│
├── Power Automate (Free) — personal productivity
│   └── Standard connectors; 750 run-minutes/month
│
├── Power Automate Per User — $15/user/month (approx.)
│   ├── All Premium connectors
│   ├── AI Builder credits (partial)
│   └── Unlimited cloud flow runs for that user
│
├── Power Automate Per Flow — $100/flow/month (approx.)
│   ├── All Premium connectors
│   └── Unlimited runs regardless of how many users trigger it
│
├── Power Automate RPA (Attended) — add-on to Per User
│   └── Run desktop flows while user is logged in
│
└── Power Automate RPA (Unattended) — add-on to Per Flow
    └── Run desktop flows on dedicated machines, no user login required
```

</div>
</div>

> **On screen:** Your current license tier appears under your profile avatar > `View account`. In the Power Automate portal, attempting to add a Premium connector when you have only a Standard license shows an upgrade prompt inline in the flow designer — it does not block you from designing the flow, only from saving it in a licensed environment.

**Choosing the right license:**

| Scenario | Recommended License |
|---|---|
| Personal productivity automations using M365 apps only | Microsoft 365 included plan |
| Business automations using SQL, Salesforce, HTTP, or Dataverse | Per User |
| High-volume shared flows (many users trigger the same flow) | Per Flow |
| Automating Windows desktop apps (attended) | Per User + RPA Attended |
| Fully unattended robot on a VM | Per Flow + RPA Unattended |

Pricing changes regularly. Always verify current pricing at `https://powerautomate.microsoft.com/pricing/` before making license recommendations.

---

## Data Loss Prevention (DLP) Policies

DLP policies, set by administrators in the **Power Platform Admin Center** (a separate portal at `admin.powerplatform.microsoft.com`), control which connectors can be used together in a single flow and in which environments.

**Three connector groupings in DLP:**
- **Business** — approved for organizational data (e.g., SharePoint, Dataverse, Outlook)
- **Non-Business** — approved for personal/public data (e.g., Twitter/X, personal Dropbox)
- **Blocked** — cannot be used in any flow in this environment

A flow that mixes a Business-group connector with a Non-Business-group connector will be suspended by the DLP policy. This prevents flows from inadvertently copying sensitive organizational data to unauthorized external services.

> **On screen:** When a flow violates a DLP policy, the flow detail page shows a "Policy violation" banner with the policy name and the conflicting connectors highlighted. The flow is automatically disabled until the policy violation is resolved.

As a flow maker (not an admin), you cannot change DLP policies — you can only see the consequences. If a connector is blocked or restricted, escalate to your Power Platform administrator.

---

## Templates Gallery (Deep Dive)

The Templates gallery is more than a convenience feature — it is a learning resource for understanding common patterns.

**Finding the right template:**

1. Navigate to `Templates` in the left sidebar
2. Use the search bar — search by connector (e.g., "SharePoint"), use case (e.g., "approval"), or keyword
3. Filter by flow type using the chips below the search bar
4. Click a template card to preview the full flow structure before using it

**Template preview page:**
Shows the complete flow design — trigger, all actions, conditions — as a visual diagram. Below the diagram, a "Required connections" section lists every connector the template uses. You must have (or create) connections for each before the flow can run.

> **On screen:** The "Use template" button on the template preview page is blue and prominent. Clicking it prompts you to create any missing connections, then drops you into the flow designer with all steps pre-populated. You then customize names, values, and conditions for your specific scenario.

---

## Connections to Other Modules

- **Builds on:** Guide 01 (Platform Overview) — all terminology used here is defined there
- **Leads to:** Module 01 — Creating your first automated cloud flow; you will navigate Create > Automated cloud flow
- **Related to:** Module 07 (Desktop Flows) — the Machines section under Monitor becomes critical
- **Related to:** Module 09 (Copilot Agents) — the AI Hub section covers the AI Builder models used by Copilot

---

## Further Reading

- [Power Automate portal documentation](https://learn.microsoft.com/en-us/power-automate/getting-started) — official overview of the portal UI
- [Power Platform licensing overview](https://learn.microsoft.com/en-us/power-platform/admin/pricing-billing-skus) — authoritative licensing reference
- [DLP policies documentation](https://learn.microsoft.com/en-us/power-platform/admin/wp-data-loss-prevention) — how admins configure connector groupings
- [Solutions overview](https://learn.microsoft.com/en-us/power-platform/alm/solution-concepts-alm) — ALM practices for environment promotion
- [On-premises data gateway](https://learn.microsoft.com/en-us/power-automate/gateway-reference) — when and how to deploy gateways


---

## Cross-References

<a class="link-card" href="./02_navigating_portal_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Interactive slide deck covering the key concepts with visual examples.</div>
</a>

<a class="link-card" href="../notebooks/01_power_automate_overview.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises and real data.</div>
</a>
