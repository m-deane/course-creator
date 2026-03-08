# Connectors Deep Dive

## In Brief

A connector is a pre-built wrapper around an external service's API. Every action and trigger in Power Automate belongs to a connector. Choosing the right connectors — and understanding their licensing tier, authentication requirements, and rate limits — is the foundation of building reliable, production-grade flows.

## Learning Objectives

By the end of this guide you will be able to:

1. Distinguish standard, premium, and custom connectors and identify which license tier each requires
2. Configure authentication for OAuth, API key, and shared-access connectors
3. Identify the key actions and triggers available in the seven most-used connectors
4. Explain custom connector creation at a conceptual level
5. Describe rate limiting and throttling behaviour and design flows that handle it

---

## Connector Categories and Licensing

Power Automate connectors fall into three licensing tiers. The tier determines what plan your organization needs to run flows that use the connector.

```
┌─────────────────────────────────────────────────────────────────────┐
│                      CONNECTOR CATEGORIES                           │
│                                                                     │
│   STANDARD          PREMIUM           CUSTOM                        │
│   (included)        (paid plan)       (you build it)                │
│                                                                     │
│   Office 365        SQL Server        Internal API                  │
│   Outlook           Salesforce        Legacy system                 │
│   SharePoint        DocuSign          Partner service               │
│   Teams             Adobe PDF         IoT device                    │
│   OneDrive          HTTP              Proprietary database          │
│   Excel Online      Azure services    Any OpenAPI service           │
│   Forms             Dataverse         Any REST endpoint             │
│   Planner                                                           │
│   Approvals                                                         │
│   MSN Weather                                                       │
│   Notifications                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

| Tier | License required | Connector count (approx) | Key characteristic |
|------|-----------------|------------------------|-------------------|
| Standard | Microsoft 365 plan (any) | 400+ | Microsoft and major SaaS services |
| Premium | Power Automate per-user or per-flow plan | 200+ | Databases, external SaaS, HTTP |
| Custom | Same plan as built-in connectors used | Unlimited | You define actions and triggers |

> **On screen:** In the flow designer, when you search for connectors in the "Choose an operation" panel, a **Premium** label badge appears on any premium connector. Your flows will save successfully with premium connectors but will be disabled if your license is downgraded below the required tier.

---

## Key Connector Walkthroughs

### 1. Office 365 Outlook

**Tier:** Standard
**Authentication:** OAuth 2.0 (Microsoft identity platform — signs in with your M365 account)

**Primary triggers:**

| Trigger | Description |
|---------|-------------|
| When a new email arrives (V3) | Webhook — fires immediately when email arrives in a folder |
| When an email is flagged | Fires when a message flag status changes |
| When a new email mentioning me arrives | Fires on @mentions only |

**Primary actions:**

| Action | Description |
|--------|-------------|
| Send an email (V2) | Send from your own address or a shared mailbox |
| Get email | Retrieve a specific email by message ID |
| Reply to email (V2) | Thread reply, preserving conversation ID |
| Forward email (V3) | Forward with optional additional content |
| Move email | Move to another folder by folder path |
| Flag email | Set, clear, or mark complete on message flags |
| Export email | Get raw MIME content for archiving or processing |
| Create event | Add a calendar event |
| Get calendar view | List events in a date range |

> **On screen:** To send from a **shared mailbox**, expand the **Send an email (V2)** action card and enable **Show advanced options**. Set the **From** field to the shared mailbox address (e.g. `helpdesk@contoso.com`). You must have "Send As" or "Send on Behalf" permissions on that mailbox in Exchange.

**Gotcha:** The V2 and V3 variants of triggers and actions are not interchangeable. V3 of "When a new email arrives" supports attachment handling and the `hasAttachments` filter that V2 lacks. Always use the highest available version.

---

### 2. Microsoft Teams

**Tier:** Standard
**Authentication:** OAuth 2.0 (same Microsoft identity)

**Primary triggers:**

| Trigger | Description |
|---------|-------------|
| When a new channel message is added | Polls for new messages in a specific channel |
| When a Teams message action is triggered | Webhook — user right-clicks a message |
| When a new chat message is received | Polls for new messages in 1:1 or group chats |

**Primary actions:**

| Action | Description |
|--------|-------------|
| Post a message in a chat or channel | Post as the flow bot or as yourself |
| Post an Adaptive Card and wait for a response | Send a card UI and block until a user responds |
| Create a team | Provision a new team |
| Add a member to a team | Add user by email |
| Get message details | Retrieve full message content by ID |
| List channels | Get all channels in a team |

> **On screen:** The **Post a message in a chat or channel** action has a **Post as** field. Choose **Flow bot** to post as the "Power Automate" bot identity (no license overhead). Choose **User** to post as the signed-in user identity — this requires delegated permissions and the message appears to come from a real person.

**Key capability — Adaptive Cards:**

The **Post an Adaptive Card and wait for a response** action blocks the flow until a Teams user interacts with a card you design. The card can contain dropdowns, text inputs, and buttons. The user's response is available as dynamic content for subsequent actions. This is the preferred alternative to email-based approvals when your organisation lives in Teams.

---

### 3. SharePoint

**Tier:** Standard
**Authentication:** OAuth 2.0 (Microsoft identity)

SharePoint is the most widely-used connector in Power Automate and has the largest action library.

**Primary triggers:**

| Trigger | Description |
|---------|-------------|
| When an item is created | Polling — fires when a new list item appears |
| When an item is modified | Polling — fires on any column change to an existing item |
| When a file is created (properties only) | Polling — fires when a new file is added to a library |
| When a file is modified (properties only) | Polling — fires on file metadata or content change |

**Note:** All SharePoint list/library triggers are **polling** — expect up to a 3-minute lag on standard plans.

**Primary actions:**

| Action | Description |
|--------|-------------|
| Get item | Read a single list item by ID |
| Get items | List items with optional OData filter, sort, top N |
| Create item | Add a new list item |
| Update item | Update specific columns on an existing item |
| Delete item | Remove an item by ID |
| Get file content | Read the binary content of a file |
| Get file metadata | Read file properties (name, size, URL, created by) |
| Create file | Upload a new file to a library |
| Update file content | Overwrite an existing file's content |
| Send an HTTP request to SharePoint | Raw SharePoint REST API call for advanced operations |

> **On screen:** For **Get items**, always set the **Top Count** field to a specific number (e.g. `100`). Without it, Power Automate defaults to returning only 100 items. To page through large lists, enable **Pagination** in the action's Settings menu and set the threshold.

**OData filter example for Get items:**

```
Status eq 'Pending' and Modified gt '2024-01-01T00:00:00Z'
```

This retrieves only items where Status is "Pending" and Modified date is after 1 January 2024. Filtering at the SharePoint layer is faster and cheaper than retrieving all items and filtering in a Condition action.

---

### 4. OneDrive for Business

**Tier:** Standard
**Authentication:** OAuth 2.0 (Microsoft identity)

**Primary triggers:**

| Trigger | Description |
|---------|-------------|
| When a file is created | Webhook — fires when a new file appears in a watched folder |
| When a file is modified | Webhook — fires when file content changes |

**Note:** OneDrive triggers are **webhook-based** — near-instant latency, unlike SharePoint polling triggers.

**Primary actions:**

| Action | Description |
|--------|-------------|
| Get file content | Read binary content of a file |
| Get file metadata | Name, size, last modified, sharing links |
| Create file | Upload a new file |
| Update file | Overwrite file content |
| List files in folder | Enumerate folder contents |
| Convert file | Convert Office documents to PDF or other formats |
| Copy file | Duplicate a file to another path |
| Delete file | Remove a file |
| Create share link | Generate a public or organisation-scoped link |

> **On screen:** The **Convert file** action supports these conversions: `.docx → .pdf`, `.xlsx → .pdf`, `.pptx → .pdf`, `.html → .pdf`. After conversion, the output is a binary `File Content` token you can pass to a **Send an email** action as an attachment or a **Create file** action to save the PDF.

---

### 5. Excel Online (Business)

**Tier:** Standard
**Authentication:** OAuth 2.0

Excel Online operates on `.xlsx` files stored in **OneDrive for Business** or **SharePoint**. The file must be stored in one of these locations — local files on a desktop are not accessible.

**Primary actions:**

| Action | Description |
|--------|-------------|
| Add a row into a table | Append a new row to a named Excel table |
| List rows present in a table | Read all rows from a named table |
| Get a row | Read a single row by key column value |
| Update a row | Change values in a specific row |
| Delete a row | Remove a row by key column value |
| Run script | Execute an Office Script (TypeScript) stored in the workbook |

> **On screen:** The Excel connector requires that data lives inside a **named Table** (Insert → Table in Excel, not just a range). When you open the action card and select your file, the **Table** dropdown only shows named tables. If it is empty, the file has no named tables yet.

**Power move — Run script:** The **Run script** action calls an Office Script written in TypeScript that you author directly in Excel Online. This lets you execute arbitrary spreadsheet logic (conditional formatting, complex calculations, pivot refresh) from within a flow, returning results as structured output.

---

### 6. HTTP Connector

**Tier:** Premium
**Authentication:** None, Basic, Client Certificate, Active Directory OAuth, Raw

The HTTP connector makes direct HTTP/HTTPS calls to any REST or SOAP API endpoint. It is the Swiss Army knife connector for integrating with systems that do not have a dedicated connector.

> **On screen:** Search for `HTTP` in the connector browser. You will see several options. Select the one labelled simply **HTTP** (not "HTTP + Swagger" or "HTTP Webhook"). This appears with the **Premium** badge.

**Configuration fields:**

| Field | Description |
|-------|-------------|
| Method | GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS |
| URI | Full URL of the API endpoint, with optional query string |
| Headers | Key-value pairs (e.g. `Content-Type: application/json`) |
| Body | Request body for POST/PUT/PATCH (JSON, XML, or plain text) |
| Authentication | Authentication type and credentials |

**Authentication types:**

| Type | When to use |
|------|-------------|
| None | Public APIs with no auth |
| Basic | Username + password sent as Base64 header |
| Client Certificate | Mutual TLS with a .pfx certificate |
| Active Directory OAuth | Azure AD-protected APIs (app registration required) |
| Raw | Custom auth headers (API keys, Bearer tokens) |

> **On screen:** For API key authentication, set **Authentication** to `Raw`. In the **Value** field, enter the complete header value: `Bearer YOUR_API_KEY` or `ApiKey YOUR_KEY` depending on the API's specification. Store the actual key in a **Power Automate Environment Variable** or retrieve it from **Azure Key Vault** using the Key Vault connector — never hardcode secrets in action fields.

**Parsing the HTTP response:**

The raw HTTP response body is a JSON string. To access individual fields, add a **Parse JSON** action (from the Data Operations connector) immediately after the HTTP action. Paste a sample response body and click **Generate from sample** — Power Automate generates the schema automatically and exposes each field as a dynamic content token.

---

### 7. SQL Server

**Tier:** Premium
**Authentication:** SQL Server authentication or Windows/Azure AD

SQL Server is the entry point for flows that read from or write to relational databases — whether on-premises (via a data gateway) or Azure SQL.

> **On screen:** When you first add a SQL Server action, a "New connection" dialog appears. Enter:
> - **Authentication Type:** SQL Server Authentication (or Windows Authentication)
> - **SQL Server name:** e.g. `myserver.database.windows.net`
> - **SQL Database name:** e.g. `OperationsDB`
> - **Username / Password:** your SQL credentials

For on-premises SQL Server, also select the **On-premises data gateway** installed on a machine with network access to the database server.

**Primary triggers:**

| Trigger | Description |
|---------|-------------|
| When an item is created (V2) | Polling — new rows in a table |
| When an item is modified (V2) | Polling — row updates detected via timestamp column |

**Primary actions:**

| Action | Description |
|--------|-------------|
| Get rows (V2) | SELECT with optional filter, sort, top N |
| Get row (V2) | SELECT a single row by primary key value |
| Insert row (V2) | INSERT a new row |
| Update row (V2) | UPDATE a row by primary key |
| Delete row (V2) | DELETE a row by primary key |
| Execute stored procedure (V2) | Run a stored proc with parameters |
| Execute SQL query (V2) | Run raw T-SQL (use carefully — injection risk) |

> **On screen:** Prefer **Execute stored procedure** over **Execute SQL query** for complex operations. Stored procedures are tested, parameterised, and version-controlled in the database. Raw SQL queries typed into a flow action field are harder to maintain and audit.

---

## Connection Authentication Types

Every connector requires a **connection** — a saved credential that the connector uses to call the external service. Power Automate stores connections in the environment and reuses them across flows.

```
Flow A  ──► Outlook Connection (OAuth token for alice@contoso.com) ──► Exchange Online
Flow B  ──► Outlook Connection (OAuth token for alice@contoso.com) ──► Exchange Online
Flow C  ──► SQL Connection (SQL auth for db_reader account) ──► Azure SQL
```

### OAuth 2.0 (Delegated)

Used by: Outlook, Teams, SharePoint, OneDrive, Excel Online

```
Power Automate                Microsoft Identity
     │                              │
     │──── redirect to login ──────►│
     │◄─── authorization code ──────│
     │──── exchange for tokens ────►│
     │◄─── access token (1 hr) ─────│
     │      refresh token (90 days) │
     │                              │
     │ stores tokens in connection  │
     │ auto-refreshes silently      │
```

The flow runs as the user who created the connection — emails sent by the Outlook action come from that user's address, files created in SharePoint are owned by that user. If the creating user leaves the organisation, the connection breaks. Use a **service account** (a shared M365 mailbox) for production flows.

### API Key

Used by: Various third-party connectors (GitHub, Stripe, Twilio)

A static key string is embedded in every API request. The key does not expire on a schedule but can be revoked from the service's admin panel.

> **On screen:** When creating the connection, the connector prompts specifically for the API key. Enter the key value. Power Automate stores it encrypted in the connection definition.

### Shared Access (Connection String)

Used by: Azure Blob Storage, Azure Service Bus, Azure Event Hubs

The connection string contains the endpoint URL and an access key or SAS token. Store the connection string value as an **Environment Variable** of type Secret, then reference it in the connection dialog.

---

## Custom Connector Creation Overview

When no built-in connector covers your API, you build a custom connector.

```
Your API (REST/SOAP)
    │
    ▼
OpenAPI (Swagger) definition
    │
    ▼
Custom Connector wizard in Power Automate
    │
    ├── General: name, icon, description, host URL
    ├── Security: auth type (API key, OAuth, Basic, None)
    ├── Definition: import OpenAPI or define manually
    │     ├── Actions (GET, POST, PUT, DELETE endpoints)
    │     └── Triggers (webhook registration endpoints)
    └── Test: verify each action against the live API
    │
    ▼
Connector available in all flows in the environment
```

> **On screen:** In the Power Automate portal, click **Data** in the left nav, then **Custom connectors**. Click **+ New custom connector**. Choose **Import an OpenAPI file** if your API has a Swagger spec, or **Create from blank** to define endpoints manually.

**Minimum requirements for a functional custom connector:**
- Host URL of the API
- At least one action with a defined request schema (inputs) and response schema (outputs)
- Authentication method configured to match the API

Custom connectors are **environment-scoped** — create them in the environment where they will be used, or export and import them across environments.

---

## Rate Limits and Throttling

Every connector enforces limits on how many API calls you can make per unit of time. Exceeding the limit causes the connector to return a `429 Too Many Requests` response and the flow action fails.

### Power Platform Request Limits

Microsoft enforces platform-level request limits per licensed user per day:

| License | Daily API call limit |
|---------|---------------------|
| Microsoft 365 (no Power Automate add-on) | 6,000 |
| Power Automate per-user plan | 40,000 |
| Power Automate per-flow plan | 250,000 |

### Connector-Level Throttling

Individual connectors add their own limits on top of platform limits:

| Connector | Throttle limit |
|-----------|---------------|
| SharePoint | 600 requests per 60 seconds per connection |
| SQL Server | Depends on Azure SQL DTU tier |
| Outlook | Microsoft Graph: 10,000 requests per 10 minutes per user |
| HTTP | Depends entirely on the target API |

### Handling Throttling in Flows

> **On screen:** Open a connector action card. Click the **…** menu. Select **Settings**. Under **Retry Policy**, set:
> - **Type:** Exponential interval
> - **Count:** 4 (retry up to 4 times)
> - **Interval:** PT5S (5 seconds initial wait, exponentially increasing)

This configuration tells Power Automate to automatically retry the action with increasing delays when it receives a `429` or `5xx` response. For most business flows, this handles transient throttling without any custom logic.

For flows that process large lists of items, use **Apply to each** with concurrency set to `1` (sequential) rather than the default parallel execution. Sequential processing reduces the peak request rate at the cost of longer total run time.

---

## Browsing and Adding Connectors in the Portal

### Finding a Connector

> **On screen:** In the flow designer, click **+ New step**. The "Choose an operation" panel opens on the right side. It shows:
> - **Search box** at the top — search by connector name or action keyword
> - **All** tab — every connector in alphabetical order
> - **Premium** tab — only premium connectors
> - **Custom** tab — only your environment's custom connectors
> - **Built-in** tab — non-connector operations (Variables, Conditions, Apply to each, etc.)

> **On screen:** To browse a connector's full action list before adding it: search for the connector name, click the connector tile (not an action row). The panel expands to show all **Triggers** and **Actions** tabs for that connector.

### Switching the Connection on an Existing Action

> **On screen:** Open the action card that uses the connection you want to change. Click the **…** menu. Select **My connections**. A dropdown appears listing all available connections for this connector type. Select the connection you want, or click **+ Add new connection** to authenticate a new account.

### Viewing Connection Health

> **On screen:** In the left navigation rail, click **Data** → **Connections**. This page lists every connection in the environment. A green circle indicates an active connection. A yellow warning icon indicates the connection token has expired or the credentials have changed — click the connection and re-authenticate.

---

## Common Connector Mistakes

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Building a flow with premium connectors on a free plan | Flow saves but is immediately disabled | Check connector tier before architecting the flow |
| Connection owned by a personal account (not service account) | Flow breaks when the employee leaves | Create connections using a dedicated service account M365 identity |
| Not paginating SharePoint Get items | Only first 100 items returned | Enable Pagination in action Settings, set threshold to list size |
| Hardcoding API keys in HTTP action Body field | Secret visible in run history | Store secrets in Environment Variables or Azure Key Vault |
| Default concurrency in Apply to each with rate-limited connector | 429 errors from parallel calls | Set Apply to each concurrency to 1 for throttle-sensitive connectors |
| Using Execute SQL query with string concatenation | SQL injection vulnerability | Use Execute stored procedure with proper parameterisation |

---

## Connections

- **Builds on:** Guide 01 — Trigger Types (triggers belong to connectors; connector tier affects trigger availability)
- **Leads to:** Module 03 — Data Operations and Expressions (manipulating data returned by connector actions)
- **Related to:** Module 05 — SharePoint and Excel in depth (SharePoint connector advanced patterns)

---

## Further Reading

- [Connector reference — all connectors](https://learn.microsoft.com/en-us/connectors/connector-reference/)
- [Premium connectors list](https://learn.microsoft.com/en-us/connectors/connector-reference/connector-reference-premium-connectors)
- [Custom connector overview](https://learn.microsoft.com/en-us/connectors/custom-connectors/define-openapi-definition)
- [Power Platform request limits](https://learn.microsoft.com/en-us/power-platform/admin/api-request-limits-allocations)
- [Power Automate retry policies](https://learn.microsoft.com/en-us/power-automate/actions-reference/retry-policy)
