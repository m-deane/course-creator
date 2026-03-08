# Power Platform Architecture Reference

A complete reference document for the Microsoft Power Platform components, connector ecosystem, environment model, and governance framework. Use this alongside the Module 00 guides when you need more depth on a specific area.

---

## Power Platform Components

### Power Automate

**Purpose:** Automate workflows, integrate services, and orchestrate business processes.

**Key capabilities:**
- Cloud flows: event-driven, scheduled, and manual automation running in Microsoft's cloud
- Desktop flows: RPA automation of Windows applications and websites via the Power Automate Desktop agent
- Business process flows: Stage-gated, human-guided process overlays on Dataverse model-driven apps
- AI Builder integration: Document processing, classification, object detection, and sentiment models as flow actions
- Process mining: Event log analysis to identify automation opportunities before building

**Execution model:**
- Cloud flows run on Microsoft-managed infrastructure in the region of the environment
- Each run is stateless — state persists in external systems (Dataverse, SharePoint, SQL)
- Run quota depends on license tier (see Licensing section)
- Retry policies are configurable per-action for transient failures

**Official documentation:** https://learn.microsoft.com/en-us/power-automate/

---

### Power Apps

**Purpose:** Build custom web and mobile applications using low-code tooling.

**App types:**
- **Canvas apps:** Pixel-perfect UI design, connect to virtually any data source via connectors, run in browser and mobile app
- **Model-driven apps:** Auto-generated UI from Dataverse data schema, optimized for business process management
- **Power Pages:** External-facing websites (formerly Power Apps Portals) — separate product with its own licensing

**Relationship to Power Automate:**
- Canvas apps can trigger Instant flows with user input via the Power Automate connector
- Model-driven apps embed Business Process Flows as UI overlays
- Power Apps Component Framework (PCF) controls can call flows via the connector

**Official documentation:** https://learn.microsoft.com/en-us/power-apps/

---

### Power BI

**Purpose:** Business intelligence, data visualization, and self-service analytics.

**Relationship to Power Automate:**
- Power Automate flows can push data to Power BI streaming datasets (real-time dashboards)
- Power BI reports embed a "Power Automate" visual that lets report viewers trigger flows directly from a dashboard
- Power Automate can alert stakeholders when a Power BI data alert fires (data threshold exceeded)

**Key concepts:**
- **Datasets:** The data layer (connected to SQL, SharePoint, Excel, Dataverse, etc.)
- **Reports:** Visualizations built on top of datasets
- **Dashboards:** Pinned tiles from multiple reports; supports real-time streaming
- **Dataflows:** ETL pipelines that prepare and transform data before loading into datasets

**Official documentation:** https://learn.microsoft.com/en-us/power-bi/

---

### Copilot Studio

**Purpose:** Build, test, and publish AI-powered conversational agents (chatbots and copilots).

**Relationship to Power Automate:**
- Copilot Studio agents can call Power Automate flows as actions (to look up data, submit forms, trigger processes)
- Flows can call Copilot Studio agents via HTTP to generate AI responses
- Both share the same connector and environment model

**Key concepts:**
- **Topics:** Conversation branches triggered by user intent
- **Actions:** Operations the agent performs (call a flow, query a data source, generate text)
- **Knowledge sources:** Documents, websites, or Dataverse tables the agent searches when answering questions
- **Channels:** Teams, web chat, mobile apps, telephony — where the agent is published

**Official documentation:** https://learn.microsoft.com/en-us/microsoft-copilot-studio/

---

### Power Pages

**Purpose:** Build external-facing business websites with forms, lists, and authenticated portals.

**Relationship to Power Automate:**
- Form submissions on Power Pages can trigger flows via Power Automate integration
- Flows can read and write Dataverse records that Power Pages surfaces to external users

**Official documentation:** https://learn.microsoft.com/en-us/power-pages/

---

### Microsoft Dataverse

**Purpose:** The shared, governed, enterprise-grade data platform for the entire Power Platform.

**Architecture:**
- A relational store with a pre-defined schema (tables, columns, relationships) and extensible custom tables
- Built-in security model: organization-level, business unit-level, team-level, and record-level access control
- Audit logging: every create/update/delete can be tracked to a specific user and timestamp
- File and image storage integrated with the relational schema
- Dual-write integration with Dynamics 365 Finance and Operations (ERP sync)

**Key entities (tables) included by default:**
- Account, Contact, Lead, Opportunity (CRM foundation)
- Activity types: Email, Phone Call, Task, Appointment
- User, Team, Business Unit (org structure)
- Solution, Publisher (ALM support)

**Storage tiers:**
- Database capacity (rows): charged per GB, included in base entitlement
- File capacity (attachments): separate entitlement
- Log capacity (audit records): separate entitlement

**Official documentation:** https://learn.microsoft.com/en-us/power-apps/maker/data-platform/

---

## Connector Categories

Connectors are the integration primitives of Power Automate. They fall into three licensing tiers and several functional categories.

### Tier 1: Standard Connectors

Included with Microsoft 365 subscriptions that include Power Automate (most M365 Business and Enterprise plans).

| Connector | Category | Common Uses |
|---|---|---|
| SharePoint | Microsoft 365 | Document libraries, lists, metadata |
| Microsoft Teams | Microsoft 365 | Post messages, create channels, manage meetings |
| Outlook / Office 365 Outlook | Microsoft 365 | Send email, read inbox, calendar operations |
| OneDrive for Business | Microsoft 365 | File read/write, folder management |
| Excel Online (Business) | Microsoft 365 | Read/write Excel tables stored in SharePoint/OneDrive |
| Microsoft Forms | Microsoft 365 | Respond to form submissions |
| Planner | Microsoft 365 | Task creation and updates |
| Approvals | Power Automate built-in | Send approval requests, capture responses |
| Notifications | Power Automate built-in | Mobile push notifications to Power Automate app |
| Control (Condition, Loop, Scope) | Power Automate built-in | Flow control logic |
| Data Operations | Power Automate built-in | Parse JSON, compose, select, filter arrays |
| Variables | Power Automate built-in | Initialize, increment, set, append |
| RSS | Productivity | Monitor web feeds |
| Twitter/X | Social | Monitor and post (subject to API changes) |

### Tier 2: Premium Connectors

Require a Power Automate Per User or Per Flow license (or a standalone Power Apps Premium license).

| Connector | Category | Common Uses |
|---|---|---|
| Microsoft Dataverse | Microsoft | Read/write Dataverse tables directly |
| SQL Server | Database | Query and write SQL databases (cloud or on-premises via gateway) |
| Azure SQL Database | Database | SQL Server on Azure |
| Salesforce | CRM | Sync contacts, opportunities, cases |
| Dynamics 365 | Microsoft ERP/CRM | Read and update Dynamics records |
| ServiceNow | ITSM | Create incidents, update CMDB records |
| SAP ERP | ERP | Read/write SAP systems (requires SAP connector gateway) |
| HTTP | Generic | Call any REST API with any HTTP method |
| HTTP with Azure AD | Generic | Authenticated REST calls to Azure AD-protected APIs |
| Adobe Sign | Document | Send documents for electronic signature |
| DocuSign | Document | Envelope creation and status tracking |
| Stripe | Payments | Process payments, manage subscriptions |
| Twilio | Communications | Send SMS and WhatsApp messages |
| SendGrid | Email | Transactional email delivery |
| Jira | Development | Create and update Jira issues |
| GitHub | Development | Manage repositories, issues, PRs |
| AI Builder | AI | Document processing, text classification, object detection |

### Tier 3: Custom Connectors

Built by your organization to wrap internal APIs, partner APIs, or any service not covered by the 1,000+ built-in connectors.

**How custom connectors work:**
1. Provide an OpenAPI (Swagger) 2.0 or 3.0 specification describing the API's endpoints, parameters, and authentication
2. The Power Automate Custom Connector wizard converts this into a connector with named triggers and actions
3. Once certified by your admin, the connector is available in all flows in the environment (or tenant-wide)

**Authentication options for custom connectors:**
- No auth (public APIs)
- API Key (passed in header or query string)
- Basic authentication (username/password in Authorization header)
- OAuth 2.0 (authorization code or client credentials grant)
- Azure Active Directory (OAuth with AAD as the identity provider)
- Windows authentication (for on-premises APIs via the data gateway)

**Certification path:**
Custom connectors can be submitted to Microsoft for certification as Independent Software Vendor (ISV) connectors — making them available to all Power Automate tenants as Standard or Premium connectors.

**Official documentation:** https://learn.microsoft.com/en-us/connectors/custom-connectors/

---

## Environment Types and Management

Environments are the primary administrative unit in Power Automate. Understanding environment types and lifecycle management is essential for any practitioner beyond basic personal automation.

### Environment Types

| Type | Created By | Purpose | Reset Allowed |
|---|---|---|---|
| Default | Auto-created for every tenant | Personal productivity, experimentation | No |
| Production | Admin creates | Live business processes | No |
| Sandbox | Admin creates | Development, testing, training | Yes |
| Developer | Individual user (Developer Plan) | Personal development sandbox | No |
| Trial | User starts trial | 30-day evaluation | No |

**Default environment characteristics:**
- Name follows the pattern: `{Organization Name} (default)`
- Every licensed user in the tenant can create flows here
- Cannot be deleted (only admins can restrict access)
- Has Dataverse provisioned by default in most tenants
- Not appropriate for business-critical flows (no governance isolation)

**Sandbox environment reset:**
Resetting a sandbox wipes all flows, apps, Dataverse data, and connections. This is used to restore a development or training environment to a clean state. Reset is irreversible — all content is permanently deleted.

### Environment Management

Environments are managed in the **Power Platform Admin Center** (`admin.powerplatform.microsoft.com`), not in the Power Automate portal.

Administrative tasks available in the Admin Center:
- Create, copy, and delete environments
- Set environment capacity (Dataverse storage allocation)
- Manage makers (who can create in each environment)
- Apply and manage DLP policies
- View tenant-level analytics (flow run counts, connector usage, active users)
- Configure Customer Lockbox and managed identities

**Environment capacity:**
Each environment consumes Power Platform storage capacity (Dataverse database storage). Storage is pooled at the tenant level and allocated from a base entitlement. Tenants with many environments and large Dataverse databases may need to purchase additional storage capacity.

### Environment Lifecycle (ALM Pattern)

```
Developer Environment  →  Development  →  Test  →  Production
(individual)              (team)           (QA)     (live)

Each promotion step:
1. Export solution as Managed from source environment
2. Import Managed solution into target environment
3. Configure environment-specific connection references
4. Validate flows run correctly
5. Document the deployment in change log
```

**Connection references:** When flows are packaged in solutions and promoted across environments, they use **connection references** instead of hard-coded connections. A connection reference is a pointer to a connection that can be re-mapped to a different account/credential in each environment. This allows the same flow definition to use `dev-service-account@org.com` in Development and `prod-service-account@org.com` in Production without modifying the flow.

**Environment variables:** Solutions can include environment variables — named key/value pairs that the flow references. The values differ per environment (e.g., a SharePoint site URL that differs between Dev and Prod). This keeps the flow definition environment-agnostic.

---

## Data Loss Prevention (DLP) Policy Overview

DLP policies protect organizational data by controlling which connectors can be used together in a single flow. They are configured by Power Platform administrators and enforced automatically by the platform.

### Connector Groups

Administrators assign each connector to one of three groups per DLP policy:

| Group | Description | Behavior |
|---|---|---|
| **Business** | Approved for organizational (corporate) data | Can share data with other Business-group connectors |
| **Non-Business** | Approved for personal/non-corporate data | Can share data with other Non-Business connectors |
| **Blocked** | Not allowed in any flow in this environment | Flows using this connector are suspended |

**Enforcement:** A flow that mixes connectors from the Business group and the Non-Business group in the same flow violates the DLP policy. The platform automatically suspends the flow and notifies the flow owner.

**Example policy configuration:**
```
Business group:   SharePoint, Dataverse, Teams, Outlook, SQL Server, Salesforce
Non-Business:     Twitter/X, Dropbox Personal, Gmail
Blocked:          Custom HTTP (to prevent exfiltration to unknown endpoints)
```

With this policy, a flow that reads from SharePoint (Business) and posts to Twitter/X (Non-Business) would be suspended. A flow that reads from SharePoint and sends a Teams message is allowed (both Business group).

### DLP Scope

DLP policies can be scoped to:
- **All environments** in the tenant (tenant-wide policy)
- **Specific environments** (targeted policy, can be more or less restrictive)
- **Excluding specific environments** (e.g., exclude the Developer environments from the strictest policy)

Multiple policies can apply to the same environment. The most restrictive policy wins.

### DLP and Custom Connectors

Custom connectors default to the Non-Business group until explicitly moved by an administrator. This prevents newly-created custom connectors from inadvertently accessing Business data until reviewed.

### Impact on Flow Makers

Flow makers cannot see or modify DLP policies — they can only observe the effect (a flow being suspended with a policy violation banner). If a connector is blocked or restricted, the maker must:
1. Contact the Power Platform administrator to review the policy
2. Or redesign the flow to use only connectors from the same group

**Official documentation:** https://learn.microsoft.com/en-us/power-platform/admin/wp-data-loss-prevention

---

## Licensing Quick Reference

Licensing determines which connectors are available, the run quota, and RPA capabilities. This is a simplified overview — always verify current pricing at `https://powerautomate.microsoft.com/pricing/`.

| License | Included With | Connectors | Run Quota | RPA |
|---|---|---|---|---|
| Seeded (M365) | Microsoft 365 Business/Enterprise | Standard only | 6,000 API calls/month per licensed user in the tenant | No |
| Power Automate Free | Free (limited) | Standard only | 750 run-minutes/month | No |
| Per User | ~$15/user/month | Standard + Premium | Unlimited cloud flow runs for that user | Attended (add-on) |
| Per Flow | ~$100/flow/month | Standard + Premium | Unlimited runs regardless of user count | Attended (add-on) |
| Per User + RPA | Per User + ~$40/user/month | Standard + Premium | Unlimited | Attended (included) |
| Per Flow + Unattended | Per Flow + ~$150/machine/month | Standard + Premium | Unlimited | Attended + Unattended |

**Choosing Per User vs. Per Flow:**
- Per User: cost-effective when a specific user (or service account) triggers most flows
- Per Flow: cost-effective when a flow is triggered by many different users (shared automation)
- Example: An approval flow triggered by 200 different employees is better on Per Flow ($100/month total) than Per User ($15 × 200 = $3,000/month)

**AI Builder credits:** AI Builder models consume credits (a separate entitlement). Per User plans include a partial allocation; additional credits are purchased as an add-on. AI Builder usage (document extraction, classification runs) is metered per model invocation.

---

## Official Microsoft Documentation Links

| Resource | URL | Description |
|---|---|---|
| Power Automate docs | https://learn.microsoft.com/en-us/power-automate/ | Complete feature documentation |
| Connector reference | https://learn.microsoft.com/en-us/connectors/connector-reference/ | All 1,000+ connectors with trigger/action details |
| Power Platform admin docs | https://learn.microsoft.com/en-us/power-platform/admin/ | Environment, DLP, and governance management |
| Licensing guide (PDF) | https://go.microsoft.com/fwlink/?linkid=2085130 | Definitive license comparison, updated quarterly |
| ALM guide | https://learn.microsoft.com/en-us/power-platform/alm/ | Solution-based deployment practices |
| DLP policies | https://learn.microsoft.com/en-us/power-platform/admin/wp-data-loss-prevention | DLP configuration and enforcement |
| Custom connectors | https://learn.microsoft.com/en-us/connectors/custom-connectors/ | Custom connector development guide |
| Power Automate Management API | https://learn.microsoft.com/en-us/power-automate/web-api | REST API for programmatic management |
| Microsoft Graph — flows | https://learn.microsoft.com/en-us/graph/api/resources/flow | Graph API flows endpoint reference |
| Power Platform CLI | https://learn.microsoft.com/en-us/power-platform/developer/cli/introduction | `pac` CLI for automation and DevOps |
| Microsoft Power Automate Community | https://powerusers.microsoft.com/t5/Microsoft-Power-Automate/ct-p/MPACommunity | Q&A, templates, and use case discussion |
| Power Automate Blog | https://powerautomate.microsoft.com/en-us/blog/ | Feature announcements and release notes |
