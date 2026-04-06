# Trigger Types in Power Automate

> **Reading time:** ~14 min | **Module:** 2 — Triggers & Connectors | **Prerequisites:** Module 1

## In Brief

A trigger is the event that starts every flow. Power Automate organises triggers into three families: **automated** (react to an external event), **instant** (fire on demand), and **scheduled** (run on a timer). Choosing the right trigger family determines how your flow behaves, how often it can run, and what data it receives at startup.

<div class="callout-key">
<strong>Key Concept:</strong> A trigger is the event that starts every flow. Power Automate organises triggers into three families: **automated** (react to an external event), **instant** (fire on demand), and **scheduled** (run on a timer).
</div>


## Learning Objectives

By the end of this guide you will be able to:

<div class="callout-insight">
<strong>Insight:</strong> By the end of this guide you will be able to:

1.
</div>


1. Distinguish automated, instant, and scheduled trigger families and give an example of each
2. Explain the difference between polling and webhook trigger mechanisms
3. Add trigger conditions to reduce unnecessary flow runs
4. Navigate the Power Automate portal to locate and configure any trigger type

---

## The Three Trigger Families

```
                    ALL TRIGGERS
                         │
          ┌──────────────┼──────────────┐
          │              │              │
      AUTOMATED      INSTANT       SCHEDULED
    (event-driven)  (on-demand)    (timer)
          │              │              │
   email arrives    button press   recurrence
   file created     Power Apps     sliding window
   item changed     Teams action
```

<div class="callout-key">
<strong>Key Point:</strong> Every flow has exactly one trigger card — the topmost card on the canvas.
</div>


Every flow has exactly one trigger card — the topmost card on the canvas. The trigger determines:

- **When** the flow starts
- **What inputs** are available as dynamic content
- **How frequently** the flow can fire
- **Which connector license** applies

---

## Automated Triggers (Event-Driven)

An automated trigger listens for a specific event in a connected system and starts the flow the moment that event occurs. The flow does not run on a schedule; it wakes up only when something happens.

<div class="callout-info">
<strong>Info:</strong> An automated trigger listens for a specific event in a connected system and starts the flow the moment that event occurs.
</div>


### When an email arrives — Office 365 Outlook

> **On screen:** In the Power Automate portal, click **+ Create**, then **Automated cloud flow**. In the "Choose your flow's trigger" search box, type `email`. Select **When a new email arrives (V3)** under Office 365 Outlook. Click **Create**.

The trigger card exposes these configuration fields:

| Field | Purpose |
|-------|---------|
| Folder | Which mailbox folder to monitor (default: Inbox) |
| Include Attachments | Load attachment content into the flow |
| To | Filter: only trigger when the To address matches |
| CC | Filter: only trigger when the CC address matches |
| From | Filter: only trigger when the From address matches |
| Subject Filter | Only trigger when subject contains this string |
| Importance | Only trigger for Normal, High, or Low importance email |

> **On screen:** Expand **Show advanced options** to reveal the filtering fields. Setting **Subject Filter** to `URGENT:` means the flow ignores every email whose subject does not contain that exact string — saving flow runs and keeping your run history clean.

**Dynamic content available from this trigger:**

- `From`, `To`, `Subject`, `Body`, `Attachments`, `ReceivedTime`, `MessageId`, `ConversationId`

### When a file is created — SharePoint / OneDrive

> **On screen:** Search for `file created`. You will see two similar options:
> - **When a file is created (properties only)** — SharePoint: fires when any new file is added to a library
> - **When a file is created** — OneDrive for Business: fires on new files in a specific folder

The SharePoint variant returns metadata only by default (name, URL, timestamps). To read the file content you add a separate **Get file content** action downstream.

| Field | Purpose |
|-------|---------|
| Site Address | URL of the SharePoint site |
| Library Name | Document library to watch |
| Folder | Narrow to a specific subfolder (optional) |
| Include subfolders | Watch the folder and all children |

> **On screen:** For OneDrive, set **Folder** to `/Shared Documents/Reports`. Only files placed directly in that path trigger the flow; files in parent or sibling folders are ignored.

### When a SharePoint list item changes — SharePoint

> **On screen:** Search for `item modified`. Select **When an item is modified** under SharePoint.

| Field | Purpose |
|-------|---------|
| Site Address | Target SharePoint site |
| List Name | The list or library to watch |

**Important:** This trigger fires on every column change. If a user edits an item three times in quick succession, the flow may run three times. Use a trigger condition (covered below) or a **Condition** action in the flow body to handle this.

Dynamic content includes all list column values from the item that changed, plus `_ModifiedBy`, `_Modified`, and `ID`.

---

## Instant Triggers (On-Demand)

Instant triggers do not listen passively. A user or system actively fires them.

<div class="callout-warning">
<strong>Warning:</strong> Instant triggers do not listen passively.
</div>


### Manual / Button Trigger

> **On screen:** Click **+ Create**, then **Instant cloud flow**. The default trigger shown is **Manually trigger a flow**. Click **Create**.

This is the simplest trigger. A **Run** button appears on the flow's detail page and in the Power Automate mobile app.

You can add **inputs** to collect information before the flow runs:

> **On screen:** In the trigger card, click **+ Add an input**. Options are: Text, Yes/No, File, Email, Number, Date.

Example: add a **Text** input named `Recipient Email`. At run time, the person clicking the button is prompted to enter the address. The input value appears as dynamic content (`Recipient Email`) in all downstream actions.

Use cases:
- Ad-hoc report generation
- One-off data exports
- Testing flows before connecting a real trigger

### Power Apps Trigger

> **On screen:** Search for `Power Apps`. Select **PowerApps (V2)** as the trigger.

This trigger is invoked by a Power Apps canvas app calling `FlowName.Run(...)`. The flow can receive parameters passed from the app and return a response back to the app — enabling bidirectional communication.

The trigger card lets you define typed input parameters:

| Input type | Example use |
|-----------|-------------|
| Text | Customer name passed from a form |
| Number | Quantity selected in the app |
| Boolean | Toggle state from the app |
| File | Document uploaded in the app |

After defining inputs, a **Respond to a PowerApp or flow** action at the end of the flow sends return values back to the calling app.

### Teams Message Action Trigger

> **On screen:** Search for `Teams message action`. Select **When a Teams message action is triggered** under Microsoft Teams.

This trigger registers the flow as a context-menu action on any Teams message. When a user right-clicks a message and selects the flow name from the **More actions** menu, the flow fires with the message content as dynamic content.

Dynamic content includes: `messageBody`, `messageSender`, `teamId`, `channelId`, `messageLink`.

Use cases:
- Save a Teams message to a SharePoint list
- Escalate a message by creating a Planner task
- Translate the message body and post the result in a thread reply

---

## Scheduled Triggers (Timer-Based)

Scheduled triggers fire automatically on a repeating clock, independent of any user action or external event.

<div class="callout-insight">
<strong>Insight:</strong> Scheduled triggers fire automatically on a repeating clock, independent of any user action or external event.
</div>


### Recurrence Trigger

> **On screen:** Click **+ Create**, then **Scheduled cloud flow**. The "Build a scheduled cloud flow" dialog appears.

| Field | Purpose |
|-------|---------|
| Flow name | Name shown in your flow list |
| Starting | Date and time of the first run |
| Repeat every | Number + unit (Minute, Hour, Day, Week, Month) |

> **On screen:** After the flow is created, open the **Recurrence** trigger card. Click **Show advanced options** to access:

| Advanced field | Purpose |
|---------------|---------|
| Time zone | Locks the schedule to a specific zone regardless of server location |
| At these hours | Comma-separated hours (0–23) within the day to trigger |
| At these minutes | Minutes past the hour to trigger |
| On these days | Days of the week: Monday, Tuesday, … |

Example: run every weekday at 7:30 AM London time — set **Frequency** to `Week`, **At these hours** to `7`, **At these minutes** to `30`, **On these days** to `Monday, Tuesday, Wednesday, Thursday, Friday`, **Time zone** to `(UTC+00:00) Dublin, Edinburgh, Lisbon, London`.

The Recurrence trigger has **no dynamic content** — it fires on the clock and provides no event data. Any data the flow needs must be fetched by the first action.

### Sliding Window Trigger

> **On screen:** Click **+ Create**, then **Automated cloud flow**. Search for `Sliding Window`. Select **Sliding window** (built-in).

The Sliding Window trigger is a specialised scheduler that guarantees **no missed intervals**. If a flow run fails or the service has downtime, Sliding Window automatically queues catch-up runs for every missed interval once the service recovers.

| Field | Purpose |
|-------|---------|
| Interval + Frequency | Same as Recurrence |
| Start time | The anchor point for interval calculation |
| Delay | Optional wait after interval start before the run begins |

**Key difference from Recurrence:** Recurrence fires "at" a time and does not backfill missed runs. Sliding Window fires "for" every interval window that has passed, even late.

Use cases: financial data imports where every daily batch must be processed even if the system was down overnight.

---

## Trigger Conditions and Filtering

Without conditions, a trigger fires every time its event occurs — even when the flow body would do nothing useful. Trigger conditions evaluate *before* the flow run is charged to your quota, so they are the most efficient way to filter.

### Adding a Trigger Condition

> **On screen:** Open the trigger card. Click the **…** (ellipsis) menu in the top-right corner of the card. Select **Settings**.

> **On screen:** In the Settings panel, scroll to **Trigger Conditions**. Click **+ Add**. A text box appears.

Trigger conditions use the same expression language as flow expressions. They must evaluate to `true` for the flow to run.

Examples:

| Condition expression | What it filters |
|---------------------|----------------|
| `@equals(triggerBody()?['Importance'], 'High')` | Only high-importance emails |
| `@startsWith(triggerBody()?['subject'], 'URGENT')` | Subject begins with "URGENT" |
| `@not(equals(triggerBody()?['Author']?['DisplayName'], 'System Account'))` | Ignore system-generated changes |
| `@greater(triggerBody()?['Quantity'], 100)` | List item Quantity column exceeds 100 |

You can add multiple conditions — **all must be true** simultaneously (logical AND).

> **On screen:** After adding conditions, click **Done**. The trigger card now shows a small filter icon indicating conditions are active.

---

## Polling vs Webhook Triggers

Power Automate uses two underlying mechanisms to detect events. You do not choose between them — the connector determines which mechanism it uses. Understanding the difference explains the latency and run-frequency characteristics you observe.

### Polling Triggers

The Power Automate service calls the connector's API on a fixed interval to check whether anything new has occurred since the last check. If new items are found, a flow run is started for each one.

```
Power Automate          Connector API
     │                       │
     │──── GET /items ───────►│  (every 3 minutes)
     │◄─── 0 new items ───────│  → no run
     │                       │
     │──── GET /items ───────►│  (3 minutes later)
     │◄─── 2 new items ───────│  → 2 flow runs start
```

- **Latency:** Up to one full polling interval (3 minutes for most connectors on a paid plan; up to 15 minutes on the free plan)
- **Cost:** Every poll counts as an API call to the connector; high-volume connectors may hit rate limits
- **Examples:** SharePoint "When an item is created", MSN Weather (any action used as polling source), SQL Server triggers

### Webhook (Push) Triggers

The connector registers a callback URL with the source service. When the event occurs, the source service sends an HTTP POST directly to Power Automate with the event payload.

```
Source service     Power Automate webhook endpoint
     │                       │
     │ event occurs           │
     │──── POST /callback ───►│  → flow run starts immediately
```

- **Latency:** Near-instant (seconds)
- **Cost:** No repeated polling; the run only starts when the event happens
- **Examples:** Office 365 Outlook "When a new email arrives", Microsoft Forms "When a new response is submitted", HTTP webhook trigger

> **On screen:** To identify which mechanism a trigger uses, open the connector's reference page at `https://learn.microsoft.com/en-us/connectors/[connector-name]/`. Look for the tag **Trigger type: Polling** or **Trigger type: Webhook** in the trigger's metadata section.

**Practical implications:**

| Consideration | Polling | Webhook |
|--------------|---------|---------|
| Latency | Minutes | Seconds |
| Missed events | Caught on next poll | Not missed (service delivers) |
| Rate limit risk | Higher | Lower |
| Connector support | Universal | Requires source support |

---


<div class="compare">
<div class="compare-card">
<div class="header before">Polling</div>
<div class="body">

See detailed comparison in the table above.

</div>
</div>
<div class="compare-card">
<div class="header after">Webhook Triggers</div>
<div class="body">

See detailed comparison in the table above.

</div>
</div>
</div>

## Navigating Trigger Selection in the Portal

### Finding the Right Trigger Quickly

> **On screen:** When creating an **Automated cloud flow**, the "Choose your flow's trigger" panel has a search box at the top. Type a noun describing the event — not the connector name. For example, type `message` to find Teams, Outlook, and SMS-related triggers together.

> **On screen:** Below the search results, the panel shows two tabs: **All** and **Premium**. Switch to **Premium** to see only triggers that require a premium license — useful when scoping what your organization's plan supports.

### Changing a Trigger After Creation

> **On screen:** On the flow canvas, click the trigger card to expand it. Click the **…** menu. Select **Delete**. A dialog warns that downstream steps may break. Confirm deletion. Click **+ Add a trigger** that replaces the empty canvas top.

Changing the trigger family often invalidates dynamic content tokens used in downstream actions — red error markers appear on affected cards. Review each affected card and re-map tokens from the new trigger.

---

## Common Trigger Mistakes

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Using Recurrence when an event-driven trigger exists | Flow runs unnecessarily; event still processed late | Switch to the appropriate automated trigger |
| Not setting a trigger condition on SharePoint "item modified" | Flow fires on every column edit, including system timestamps | Add a condition checking the column that actually matters |
| Forgetting to set a time zone on Recurrence | Flow runs at UTC midnight even when you expected 8 AM local time | Always set the **Time zone** field explicitly |
| Leaving polling interval at default on free plan | 15-minute delay seems like flow is broken | Upgrade plan or accept the latency; document expected delay |
| Adding inputs to Recurrence trigger | Recurrence has no inputs panel — the UI does not show one | Use Manually trigger for flows that need user input |

---

## Connections


<div class="callout-info">
<strong>Info:</strong> This section maps how this guide connects to the broader course. Use these links to navigate related material.
</div>

- **Builds on:** Module 01 — Creating Your First Cloud Flow (trigger basics, canvas navigation)
- **Leads to:** Guide 02 — Connectors Deep Dive (what happens after the trigger fires)
- **Related to:** Module 03 — Expressions and conditions used in trigger condition expressions

---


## Practice Questions

**Question 1 — Conceptual:** Based on the concepts in this guide, explain in your own words why the core technique matters and when you would choose it over alternatives.

**Question 2 — Application:** Sketch out how you would apply the main concept from this guide to a real-world dataset or problem you have encountered. What would you need to watch out for?


## Further Reading

- [Power Automate trigger documentation](https://learn.microsoft.com/en-us/power-automate/triggers-introduction)
- [Polling vs push triggers explained](https://learn.microsoft.com/en-us/connectors/custom-connectors/connection-parameters#triggers)
- [Trigger conditions reference](https://learn.microsoft.com/en-us/power-automate/triggers-introduction#add-a-trigger-condition)
- [Recurrence trigger advanced options](https://learn.microsoft.com/en-us/power-automate/desktop-flows/recurrence)


---

## Cross-References

<a class="link-card" href="./01_trigger_types_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Interactive slide deck covering the key concepts with visual examples.</div>
</a>

<a class="link-card" href="../notebooks/01_list_connectors_api.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises and real data.</div>
</a>
