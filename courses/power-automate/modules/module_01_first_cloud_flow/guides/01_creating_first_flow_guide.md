# Creating Your First Cloud Flow

## In Brief

A cloud flow is a sequence of automated steps—trigger, optional conditions, and actions—that runs in the Microsoft cloud without any server to manage. In this guide you build a working flow that fetches today's weather and emails it to you every morning.

## Learning Objectives

By the end of this guide you will be able to:

1. Navigate the Power Automate portal to create a new flow
2. Choose and configure a trigger
3. Add and connect actions using dynamic content
4. Save and run a flow manually
5. Read the canvas layout with confidence

---

## What Is a Flow?

Every Power Automate flow follows the same three-part skeleton:

```
Trigger  →  (optional Logic)  →  Actions
```

- **Trigger** — the event that starts the flow (a button press, an email arriving, a schedule)
- **Logic** — optional conditions and loops that control which actions run
- **Actions** — the work the flow does (send an email, create a file, post to Teams)

Microsoft runs this entire sequence in the cloud. You do not provision servers or manage infrastructure.

---

## The Flow Designer Canvas

Before building anything, know the layout of the canvas you will use.

```
┌─────────────────────────────────────────────────────────────────┐
│  TOOLBAR  [Save]  [Test]  [Flow Checker]  [New step]           │
├──────────────┬──────────────────────────────────────────────────┤
│              │                                                   │
│  LEFT PANEL  │                    CANVAS                        │
│              │    ┌──────────────────────┐                      │
│  Connectors  │    │  TRIGGER CARD        │                      │
│  My flows    │    │  Manually trigger    │                      │
│  Templates   │    └──────────┬───────────┘                      │
│              │               │                                   │
│              │    ┌──────────▼───────────┐                      │
│              │    │  ACTION CARD         │                      │
│              │    │  Get current weather │                      │
│              │    └──────────┬───────────┘                      │
│              │               │                                   │
│              │    ┌──────────▼───────────┐                      │
│              │    │  ACTION CARD         │                      │
│              │    │  Send an email       │                      │
│              │    └──────────────────────┘                      │
│              │                                                   │
│              │    [+ New step]                                   │
└──────────────┴──────────────────────────────────────────────────┘
```

Key areas:

| Area | Purpose |
|------|---------|
| Toolbar | Save, test, validate, and navigate back to My flows |
| Canvas (center) | Where you assemble trigger and action cards |
| Left panel | Browse connectors, switch between your flows |
| Each card | Expand/collapse to edit a step's configuration |

---

## Build the Flow: Daily Weather Email

You will create a flow that runs on a schedule, calls the MSN Weather connector to get today's conditions, and sends the result to your own email address via the Office 365 Outlook connector.

### Step 1 — Open Power Automate

> **On screen:** Go to [https://make.powerautomate.com](https://make.powerautomate.com) and sign in with your Microsoft 365 work or school account.

The landing page shows **Home** in the left navigation rail. The centre area displays recommended templates and recent flows.

### Step 2 — Start a New Scheduled Flow

> **On screen:** In the left rail click **+ Create**. The "Start from blank" section appears at the top. Click **Scheduled cloud flow**.

The "Build a scheduled cloud flow" dialog opens.

> **On screen:** Fill in the dialog:
>
> - **Flow name:** `Daily Weather Email`
> - **Starting:** (leave as today's date and current time)
> - **Repeat every:** `1` `Day`
>
> Click **Create**.

Power Automate creates the flow and drops you directly onto the canvas. The first card already shows **Recurrence** — your schedule trigger.

### Step 3 — Inspect the Recurrence Trigger

> **On screen:** Click the **Recurrence** card to expand it.

You will see:

| Field | Value |
|-------|-------|
| Interval | 1 |
| Frequency | Day |
| Time zone | (your local zone) |
| At these hours | (empty — defaults to midnight) |

> **On screen:** In the **At these hours** field type `8` to run the flow at 8 AM every day. Leave all other fields at their defaults. Click anywhere outside the card to collapse it.

### Step 4 — Add the MSN Weather Action

> **On screen:** Below the Recurrence card, click **+ New step**.

The "Choose an operation" search panel opens on the right side of the canvas.

> **On screen:** In the search box type `MSN Weather`. Under **Connectors**, click **MSN Weather**. Then click the action **Get current weather**.

A new card titled **Get current weather** appears on the canvas beneath the trigger.

> **On screen:** Configure the card:
>
> - **Location:** type your city, for example `London, UK`
> - **Units:** select `Imperial` or `Metric` based on your preference

No authentication is required — MSN Weather is a free, Microsoft-managed connector.

### Step 5 — Add the Outlook Send Email Action

> **On screen:** Below the **Get current weather** card, click **+ New step**.

> **On screen:** In the search box type `Send an email`. Under **Connectors**, click **Office 365 Outlook**. Then click the action **Send an email (V2)**.

A new card titled **Send an email (V2)** appears.

> **On screen:** Click inside the **To** field and type your own email address.

> **On screen:** Click inside the **Subject** field and type:

```
Today's Weather:
```

Then click **Add dynamic content** (the lightning bolt icon that appears when you click into a field). From the panel that slides out on the right, select **Summary** under the "Get current weather" group. This inserts `Summary` as a dynamic token — Power Automate will replace it at runtime with the actual weather summary text.

Your subject line now reads: `Today's Weather: [Summary]`

> **On screen:** Click inside the **Body** field. Type the following, inserting dynamic content tokens from the "Get current weather" group where indicated:

```
Good morning!

Here is today's weather for [Location]:

Conditions:   [Summary]
Temperature:  [Temperature] degrees
Feels Like:   [Feels Like] degrees
Humidity:     [Humidity]%
Wind Speed:   [Wind Speed]

Have a great day!
```

To insert each bracketed item: click the cursor into position, click **Add dynamic content**, then select the matching field from the **Get current weather** section in the panel.

### Step 6 — Save the Flow

> **On screen:** Click **Save** in the top toolbar. Wait for the green "Your flow is ready to use" banner.

If any required field is missing, a red banner appears instead and lists the problem. Fix the highlighted card and save again.

### Step 7 — Run a Manual Test

> **On screen:** Click **Test** in the toolbar. Choose **Manually** then click **Test**.

> **On screen:** On the next screen click **Run flow**. Power Automate triggers the flow immediately, bypassing the schedule.

> **On screen:** Click **Done**. You are redirected to the run detail page, which shows each step with a green tick on success or a red X on failure. Check your inbox — the weather email should arrive within 60 seconds.

---

## How Dynamic Content Works

Dynamic content is Power Automate's mechanism for passing data between steps.

```
┌──────────────────────────────────┐
│  Get current weather             │
│  Outputs:                        │
│    - Summary        "Mostly sunny"│
│    - Temperature    72            │
│    - Humidity       45            │
└──────────────────┬───────────────┘
                   │ tokens flow down
┌──────────────────▼───────────────┐
│  Send an email                   │
│  Body: "Temperature: [72]"       │
└──────────────────────────────────┘
```

Each action card exposes its outputs as tokens. Any downstream card can consume those tokens. Power Automate resolves them at runtime and substitutes the real values.

The **dynamic content panel** organises tokens by the step that produces them. Only outputs from steps that are logically upstream of the current card appear in the panel.

---

## Common First-Flow Mistakes

| Mistake | What happens | Fix |
|---------|-------------|-----|
| Wrong account signed in | Outlook action fails with "unauthorized" | Sign out and sign in with your Microsoft 365 work/school account |
| Location field left blank in MSN Weather | Flow fails at the weather step | Enter a city name or postal code |
| Dynamic content from wrong step selected | Email shows placeholder text | In the body, delete the wrong token, re-open dynamic content, and select from the correct step group |
| Flow not saved before testing | Test runs the last saved version | Always click **Save** before **Test** |
| Scheduled time is midnight (default) | Flow runs but you miss it | Set **At these hours** to `8` or your preferred morning hour |

---

## Connections

- **Builds on:** Microsoft 365 account access, basic email familiarity
- **Leads to:** Guide 02 — Testing and Debugging Flows
- **Next module:** Module 02 — Triggers and Connectors (deeper connector catalog)

---

## Further Reading

- [Power Automate documentation — Create a cloud flow](https://learn.microsoft.com/en-us/power-automate/get-started-logic-flow)
- [MSN Weather connector reference](https://learn.microsoft.com/en-us/connectors/msnweather/)
- [Office 365 Outlook connector reference](https://learn.microsoft.com/en-us/connectors/office365/)
