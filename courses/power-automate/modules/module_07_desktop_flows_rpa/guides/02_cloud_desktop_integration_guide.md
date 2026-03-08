# Cloud ↔ Desktop Integration

## In Brief

A desktop flow running alone on a machine is useful for manual tasks but reaches its full potential when a cloud flow orchestrates it — passing data in, receiving results out, and chaining the desktop interaction into a broader automated pipeline. This guide builds that complete integration: a cloud flow triggers a desktop flow that extracts invoice data from a legacy billing application and writes it to Excel, then the cloud flow sends the result for approval.

## Learning Objectives

By the end of this guide you will be able to:

1. Build a desktop flow that extracts data from a desktop application and writes it to Excel
2. Record, edit, and test the desktop flow step by step
3. Create a cloud flow that calls the desktop flow and consumes its output
4. Explain the difference between attended and unattended execution
5. Configure machine groups for unattended flows
6. Apply error handling patterns across the cloud–desktop boundary
7. Describe the hybrid automation pattern and when to use it

---

## The Scenario: Legacy Billing System to Excel

The automation in this guide solves a real-world problem: a finance team manually opens a legacy billing application (a thick-client Windows app with no API), enters a customer ID, reads the outstanding invoice total from the screen, and enters it into an Excel tracker. This happens for dozens of customers every day.

The automation replaces that swivel-chair work with:

```
Cloud flow (Scheduled, 8 AM daily)
  → passes each customer ID to the desktop flow
  → desktop flow opens billing app, looks up total, returns value
  → cloud flow appends result to SharePoint Excel tracker
  → cloud flow sends summary email to finance manager
```

---

## Building the Desktop Flow

### Step 1 — Create a New Desktop Flow

> **On screen:** Open Power Automate Desktop. On the **My flows** home screen, click **New flow** in the top-left. Enter the name `Extract Invoice Data`. Click **Create**. The designer opens with an empty workspace.

### Step 2 — Declare Input and Output Variables

Before recording anything, define the data contract.

> **On screen:** In the **Variables panel** (right side), click **+** → **Input**. Fill in:
>
> | Field | Value |
> |---|---|
> | Variable name | `CustomerID` |
> | Data type | Text |
> | Default value | `C-0001` |
> | Description | Customer account ID to look up in the billing system |
>
> Click **OK**. The variable appears under **Input/Output** at the top of the Variables panel.
>
> Click **+** → **Output**. Fill in:
>
> | Field | Value |
> |---|---|
> | Variable name | `InvoiceTotal` |
> | Data type | Number |
> | Description | Outstanding invoice total in USD from the billing system |
>
> Click **OK**.

### Step 3 — Record the Lookup Sequence

With the legacy billing application open on your desktop:

> **On screen:** In the designer toolbar, click **Record**. The Recorder toolbar appears floating above the taskbar. Position it so it does not obscure the billing application. Click the red **Record** button to begin capture.

Perform the following actions in the billing application:

1. Click the **Customer ID** text field
2. Type a customer ID (use the default `C-0001` during recording)
3. Click the **Search** or **Look Up** button
4. Wait for the results screen to appear
5. Click on the invoice total value to select it (if it is selectable text)

> **On screen:** Click **Done** on the Recorder toolbar. The captured actions appear in the workspace. You will see approximately 4–6 actions: Click UI element (the Customer ID field), Fill text field, Click UI element (the Search button), Wait for UI element (the results area), and possibly a Get UI element attribute action if the recorder captured the read step.

### Step 4 — Replace Hardcoded Customer ID with Variable Reference

The recorder captured the literal string `C-0001` you typed. Replace it with the input variable.

> **On screen:** In the workspace, double-click the **Fill text field** action (the one that typed `C-0001`). In the configuration dialog, find the **Text to fill** field — it shows `C-0001`. Delete that value and type `%CustomerID%`. Press Tab — Power Automate Desktop validates the variable name and shows the variable type. Click **Save**.

### Step 5 — Add the Read Step for Invoice Total

The recorder may not have captured the read step cleanly. Add it explicitly.

> **On screen:** In the Action panel (left side), expand **UI Automation** → **Get details of UI element in window**. Drag it into the workspace below the Wait action. In the configuration dialog, click the **UI element** field, then click **Add UI element**. Switch to the billing application and hover over the invoice total text area. When the blue highlight appears around the correct element, click to capture it. Back in the dialog, set **Attribute name** to `Own Text`. Set **Store the operation result into** to `%RawInvoiceTotal%`. Click **Save**.

### Step 6 — Convert Text to Number

The value read from the UI is a Text string (e.g., `"$4,850.00"`). Convert it to a Number before assigning to the output variable.

> **On screen:** In the Action panel, search for **Convert text to number**. Drag it below the Get details action. In the dialog:
>
> | Field | Value |
> |---|---|
> | Text to convert | `%RawInvoiceTotal%` |
> | Store result into | `%InvoiceTotal%` |
>
> Note: If the text contains currency symbols or commas (e.g., `$4,850.00`), add a **Replace text** action before the conversion to strip those characters:
>
> - Text to parse: `%RawInvoiceTotal%`
> - Text to find: `$` → Replace with: (empty)
> - Repeat for: `,` → Replace with: (empty)
> - Store result into: `%CleanedTotal%`
>
> Then convert `%CleanedTotal%` to a number.

### Step 7 — Test Step by Step

> **On screen:** Make sure the billing application is open and on the screen. Press **F5** to run the flow. Watch each action highlight as it executes. After the flow completes, look at the Variables panel — `%InvoiceTotal%` should show a numeric value (e.g., `4850`). If it shows `0` or an error, press **F10** (Step) from the beginning and watch where it fails.

### Step 8 — Save the Flow

> **On screen:** Press **Ctrl+S**. The flow is saved. Do not close the designer yet — you will return to it if the cloud flow integration reveals issues.

---

## Triggering the Desktop Flow from a Cloud Flow

### Step 1 — Open the Cloud Flow Builder

> **On screen:** Navigate to `make.powerautomate.com`. Click **My flows** → **New flow** → **Scheduled cloud flow**. Name it `Daily Invoice Extract`. Set the recurrence to **1 Day**, starting at **8:00 AM** in your time zone. Click **Create**.

### Step 2 — Add a Variable for the Customer List

For the walkthrough, use a hardcoded array of customer IDs. In production this would come from SharePoint or Dataverse.

> **On screen:** Click **New step** → search for **Initialize variable**. Configure:
>
> | Field | Value |
> |---|---|
> | Name | `CustomerIDs` |
> | Type | Array |
> | Value | `["C-0001","C-0002","C-0003"]` |

### Step 3 — Add an Apply to Each Loop

> **On screen:** Click **New step** → **Control** → **Apply to each**. In the **Select an output from previous steps** field, click inside and use the dynamic content picker to select **CustomerIDs** (the variable from the Initialize variable action).

### Step 4 — Add the Desktop Flow Action Inside the Loop

> **On screen:** Inside the **Apply to each** loop, click **Add an action**. Search for **desktop flows**. Select **Run a flow built with Power Automate Desktop** (this is the action that invokes a desktop flow from a cloud flow).
>
> In the action card, configure:
>
> | Field | Value |
> |---|---|
> | Desktop flow | `Extract Invoice Data` (select from the dropdown) |
> | Run mode | `Attended` (covered below — for now use Attended for testing) |
> | CustomerID | Click in the field → Dynamic content → **Current item** (the loop variable from Apply to each) |
>
> The output section of the action automatically shows `InvoiceTotal` as an available output because you declared it as an output variable in the desktop flow.

### Step 5 — Append the Result to an Excel File via SharePoint

> **On screen:** After the **Run desktop flow** action, still inside the loop, click **Add an action**. Search for **Excel Online (Business)**. Select **Add a row into a table**.
>
> Configure:
>
> | Field | Value |
> |---|---|
> | Location | SharePoint site URL |
> | Document library | Documents |
> | File | `/Finance/InvoiceTracker.xlsx` |
> | Table | `InvoiceData` |
> | CustomerID column | **Current item** (from Apply to each) |
> | InvoiceTotal column | **InvoiceTotal** (from the Run desktop flow action output) |
> | ExtractDate | `utcNow()` expression |

### Step 6 — Send a Summary Email After the Loop

> **On screen:** Outside the loop (click below the Apply to each block), add an action → **Office 365 Outlook** → **Send an email (V2)**. Set To, Subject (`Daily Invoice Extract Complete`), and Body with dynamic content summarizing the run.

---

## Attended vs Unattended Execution

The execution mode determines whether a human must be logged into the machine when the desktop flow runs.

### Attended Mode

| Characteristic | Detail |
|---|---|
| **User required** | Yes — a user must be logged in and have Power Automate Desktop running |
| **Screen interaction** | The flow runs visibly; the user sees actions happening |
| **License required** | Power Automate per-user plan |
| **Trigger source** | Cloud flow or manual run from designer |
| **Best for** | Flows that need the user's active session, Outlook profile, or local credentials |
| **Example** | Finance analyst triggers an invoice extract; the flow runs on their own machine |

> **On screen:** In the **Run a flow built with Power Automate Desktop** action card, the **Run mode** field shows **Attended** by default. When set to Attended, the cloud flow will route execution to a machine where the specified user is currently logged in and has Power Automate Desktop open.

### Unattended Mode

| Characteristic | Detail |
|---|---|
| **User required** | No — the machine runs the flow in the background with no user session |
| **Screen interaction** | The flow runs on a virtual desktop; no physical display required |
| **License required** | Power Automate unattended RPA add-on (per machine) |
| **Trigger source** | Cloud flow only (cannot be triggered manually) |
| **Best for** | Overnight batch processing, scheduled reports, high-volume extraction |
| **Example** | 2 AM nightly batch: extract 500 customer invoices without any user present |

> **On screen:** To switch to Unattended, set **Run mode** to **Unattended** in the action card. You must also configure a **machine** or **machine group** in the action card — the cloud flow needs a target machine identity. Unattended flows use a service account's credentials; ensure the service account has rights to the target application.

### Side-by-Side Comparison

```
Attended                          Unattended
────────────────────────────────  ────────────────────────────────
User logged in at runtime   YES   NO (background session)
Flow visible on screen      YES   NO (virtual display)
License cost                Lower Higher (add-on per machine)
Concurrency per machine     1     Multiple (limited by license)
Service account required    No    Yes
Can run overnight           No    Yes
Good for high volume        No    Yes
```

---

## Machine Management and Machine Groups

### Single Machine Registration

A single machine works for development and low-volume attended flows. Register it through the desktop app Settings panel.

> **On screen:** In Power Automate Desktop, go to **Settings** (gear icon top-right) → **Machine settings**. The machine name appears. Toggle **Machine registration** to **On**. The machine now appears in `make.powerautomate.com` → **Monitor** → **Machines**.

### Machine Groups

Machine groups cluster multiple machines so the cloud flow can send work to whichever machine is available — enabling parallelism and failover.

```
Cloud Flow (calls machine group "InvoiceProcessors")
         │
         ▼
┌─────────────────────────┐
│   Machine Group:        │
│   InvoiceProcessors     │
│                         │
│  ┌─────────┐           │
│  │Machine A│ ← idle     │  ← Cloud flow assigns run to Machine A
│  └─────────┘           │
│  ┌─────────┐           │
│  │Machine B│ ← busy     │  ← Already running another flow
│  └─────────┘           │
│  ┌─────────┐           │
│  │Machine C│ ← idle     │  ← Available as backup
│  └─────────┘           │
└─────────────────────────┘
```

> **On screen:** Navigate to `make.powerautomate.com` → **Monitor** → **Machine groups**. Click **New machine group**. Enter a name (e.g., `InvoiceProcessors`). Click **Add machines** and select the registered machines to include. Save the group. Back in your cloud flow, in the **Run a flow built with Power Automate Desktop** action, change the **Machine or machine group** field from a specific machine to the group name.

### Machine Group Benefits

| Benefit | How It Works |
|---|---|
| **Load balancing** | Cloud flow sends each run to the next available machine |
| **Failover** | If one machine goes offline, runs route to remaining machines |
| **Scalability** | Add more machines to the group to increase throughput |
| **Simplified management** | Update the group once; all cloud flows targeting it benefit |

---

## Error Handling in Desktop Flows

Desktop flows interact with unstable surfaces — applications crash, elements fail to load, data is malformed. Build error handling into every production desktop flow.

### On Block Error

Wrap risky sections in an error-handling block.

> **On screen:** In the Action panel, search for **On block error**. Drag it into the workspace above the risky action sequence. Place the risky actions inside the block. Add an **End** action at the bottom of the block. In the On block error configuration, set:
>
> | Setting | Value |
> |---|---|
> | Name | `InvoiceLookupError` |
> | Capture unexpected logic errors | Enabled |
> | Continue flow run | Yes (to allow error reporting rather than stopping entirely) |
>
> Between the error block and End, add a **Set variable** action: `%InvoiceTotal%` = `-1` (a sentinel value the cloud flow interprets as "lookup failed").

### Retry Logic

For transient failures (application takes too long to load), add a retry loop before declaring an error.

```
Set %RetryCount% = 0
Loop
    Wait for UI element (Results panel) → timeout 5 seconds → store success in %ElementFound%
    If %ElementFound% = True → Break (exit loop)
    Set %RetryCount% = %RetryCount% + 1
    If %RetryCount% >= 3 → Set %InvoiceTotal% = -1, break
End Loop
```

### Error Signaling to the Cloud Flow

Use the output variable to signal errors:

| `%InvoiceTotal%` Value | Meaning |
|---|---|
| Positive number | Successful lookup |
| `-1` | Lookup failed (application error or element not found) |
| `-2` | Customer ID not found in system |

In the cloud flow, after the **Run desktop flow** action, add a **Condition** action checking whether `InvoiceTotal` equals `-1`. Route the failure branch to an email alert or a SharePoint error log row.

---

## Hybrid Automation Patterns

Hybrid automation combines cloud flow orchestration with desktop flow execution. This is the dominant production pattern for organizations with legacy systems.

### Pattern 1: Cloud Trigger → Desktop Execution → Cloud Follow-Up

```
Event in cloud system
(e.g., new order in Dynamics 365)
        │
        ▼
Cloud flow: fetch order details
        │
        ▼
Cloud flow: call desktop flow
  → Pass: OrderID, CustomerName, ProductCode
        │
        ▼
Desktop flow: open legacy ERP
  → Enter order data
  → Submit
  → Capture confirmation number
  → Return: ConfirmationNumber, ERPStatus
        │
        ▼
Cloud flow: update Dynamics 365 record with ConfirmationNumber
        │
        ▼
Cloud flow: send Teams notification to sales team
```

### Pattern 2: Scheduled Batch with Parallelism

```
Cloud flow: Scheduled (nightly 2 AM)
        │
        ▼
Cloud flow: query SharePoint list → get 200 pending records
        │
        ├──────────────────────────────────────────────────────┐
        │                                                      │
        ▼                                                      ▼
Desktop flow (Machine A)                     Desktop flow (Machine B)
  Process records 1–100                        Process records 101–200
        │                                                      │
        └──────────────────────────────────────────────────────┘
                              │
                              ▼
        Cloud flow: merge results, log to SharePoint
                              │
                              ▼
        Cloud flow: email summary to operations manager
```

### Pattern 3: Human-in-the-Loop with Desktop Fallback

```
Cloud flow: attempt API call to modern system
        │
        ├── API succeeds → continue with cloud flow actions
        │
        └── API fails (system down, no credentials) →
                │
                ▼
        Alert operations team via Teams
                │
                ▼
        Instant flow: operations team clicks "Run manual extraction"
                │
                ▼
        Attended desktop flow: analyst's machine opens legacy backup system
                │
                ▼
        Cloud flow: receive results, continue pipeline
```

### When to Use Each Pattern

| Pattern | Use When |
|---|---|
| Cloud trigger → Desktop → Cloud | Single-record processing, API system triggers desktop action |
| Scheduled batch with parallelism | High volume, overnight processing, multiple machines licensed |
| Human-in-the-loop fallback | Modern system unreliable, compliance requires human confirmation |

---

## Full Integration Walkthrough: Verifying End-to-End

Once the cloud flow is built and the desktop flow is saved with input/output variables:

> **On screen:** In the cloud flow builder, click **Test** (top-right of the canvas). Select **Manually** → **Test**. The flow starts immediately. Watch the run history panel for status. When it reaches the **Run a flow built with Power Automate Desktop** action, the status shows **Running** and the machine receives the instruction. Switch to the machine — Power Automate Desktop shows a notification that a desktop flow is running. Watch the billing application open, the customer ID fill in, and the result appear. After the desktop flow completes, the cloud flow continues with the SharePoint append and email steps.

If the machine is not reachable:

> **On screen:** The cloud flow action shows a red failure icon with the error **"The machine is not available or there are no machines in the machine group."** Check Monitor → Machines to verify the machine is Online. If it shows Offline, open Power Automate Desktop on that machine — this typically resolves the issue within 30 seconds.

---

## Common Pitfalls

- **Wrong run mode** — Setting Attended but the machine has no user logged in causes the cloud flow to queue forever. Set Unattended for headless machines.
- **Service account issues** — Unattended flows run as a service account. That account must have rights to the target application and any file paths the desktop flow touches.
- **Concurrency conflicts** — Two cloud flow runs targeting the same attended machine simultaneously: only one can run; the other queues. Use machine groups to parallelize.
- **Output variable not set on error path** — If the desktop flow errors before setting the output variable, the cloud flow receives an empty/null value. Always initialize output variables to a default (e.g., `-1` or `"ERROR"`) at the top of the desktop flow.
- **Application pop-ups** — Legacy apps often show unexpected dialogs (license warnings, update prompts) that block recorded steps. Add Wait for element + dismiss actions to handle known pop-ups.

---

## Connections to Other Modules

- **Builds on:** Guide 01 (desktop flow fundamentals, variables, recording), Module 04 (error handling), Module 06 (approval flows — can be chained after desktop extraction)
- **Leads to:** Module 08 (Copilot in Power Automate — AI-assisted desktop flow generation)
- **Related to:** Module 03 (expressions for processing desktop flow output values in cloud flows)

---

## Further Reading

- [Trigger a desktop flow from a cloud flow](https://learn.microsoft.com/en-us/power-automate/desktop-flows/link-pad-flow-portal) — step-by-step Microsoft documentation
- [Attended vs unattended runs](https://learn.microsoft.com/en-us/power-automate/desktop-flows/run-pad-flow) — licensing and machine requirements
- [Machine groups](https://learn.microsoft.com/en-us/power-automate/desktop-flows/manage-machine-groups) — setup, load balancing, and failover configuration
- [Error handling in desktop flows](https://learn.microsoft.com/en-us/power-automate/desktop-flows/errors) — On block error and exception types
