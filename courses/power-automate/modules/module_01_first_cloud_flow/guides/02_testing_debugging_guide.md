# Testing and Debugging Your Flow

> **Reading time:** ~12 min | **Module:** 1 — First Cloud Flow | **Prerequisites:** Module 0

## In Brief

Power Automate gives you three overlapping tools for understanding why a flow behaved the way it did: the **Test pane** (run it now and watch it live), **Run history** (browse every past execution), and **Input/Output inspection** (see the exact data each step received and produced). This guide walks through all three, plus the Flow Checker for catching configuration errors before the flow ever runs.

<div class="callout-key">
<strong>Key Concept:</strong> Power Automate gives you three overlapping tools for understanding why a flow behaved the way it did: the **Test pane** (run it now and watch it live), **Run history** (browse every past execution), and **Input/Output inspection** (see the exact data each step received and produced). This guide walks through all three, plus the Flow Checker for catching configuration errors before the flow ever runs.
</div>


## Learning Objectives

By the end of this guide you will be able to:

<div class="callout-insight">
<strong>Insight:</strong> By the end of this guide you will be able to:

1.
</div>


1. Trigger a flow manually from the Test pane
2. Read the run history to find failed executions
3. Expand any action card in run detail to inspect its inputs and outputs
4. Interpret the four run states: Succeeded, Failed, Cancelled, Running
5. Use the Flow Checker to catch missing required fields
6. Diagnose and fix the five most common first-flow errors

---

## The Testing Workflow

The recommended cycle for any flow change:

<div class="callout-key">
<strong>Key Point:</strong> The recommended cycle for any flow change:



Never skip the Save step.
</div>


```
Save → Test (manually) → Inspect run detail → Fix issue → Save → Test again
```

Never skip the Save step. The Test pane always runs the last saved version of the flow. Changes you have made on the canvas but not yet saved do not affect the test run.

---

## Step 1 — Open the Test Pane

> **On screen:** With your flow open in the designer, click **Test** in the top toolbar. The Test pane slides in from the right side of the screen.

<div class="callout-info">
<strong>Info:</strong> > **On screen:** With your flow open in the designer, click **Test** in the top toolbar.
</div>


You will see two options:

| Option | When to use it |
|--------|---------------|
| **Automatically** | Reuses the data from the most recent previous run. Useful when iterating on a fix — keeps the same input so results are comparable. |
| **Manually** | You provide the trigger event yourself. Use this for the very first test and whenever you want fresh data. |

> **On screen:** Select **Manually** then click **Test**.

---

## Step 2 — Run the Flow

> **On screen:** A confirmation screen appears. For a scheduled flow it says "Your flow will run right now." Click **Run flow**.

<div class="callout-warning">
<strong>Warning:</strong> > **On screen:** A confirmation screen appears.
</div>


> **On screen:** Click **Done**. Power Automate closes the confirmation and redirects you to the **run detail page** for this execution.

The run detail page is the same page you reach from the Run history — there is no special "test results" view. Everything goes through the same interface.

---

## Step 3 — Read the Run Detail Page

The run detail page shows the flow as a vertical stack of cards, identical to the designer canvas, but each card now displays its execution result.

<div class="callout-insight">
<strong>Insight:</strong> The run detail page shows the flow as a vertical stack of cards, identical to the designer canvas, but each card now displays its execution result.
</div>


### Card States

| Icon | Meaning |
|------|---------|
| Green circle with tick | Step succeeded |
| Red circle with X | Step failed — expand the card to read the error |
| Grey circle | Step was skipped (usually because a condition evaluated to false) |
| Spinning indicator | Step is still running |

> **On screen:** Click any card to expand it. You will see two sections: **Inputs** and **Outputs**.

### Inputs and Outputs

```
┌─────────────────────────────────────────────────────────┐
│  Get current weather                              ✓      │
│  ─────────────────────────────────────────────────────  │
│  INPUTS                                                  │
│    Location:  "London, UK"                               │
│    Units:     "Imperial"                                 │
│  OUTPUTS                                                 │
│    Summary:   "Mostly sunny"                             │
│    Temperature: 72                                       │
│    Humidity:  45                                         │
│    Feels Like: 69                                        │
│    Wind Speed: "8 mph NW"                                │
└─────────────────────────────────────────────────────────┘
```

**Inputs** show the exact values the step received — these are what you configured in the designer plus any dynamic tokens resolved by upstream steps.

**Outputs** show the exact values the step produced — these are what downstream steps consumed as dynamic content tokens.

This symmetry makes debugging precise: if the email body contains the wrong value, expand the email action and check its Inputs section. The dynamic token was resolved to the value shown there. If that value is wrong, expand the upstream step that produced it and check its Outputs.

---

## Step 4 — Browse Run History

> **On screen:** Click the back arrow in the top-left of the run detail page to return to the flow overview page. The **28 day run history** section is in the lower half of this page.

Each row in the run history table shows:

| Column | Description |
|--------|-------------|
| Start | Date and time the flow was triggered |
| Duration | How long the run took from trigger to completion |
| Status | Succeeded / Failed / Cancelled / Running |

> **On screen:** Click any row to open the run detail page for that execution. You can compare runs side-by-side by opening each in its own browser tab.

### The Four Run States

```
Succeeded  — All steps completed without error
Failed     — At least one step encountered an error that halted the flow
Cancelled  — The flow was cancelled manually or by a timeout
Running    — The flow is still executing (refresh to update)
```

A flow shows **Failed** even if only one of ten steps failed, provided that step was not inside a "Configure run after" branch designed to continue on failure. This is important: scroll down the run detail page to find which step has the red X — the failed step is not always the last one in the list.

---

## The Flow Checker

The Flow Checker scans the flow for configuration errors without running it. Run it before every test to catch issues that would cause immediate failure.

> **On screen:** In the flow designer toolbar, click **Flow Checker** (the icon that looks like a checkmark inside a circle, to the right of Test).

The Flow Checker panel opens on the right. Errors are listed with:

- The name of the card that has the problem
- A brief description of what is missing or invalid

Common Flow Checker findings:

| Finding | Cause | Fix |
|---------|-------|-----|
| "Required field 'To' is missing" | Outlook Send Email — To field empty | Click the card, enter email address |
| "Required field 'Location' is missing" | MSN Weather — Location empty | Click the card, enter city or postal code |
| "Connection required" | Connector not authenticated | Click "Sign in" on the connector card |
| "Expression is invalid" | A typed expression has a syntax error | Check any manually typed formulas in the card |

A green "Your flow is ready to use" message means the Flow Checker found no issues. This does not guarantee the flow will succeed at runtime — runtime errors (wrong credentials, API timeouts, data type mismatches) are invisible to the checker.

---

## Diagnosing Common First-Flow Errors

### Error 1 — "InvalidConnectionFields" on Outlook

**What you see:** The Send an email card has a red X. The error message in the Outputs section reads `InvalidConnectionFields` or `The specified object was not found in the store`.

**Cause:** The Office 365 Outlook connection was made with a personal Microsoft account (Outlook.com or Hotmail) instead of a Microsoft 365 work or school account.

**Fix:**
> **On screen:** Go to **Data > Connections** in the left rail. Find the Office 365 Outlook connection. Delete it. Return to the flow and click **Sign in** on the Outlook card. Use your `@yourcompany.com` or `@university.edu` account.

---

### Error 2 — "BadRequest" on MSN Weather

**What you see:** The Get current weather card has a red X. The error message reads `BadRequest` or `The value provided for parameter 'location' is not valid`.

**Cause:** The Location field is empty or contains a value the MSN Weather API cannot geocode.

**Fix:**
> **On screen:** Expand the Get current weather card in the designer. Enter a specific city and country, for example `Paris, France` or `New York, NY`. Save the flow and test again.

---

### Error 3 — Dynamic content shows the wrong value

**What you see:** The flow succeeds and an email arrives, but the body shows an unexpected value — for example, the temperature field shows a date instead of a number.

**Cause:** A dynamic content token from the wrong step was inserted. The Recurrence trigger exposes timestamp outputs that look similar to weather outputs in the dynamic content panel.

**Fix:**
> **On screen:** Open the run detail page. Expand the Send an email card and read the Inputs section. Find the field with the wrong value. Return to the designer and click that field. The incorrect token will be visible as a pill. Click the pill to select it and press Backspace. Re-open the dynamic content panel and insert the correct token from the **Get current weather** group.

---

### Error 4 — Flow runs but no email arrives

**What you see:** The run detail page shows all steps with green ticks, but your inbox is empty.

**Cause (most likely):** The email was sent to a different address — mistyped in the To field — or it was filtered to a spam/junk folder.

**Fix:** Check your spam/junk folder first. Then expand the Send an email card in the run detail page and read the Inputs section. Confirm the **To** field shows exactly the address you intended. If it does not, correct it in the designer and re-run.

---

### Error 5 — "Unauthorized" on Outlook

**What you see:** Send an email card fails with `Unauthorized` or `401`.

**Cause:** The Office 365 Outlook connection token has expired (common if the flow has not run for several months) or the connection was made with credentials that have since changed.

**Fix:**
> **On screen:** Go to **Data > Connections** in the left rail. Find the expired connection — it shows a yellow warning icon. Click it and click **Fix connection**. Re-enter your credentials. Return to the flow and test again.

---

## Reading Error Messages Efficiently

Error messages in Power Automate follow a consistent structure. Learning to read them saves time.

```
Error:
  Code:    InvalidTemplate
  Message: "The template language expression 'body('Get_current_weather')?['temperature']'
             is not valid: ..."
```

| Part | What it tells you |
|------|------------------|
| **Code** | Category of error (InvalidTemplate = expression problem, BadRequest = bad input data, Unauthorized = auth problem) |
| **Message** | Specific description — usually contains the step name and the field name |
| **innerError** | More detail (not always present) — may include the raw HTTP response from an external service |

The step name in the message (shown in single quotes with underscores, like `Get_current_weather`) maps directly to the card title in the designer. Use it to navigate to the right card.

---

## Connections


<div class="callout-info">
<strong>Info:</strong> This section maps how this guide connects to the broader course. Use these links to navigate related material.
</div>

- **Builds on:** Guide 01 — Creating Your First Cloud Flow
- **Leads to:** Module 02 — Triggers and Connectors (connector authentication in depth)
- **Related:** Module 04 — Error handling with "Configure run after" and Scope actions

---


## Practice Questions

**Question 1 — Conceptual:** Based on the concepts in this guide, explain in your own words why the core technique matters and when you would choose it over alternatives.

**Question 2 — Application:** Sketch out how you would apply the main concept from this guide to a real-world dataset or problem you have encountered. What would you need to watch out for?


## Further Reading

- [Power Automate — Troubleshoot a flow](https://learn.microsoft.com/en-us/power-automate/fix-flow-failures)
- [Power Automate — View run history](https://learn.microsoft.com/en-us/power-automate/monitor-manage-processes)
- [Power Automate — Error codes reference](https://learn.microsoft.com/en-us/power-automate/error-reference)


---

## Cross-References

<a class="link-card" href="./02_testing_debugging_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Interactive slide deck covering the key concepts with visual examples.</div>
</a>

<a class="link-card" href="../notebooks/01_trigger_flow_via_http.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises and real data.</div>
</a>
