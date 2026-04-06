# Loops and Error Handling in Power Automate

> **Reading time:** ~18 min | **Module:** 4 — Branching, Loops & Error Handling | **Prerequisites:** Module 3

## In Brief

Most real automation scenarios involve collections — a list of new orders, a set of approval responses, a batch of files to process. Power Automate provides two loop types to work through collections and repeated conditions: **Apply to Each** and **Do Until**.

Equally important is what happens when something goes wrong. Network timeouts, deleted items, service outages, and bad data are a normal part of production flows. Power Automate's error handling tools — **Configure Run After**, **Scope actions**, **Terminate**, and **Retry policies** — give you fine-grained control over how your flow responds to failure.

<div class="callout-key">

<strong>Key Concept:</strong> Most real automation scenarios involve collections — a list of new orders, a set of approval responses, a batch of files to process. Power Automate provides two loop types to work through collections and repeated conditions: **Apply to Each** and **Do Until**.

</div>


---

## Apply to Each: Iterating Over Arrays

Apply to Each runs a set of actions once for every item in an array. The array can come from anywhere: a SharePoint list query, an array variable, the body of an HTTP response, the results of a filter action.

<div class="callout-insight">

<strong>Insight:</strong> Apply to Each runs a set of actions once for every item in an array.

</div>


### When to Use Apply to Each

- Process each row returned by "Get items" (SharePoint, SQL, Dataverse)
- Send a personalized email to each member in a list
- Create a Teams message for each attachment in an email
- Update each record returned by a search

### Adding Apply to Each

> **On screen:** Click **+ New step**. Search for `apply to each`. Click **Apply to each** (listed under Built-in). The action block appears with a single field labeled **Select an output from previous steps**.

### Configuring the Input

> **On screen:** Click inside the **Select an output from previous steps** field. The dynamic content panel opens. Select the array output from a previous action — for example, `value` from a **Get items** SharePoint action (the `value` token represents the array of returned rows). The token appears in the field.

The array input must be an actual array type. If you accidentally select a single-value output, Power Automate either fails at runtime or wraps the value in a single-element array, depending on the data type.

### Adding Actions Inside the Loop

> **On screen:** Click **+ Add an action** inside the Apply to Each block. Add any actions you need. These actions have access to a new set of dynamic content: the current item's fields. Look for tokens labeled with **Current item** — for example, `Title` (Current item), `Email` (Current item), `ID` (Current item).

Each field of the current array element appears as a separate dynamic content token. For JSON arrays with complex objects, you may need to use expressions like `items('Apply_to_each')?['fieldName']` to access nested properties.

### Apply to Each — Full Example

**Scenario:** Send each person on a SharePoint list a reminder email.

Steps:
1. **Get items** from a SharePoint list (returns an array of list items in `value`)
2. **Apply to each** — input: `value` from the Get items step
3. Inside the loop: **Send an email** — To: `Email` (Current item), Subject: `Reminder: Your task is due`, Body: reference `Title` (Current item)

> **On screen:** Inside the Apply to Each block, add Send an email. In the **To** field, click **Add dynamic content** and select `Email` from the Current item group. Power Automate automatically recognizes that this token refers to the email address of the current loop iteration.

### Concurrency Control in Apply to Each

By default, Apply to Each processes items sequentially — one at a time, in order. This is safe but slow for large arrays.

**Concurrency** setting lets the loop process multiple items simultaneously, just like parallel branches.

> **On screen:** Click the **...** (ellipsis) menu in the top-right corner of the Apply to Each block. Click **Settings**. The **Concurrency Control** toggle appears. Turn it on. A slider labeled **Degree of Parallelism** appears — set it between 1 and 50. Click **Done**.

**Degree of Parallelism settings:**

| Setting | Behavior | Use When |
|---|---|---|
| Off (default) | Sequential, 1 at a time | Order matters, or actions conflict |
| 2–5 | Light parallelism | Moderate-size lists, some ordering concerns |
| 10–20 | Moderate parallelism | Large lists, order doesn't matter |
| 50 (max) | Maximum parallelism | Very large lists, independent items only |

**When NOT to enable concurrency:**

- When actions inside the loop write to the same destination (file, row, variable) — concurrent writes can conflict
- When the order of processing matters (e.g., processing transactions in sequence)
- When the external service has strict rate limits

---

## Do Until: Loop with Exit Condition

Do Until runs its contained actions repeatedly until a condition becomes true — or until it hits a count or time limit. This is the right tool when you do not know in advance how many iterations you need.

<div class="callout-key">

<strong>Key Point:</strong> Do Until runs its contained actions repeatedly until a condition becomes true — or until it hits a count or time limit.

</div>


### When to Use Do Until

- Poll an external system until a job status changes to "Complete"
- Retry an HTTP call until it returns a 200 response
- Wait until a file appears in a folder
- Keep prompting for approval until someone responds

### Adding Do Until

> **On screen:** Click **+ New step**. Search for `do until`. Click **Do until** (Built-in). The block appears with an exit condition at the bottom and **+ Add an action** inside it.

### Configuring the Exit Condition

The exit condition is the test that ends the loop. When this condition is **true**, the loop stops.

> **On screen:** At the bottom of the Do Until block, find the condition row (it shows three fields, same as a standard Condition action). Configure the left value, operator, and right value. For example: `Job Status` is equal to `Completed`.

### Configuring Limits

Do Until has two safety limits that prevent infinite loops:

> **On screen:** Click the **...** ellipsis on the Do Until block and click **Settings** (or look for the **Count** and **Timeout** fields directly on the block). Set:
> - **Count** — maximum number of iterations before the loop stops (default: 60)
> - **Timeout** — maximum total duration using ISO 8601 duration format (default: `PT1H` = 1 hour)

**ISO 8601 duration quick reference:**

| Value | Meaning |
|---|---|
| `PT30S` | 30 seconds |
| `PT5M` | 5 minutes |
| `PT1H` | 1 hour |
| `P1D` | 1 day |
| `PT2H30M` | 2 hours 30 minutes |

If either limit is reached, the loop exits even if the condition is still false. The flow does not automatically fail — it simply exits the loop and continues to the next action.

### Apply to Each vs. Do Until

| Apply to Each | Do Until |
|---|---|
| Iterates over a known array | Repeats until a condition is true |
| Runs exactly N times (N = array length) | Runs an unknown number of times |
| Input: array | Input: condition + limits |
| Good for batch processing | Good for polling, retry logic |

---


<div class="compare">
<div class="compare-card">
<div class="header before">Apply to Each</div>
<div class="body">

See detailed comparison in the table above.

</div>
</div>
<div class="compare-card">
<div class="header after">Do Until</div>
<div class="body">

See detailed comparison in the table above.

</div>
</div>
</div>

## Configure Run After: Controlling Error Flow

Every action in Power Automate has a **Run after** configuration that determines under what conditions it should execute. By default, each action runs only if the previous action **Succeeded**. Configure Run After lets you change this.

<div class="callout-danger">

<strong>Danger:</strong> Every action in Power Automate has a **Run after** configuration that determines under what conditions it should execute.

</div>


### The Four Run After States

| State | Meaning |
|---|---|
| **Succeeded** | Previous action completed without errors |
| **Failed** | Previous action encountered an error |
| **Skipped** | Previous action was skipped (because ITS predecessor failed or was skipped) |
| **Timed Out** | Previous action did not complete within its timeout period |

An action can be configured to run after any combination of these states.

### Opening Configure Run After

> **On screen:** Click the **...** (ellipsis) menu on an action. Select **Configure run after**. A panel appears listing the possible states with checkboxes. Check all states after which this action should run. Click **Done**.

### Common Patterns

**Pattern 1: Always run cleanup (equivalent of `finally`)**

Configure a cleanup action — closing a connection, deleting a temp file, sending a status notification — to run after Succeeded, Failed, AND Timed Out. This ensures cleanup happens regardless of whether the main action worked.

> **On screen:** On the cleanup action, open **Configure run after**. Check all three boxes: **is successful**, **has failed**, **has timed out**. Leave **has been skipped** unchecked only if a skipped state means the cleanup is also not needed.

**Pattern 2: Error notification**

Add a "Send error email" action after a critical action. Configure it to run only after **Failed** and **Timed Out**. This creates an error path that runs only when something goes wrong.

> **On screen:** On the Send error email action, open Configure run after. Uncheck **is successful** and **has been skipped**. Check **has failed** and **has timed out**.

**Pattern 3: Skip subsequent actions on failure**

This is the default behavior. When action A fails, all subsequent actions configured with only "Succeeded" are skipped automatically. You do not need to do anything special — this is Power Automate's default.

### Visual Indicators in the Designer

> **On screen:** After configuring Run After on an action, the connecting line between actions in the designer changes appearance. A line configured for failure shows a red dashed style, indicating "this runs on the error path." A line configured for both success and failure shows a combination style.

---

## Scope Actions: Try/Catch/Finally Patterns

A **Scope** is a container action that groups other actions together. Scopes serve two purposes:

<div class="callout-warning">

<strong>Warning:</strong> A **Scope** is a container action that groups other actions together.

</div>


1. **Organization** — collapse a group of actions to reduce visual clutter in complex flows
2. **Error handling** — catch errors from an entire group of actions with a single handler

When a Scope contains an action that fails (and the failing action is not itself handled), the Scope's overall status becomes Failed. Any action configured to run after the Scope on failure will then execute — giving you a catch block.

### Try/Catch Pattern

```
[Try Scope]
  Action 1
  Action 2
  Action 3
[Catch Scope]  ← Configure Run After: Failed, Timed Out
  Log the error
  Send alert email
  Compensate / roll back
[Finally Scope]  ← Configure Run After: Succeeded, Failed, Timed Out
  Always-run cleanup
```

### Building the Try/Catch Pattern

**Step 1: Add the Try Scope**

> **On screen:** Click **+ New step**. Search for `scope`. Click **Scope** (Built-in). A Scope container block appears. Click the title "Scope" and rename it to `Try`. Add all your primary actions inside this Scope.

**Step 2: Add the Catch Scope**

> **On screen:** Click **+ New step** (below the Try Scope, outside it). Add another Scope action. Rename it `Catch`. Click the **...** on the Catch Scope and select **Configure run after**. Check **has failed** and **has timed out**. Uncheck **is successful** and **has been skipped**. Inside the Catch Scope, add your error-handling actions.

**Step 3: Add the Finally Scope (optional)**

> **On screen:** Click **+ New step** below the Catch Scope. Add a third Scope, rename it `Finally`. Configure Run After: check **is successful**, **has failed**, and **has timed out**. Add cleanup actions inside.

### Accessing Error Details in the Catch Block

Inside the Catch Scope, you can access information about what failed using the `result()` expression:

```
result('Try')
```

This returns an array of action results from the Try Scope. Each result object contains:


The following implementation builds on the approach above:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```json
{
  "name": "ActionName",
  "status": "Failed",
  "error": {
    "code": "ActionFailed",
    "message": "The actual error message"
  }
}
```

</div>
</div>

> **On screen:** Inside the Catch Scope, add a Compose action or a Send email action. In the body, click **Expression** and type `result('Try')[0]['error']['message']` to access the first error's message. This gives you the actual error text to include in your notification.

### Nested Scope Example

A flow that processes orders and must handle partial failures:

```
[Try Scope]
  Get order details (could fail: order not found)
  Calculate totals (could fail: division error)
  Create invoice (could fail: template missing)

[Catch Scope] — runs if any Try action fails
  Compose: "Order processing failed"
  Send email to operations team
  Update order status to "Error"

[Finally Scope] — always runs
  Update processing log with timestamp
```

---

## The Terminate Action: Controlled Flow Ending

The **Terminate** action immediately stops a flow run and sets its final status. It is the Power Automate equivalent of `sys.exit()` or `process.exit()`.

<div class="callout-insight">

<strong>Insight:</strong> The **Terminate** action immediately stops a flow run and sets its final status.

</div>


### Terminate Statuses

| Status | Use Case |
|---|---|
| **Succeeded** | Flow completed its work early; no more actions needed |
| **Failed** | A condition was detected that makes continuing impossible; mark as failed |
| **Cancelled** | Flow was intentionally stopped; treat as neutral (not a failure) |

### Adding Terminate

> **On screen:** Click **+ New step**. Search for `terminate`. Click **Terminate** (Built-in). Configure the **Status** dropdown and optionally fill in the **Code** and **Message** fields. The Code field accepts a short identifier string (e.g., `ITEM_NOT_FOUND`). The Message field accepts a human-readable description.

### Common Terminate Patterns

**Terminate after detecting an invalid state:**

Place a Condition action early in the flow to validate inputs. In the **If No** branch (invalid input), add Terminate with Status = Failed and a descriptive message. The flow stops immediately, preventing downstream actions from running with bad data.

**Terminate with Succeeded when a condition makes further steps unnecessary:**

In an approval flow, if the item was already approved by another path, add a Condition check. If already approved, Terminate with Succeeded — no need to re-run the approval steps.

---

## Retry Policies: Handling Transient Failures

Many external services fail intermittently: timeouts during high load, brief outages, rate limiting. Rather than building manual retry logic, Power Automate lets you configure a **Retry policy** directly on any action.

### Accessing Retry Policy Settings

> **On screen:** Click the **...** (ellipsis) on any action. Click **Settings**. Scroll to the **Retry Policy** section. A **Type** dropdown appears.

### Retry Policy Types

| Type | Behavior |
|---|---|
| **Default** | 4 retries with exponential backoff (platform default) |
| **None** | No retries — action fails immediately on first error |
| **Fixed interval** | Retry N times, waiting the same duration between each attempt |
| **Exponential interval** | Retry N times, doubling the wait time between each attempt (with jitter) |

> **On screen:** Select **Fixed interval** from the Type dropdown. Two additional fields appear: **Count** (number of retries, max 90) and **Interval** (wait time in ISO 8601 format, e.g., `PT30S` for 30 seconds). Click **Done**.

### Retry Policy Guidelines

**Use exponential interval for:**
- External APIs that throttle requests
- Services with temporary overload conditions
- HTTP connector calls to public APIs

**Use fixed interval for:**
- Predictable transient failures (e.g., a system that reboots on a schedule)
- Low-frequency retries where backoff is not needed

**Set None for:**
- Idempotency-sensitive actions where retrying would cause duplicates
- Actions whose failure is definitive (e.g., "item already exists" — retrying won't help)

**Maximum retry count:** 90. For anything requiring more attempts, use a Do Until loop around the action instead.

---

## Common Error Patterns and Solutions

### Pattern 1: "Item not found" on dynamic lookup

**Symptom:** A "Get item" or similar action fails because the ID being looked up no longer exists (deleted between the trigger and the action).

**Solution:** Add a Condition before the lookup action. Use the `outputs()` expression or check a prior existence query. Alternatively, configure the lookup action's **Run After** to handle the failure, and in the catch path update the record to mark it as deleted.

### Pattern 2: Connector timeout on large payloads

**Symptom:** HTTP or SharePoint actions fail with timeout errors when processing large lists or files.

**Solution:** Enable concurrency on Apply to Each loops (reduces total time). Or break the operation into smaller batches using a Do Until loop with pagination.

### Pattern 3: "Concurrent modification" on shared resources

**Symptom:** Two flow runs triggered nearly simultaneously both try to update the same SharePoint item or row, causing one to fail with a conflict error.

**Solution:** Add retry policy (exponential interval, 3–5 retries) to the update action. The second run will retry after the first completes. For critical data, consider implementing an Azure Service Bus queue to serialize updates.

### Pattern 4: Loop runs forever

**Symptom:** A Do Until loop hits its iteration limit but never reaches its exit condition.

**Solution:** Check the exit condition expression — it may reference a variable that is never updated inside the loop. Ensure the loop body contains an action that eventually changes the value the condition tests. Set a reasonable Count and Timeout as safety limits.

### Pattern 5: Apply to Each processes only first item

**Symptom:** Actions inside Apply to Each always use data from the first item, not the current item.

**Solution:** Ensure you are selecting **Current item** tokens from the dynamic content panel, not tokens from earlier actions. The Current item group appears only when you are adding an action inside an Apply to Each block.

---

## Error Handling Architecture Reference

```
Flow run starts
      │
  ┌───▼────────────────────────────────────────┐
  │  Try Scope                                  │
  │  ┌──────────────┐  ┌──────────────────────┐ │
  │  │ Apply to Each│  │ Action with Retry    │ │
  │  │  (with       │  │ Policy (exponential, │ │
  │  │  concurrency)│  │  3 retries)          │ │
  │  └──────────────┘  └──────────────────────┘ │
  └─────────────────────────────────────────────┘
      │
      │ (if Try Scope fails)
      ▼
  ┌───────────────────────────────────────────┐
  │  Catch Scope  ← Run After: Failed, Timed  │
  │  Log error details                         │
  │  Send alert to operations                  │
  │  Update status record to "Error"           │
  └───────────────────────────────────────────┘
      │
      │ (always — Run After: Succeeded + Failed + Timed Out)
      ▼
  ┌───────────────────────────────────────────┐
  │  Finally Scope                             │
  │  Write completion timestamp                │
  │  Release any locks                         │
  └───────────────────────────────────────────┘
      │
  Flow ends
```

---

## Connections to Other Modules

- **Builds on:** Module 04 section 1 — Conditions and Switch (Run After builds on the branching mental model)
- **Builds on:** Module 03 — expressions used inside loop bodies and error messages
- **Leads to:** Module 05 — SharePoint and Excel flows use Apply to Each extensively for row processing
- **Related to:** Module 06 — Approval flows use Do Until to wait for responses and Scope for error handling when approvers don't respond

---


## Practice Questions

**Question 1 — Conceptual:** Based on the concepts in this guide, explain in your own words why the core technique matters and when you would choose it over alternatives.

**Question 2 — Application:** Sketch out how you would apply the main concept from this guide to a real-world dataset or problem you have encountered. What would you need to watch out for?


## Further Reading

- [Loop through arrays with Apply to each](https://learn.microsoft.com/en-us/power-automate/apply-to-each) — Official docs with examples
- [Add loops to flows](https://learn.microsoft.com/en-us/power-automate/add-loops-to-flows) — Covers both Apply to Each and Do Until
- [Handle errors in flows](https://learn.microsoft.com/en-us/power-automate/error-handling-flows) — Configure Run After and Scope patterns
- [Retry policies](https://learn.microsoft.com/en-us/power-automate/connection-retry-policy) — Retry policy configuration reference


---

## Cross-References

<a class="link-card" href="./02_loops_error_handling_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Interactive slide deck covering the key concepts with visual examples.</div>
</a>

<a class="link-card" href="../notebooks/01_flow_patterns_simulator.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises and real data.</div>
</a>
