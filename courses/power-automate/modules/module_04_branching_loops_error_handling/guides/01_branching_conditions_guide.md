# Branching and Conditions in Power Automate

> **Reading time:** ~18 min | **Module:** 4 — Branching, Loops & Error Handling | **Prerequisites:** Module 3

## In Brief

Control flow is the mechanism that lets a flow make decisions. Without branching, every flow executes the same actions in the same order every time — which covers a narrow slice of real business scenarios. With branching, a single flow handles many paths: send an approval only if the amount exceeds a threshold; route a ticket to engineering only if the category is "Bug"; run three checks in parallel to save time.

Power Automate provides four branching tools:

| Tool | Use When |
|---|---|
| **Condition** | You need a simple Yes/No split |
| **Switch** | You need three or more distinct paths based on one value |
| **Parallel Branch** | You need multiple paths to run simultaneously |
| **Nested Conditions** | A path itself requires a further decision |

This guide covers all four in depth, including UI walkthroughs for each.

<div class="callout-key">

<strong>Key Concept:</strong> Control flow is the mechanism that lets a flow make decisions. Without branching, every flow executes the same actions in the same order every time — which covers a narrow slice of real business scenarios.

</div>


---

## The Condition Action: Yes/No Branches

The Condition action is the fundamental branching tool. It evaluates a logical expression and routes execution to one of two branches: **If yes** (the expression is true) and **If no** (the expression is false).

<div class="callout-insight">

<strong>Insight:</strong> The Condition action is the fundamental branching tool.

</div>


### Mental Model

Think of a Condition as a railroad switch. The incoming train (your flow run) hits the switch, and based on a physical state (the condition result), it goes left or right. Both tracks can lead to completely different destinations — or one track can lead nowhere (you leave a branch empty).

```
Flow runs up to the condition
           |
    ┌──────▼──────┐
    │  Condition  │
    │  Evaluates  │
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │             │
  YES             NO
    │             │
[Actions]    [Actions]
    │             │
    └──────┬──────┘
           │
    Flow continues
```

After both branches complete, flow execution resumes from the first action placed after the Condition block.

### Adding a Condition to a Flow

> **On screen:** Open your flow in the designer. Click the **+** icon (New step) at the point where you want branching. In the search box that appears, type `condition`. Click **Condition** in the results. A new Condition block appears with two sections labeled **If yes** and **If no**.

The Condition block displays three fields arranged horizontally:

1. **Left value** — the item you are evaluating (a dynamic value from a previous step)
2. **Operator** — the comparison type
3. **Right value** — what you are comparing against


<div class="flow">
<div class="flow-step mint">1. Left value</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">2. Operator</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">3. Right value</div>
</div>

### Configuring the Left Value

> **On screen:** Click inside the first (leftmost) field of the Condition. A dynamic content panel opens on the right side of the screen. All outputs from previous steps appear here as clickable tokens. Click the token you want to evaluate — for example, `Priority` from a SharePoint trigger. The token appears as a blue pill inside the field.

You can also type an expression directly in this field by clicking **Expression** at the top of the dynamic content panel. Expressions use the Power Automate expression language and are evaluated before the comparison runs.

### Operator Reference

| Operator | Meaning | Example use |
|---|---|---|
| `is equal to` | Exact match | Status equals "Approved" |
| `is not equal to` | Any value except | Category not equal to "Test" |
| `is greater than` | Numeric or date comparison | Amount > 5000 |
| `is greater than or equal to` | Inclusive upper bound | Score >= 80 |
| `is less than` | Numeric or date | Days remaining < 3 |
| `is less than or equal to` | Inclusive lower bound | Count <= 10 |
| `contains` | Substring or array membership | Email contains "@contoso.com" |
| `does not contain` | Negated substring | Subject does not contain "Unsubscribe" |
| `starts with` | Prefix match | File name starts with "DRAFT_" |
| `ends with` | Suffix match | Attachment ends with ".pdf" |
| `is null` | Empty/missing value check | Approval date is null |
| `is not null` | Value is present | Manager field is not null |

> **On screen:** Click the operator dropdown (the middle field of the Condition). A list of all operators appears. Scroll to select the one you need.

### Multiple Conditions with AND / OR

A single row compares one pair of values. Real conditions often require checking multiple things. Power Automate lets you add rows and group them.

> **On screen:** At the bottom-left of the Condition block, click **+ Add** and select **Add row** to add another condition check. To change whether rows are combined with AND or OR, click the dropdown that appears between rows — it shows either **And** or **Or**. Click it to toggle. To create groups (AND inside OR), select **Add group** instead of **Add row**.

**Example — multi-condition check:**

Check that an invoice is over $5,000 AND the vendor is not on the approved list:
- Row 1: `Invoice Amount` is greater than `5000`
- Combinator: **And**
- Row 2: `Vendor Status` is not equal to `Approved`

### Adding Actions to Each Branch

> **On screen:** Click **+ Add an action** inside the **If yes** section. The standard action search panel opens. Search for and add whatever actions you need — send an email, update a record, post to Teams, etc. Repeat for the **If no** section. Either section can be left empty if that branch requires no actions.

---

## Configuring Conditions with Expressions

The visual condition builder handles most comparisons. For complex logic — string manipulation, date math, array operations — you need expressions.

<div class="callout-key">

<strong>Key Point:</strong> The visual condition builder handles most comparisons.

</div>


### Opening the Expression Editor

> **On screen:** Inside the left or right value field of a Condition row, click the **fx** icon or click **Expression** tab at the top of the dynamic content panel. A text input appears where you type the expression. Click **OK** to insert it.

### Common Expression Patterns

**String manipulation:**

```
toUpper(triggerBody()?['Title'])
```
Converts a SharePoint item's Title field to uppercase before comparison. Useful when your source data has inconsistent casing.

```
length(triggerBody()?['Description'])
```
Returns the character count of a text field. Compare against a number to enforce minimum length requirements.

**Date and time:**

```
utcNow()
```
Returns the current UTC timestamp. Compare against a date field to check whether a deadline has passed.

```
addDays(utcNow(), 7)
```
Returns a timestamp 7 days from now. Use in the right value to flag items due within a week.

```
dayOfWeek(utcNow())
```
Returns 0 (Sunday) through 6 (Saturday). Compare to 0 or 6 to skip processing on weekends.

**Null and empty checks:**

```
empty(triggerBody()?['AssignedTo'])
```
Returns `true` if the field is null or an empty string. Use with `is equal to` `true` to detect unassigned items.

**Type conversion:**

```
int(triggerBody()?['Quantity'])
```
Converts a text field to an integer before numeric comparison. SharePoint and form fields often deliver numbers as strings.

> **On screen:** In the expression editor, start typing the function name. An autocomplete dropdown appears showing matching functions with their signatures. Click a function to insert it with placeholder argument slots.

---

## Nested Conditions

A nested condition is a Condition block placed inside the **If yes** or **If no** branch of a parent Condition. This lets you ask a follow-up question once you have established a first fact.

<div class="callout-info">

<strong>Info:</strong> A nested condition is a Condition block placed inside the **If yes** or **If no** branch of a parent Condition.

</div>


### When to Use Nesting

Use nesting when the second question only makes sense if the first answer is Yes. For example:

1. Is the request amount over $10,000?
   - **Yes**: Is the requester a department head or above?
     - **Yes**: Auto-approve
     - **No**: Escalate to CFO
   - **No**: Auto-approve

If you tried to model this as a single condition with AND logic, you would lose the ability to handle each combination differently.

### Adding a Nested Condition

> **On screen:** Inside the **If yes** branch of an existing Condition, click **+ Add an action**. Search for and add another **Condition** action. A full Condition block appears nested inside the branch. Configure it exactly as you would a top-level Condition.

### Nesting Depth

Power Automate imposes no hard limit on nesting depth, but deep nesting becomes difficult to read and maintain. If you find yourself nesting three or more levels deep, consider whether a **Switch** action or multiple separate flows would be clearer.

**Practical guideline:** Two levels of nesting is generally the maximum before readability suffers. If you need three levels, refactor.

---

## The Switch Action: Multiple Branches Based on Value

A Switch action evaluates a single value and routes execution to one of several named cases. It is the Power Automate equivalent of a `switch` statement in code.

<div class="callout-insight">

<strong>Insight:</strong> A Switch action evaluates a single value and routes execution to one of several named cases.

</div>


### Condition vs. Switch

| Condition | Switch |
|---|---|
| Binary (Yes/No) | Multiple distinct paths |
| Can use any operator | Equality only — each case matches one value |
| Good for threshold checks | Good for category routing |
| Two branches maximum | Unlimited cases plus a Default |

Use Switch when you have three or more possible values that each need different handling. A chain of nested Conditions trying to do the same thing is harder to read and maintain.

### Adding a Switch Action

> **On screen:** Click **+ New step**. Search for `switch`. Click **Switch** in the results. The Switch block appears with an **On** field at the top and one case labeled **Case** below it.

### Configuring the Switch Value

> **On screen:** Click inside the **On** field at the top of the Switch block. The dynamic content panel opens. Select the output you want to route on — for example, `Status`, `Category`, or `Priority`.

The value in the **On** field is evaluated once. Each Case is then compared against it by equality.

### Adding Cases

> **On screen:** At the bottom of the Switch block, click **+ Add case**. A new case appears with an **Equals** field. Type the exact value this case should match — for example, `Approved`. Click **+ Add an action** inside the case to add the actions that run when the flow hits this case.

Repeat — add as many cases as you need. Case names do not need to be unique, but having duplicate match values causes only the first matching case to run.

### The Default Case

> **On screen:** The Switch block includes a **Default** case at the bottom. Actions placed here run when none of the named Cases match the **On** value. Leave Default empty if an unmatched value requires no action, or add a notification action to alert the team of an unexpected value.

Always populate the Default case in production flows. An unexpected value that silently does nothing is harder to diagnose than one that sends an alert.

### Full Switch Example: IT Ticket Routing

**Scenario:** Route incoming IT tickets to the correct Teams channel based on their Category field.

| Category value | Destination |
|---|---|
| Hardware | Post to `#it-hardware` channel |
| Software | Post to `#it-software` channel |
| Network | Post to `#it-network` channel |
| Security | Post to `#it-security` AND page on-call |
| *(anything else)* | Post to `#it-general` |

> **On screen:** After adding the Switch block:
> 1. Set **On** to the `Category` dynamic value from the ticket trigger
> 2. Add Case 1: Equals `Hardware`, action: Post message to `#it-hardware`
> 3. Add Case 2: Equals `Software`, action: Post message to `#it-software`
> 4. Add Case 3: Equals `Network`, action: Post message to `#it-network`
> 5. Add Case 4: Equals `Security`, actions: Post to `#it-security` + Send urgent notification
> 6. Default case: Post to `#it-general`

---


<div class="compare">
<div class="compare-card">
<div class="header before">Condition</div>
<div class="body">

See detailed comparison in the table above.

</div>
</div>
<div class="compare-card">
<div class="header after">Switch</div>
<div class="body">

See detailed comparison in the table above.

</div>
</div>
</div>

## Parallel Branches: Running Actions Simultaneously

By default, Power Automate runs actions sequentially — each one waits for the previous to finish. Parallel branches break this model: two or more branches run at the same time, and the flow waits for all of them to complete before continuing.

### Why Parallel Branches Matter

If you need to notify three people and each email takes 2 seconds, sequential execution takes 6 seconds. Parallel execution takes 2 seconds. For approval flows with multiple simultaneous reviewers, or for flows that read from several systems before aggregating, parallel branches are the right tool.

### Adding Parallel Branches

> **On screen:** Hover over the connecting line between two sequential actions. A **+** button appears. Click it. Instead of clicking **Add an action**, click **Add a parallel branch**. A side-by-side split appears with two branches. Each branch has its own **+ Add an action** button.

To add a third (or more) parallel branch:

> **On screen:** Hover over the connecting line at the same level as the existing branches. Click **+**. Click **Add a parallel branch** again. A third column appears.

### Execution Behavior

- All branches start simultaneously when the parallel section begins
- The flow does not continue past the parallel block until every branch finishes (or fails)
- If one branch throws an error, by default the entire flow fails

### Practical Limits

- Power Automate does not limit the number of parallel branches, but platform throttling applies per connector
- Branches cannot pass data directly to each other — each branch sees the same inputs from before the parallel block
- If you need to use outputs from parallel branches later in the flow, store them in variables before the parallel section ends

### Example: Parallel Approval Notifications

**Scenario:** When a purchase request is submitted, simultaneously:
- Branch 1: Send email to the department manager
- Branch 2: Post a Teams message to the finance channel
- Branch 3: Create a task in Planner for tracking

All three happen at the same time. The flow continues (to log the request in SharePoint) only after all three complete.

> **On screen:** After the trigger:
> 1. Hover on the first connecting line, click **+**, then **Add a parallel branch**
> 2. In Branch 1: Add **Send an email** action addressed to `Manager Email`
> 3. Hover on the same connecting line again, click **+**, **Add a parallel branch**
> 4. In Branch 2: Add **Post a message in a channel** Teams action
> 5. In Branch 3: Add **Create a task** Planner action
> 6. The single action after the parallel block runs only after all three branches complete

---

## Conditions Inside Switch Cases

Switch cases can contain the full range of Power Automate actions, including Condition blocks. This lets you handle simple routing with Switch while adding nuance inside individual cases.

**Example:** Route tickets by Category (Switch), then inside the Security case check the severity level (Condition) to decide between standard handling and an immediate page.

```
Switch on Category
├── Case "Hardware" → Post to #it-hardware
├── Case "Software" → Post to #it-software
├── Case "Security"
│   └── Condition: Severity equals "Critical"
│       ├── Yes → Page on-call + post to #it-security
│       └── No  → Post to #it-security only
└── Default → Post to #it-general
```

Keep the structure as flat as practical. Each additional level of nesting adds cognitive overhead for anyone maintaining the flow later.

---

## Connections to Other Modules

- **Builds on:** Module 03 — expressions used in condition values; dynamic content from triggers and actions
- **Leads to:** Module 04 section 2 — loops and error handling (Scope, Run After, Configure Run After)
- **Related to:** Module 06 — Approval flows use Condition blocks extensively to route on approval outcomes

---


## Practice Questions

**Question 1 — Conceptual:** Based on the concepts in this guide, explain in your own words why the core technique matters and when you would choose it over alternatives.

**Question 2 — Application:** Sketch out how you would apply the main concept from this guide to a real-world dataset or problem you have encountered. What would you need to watch out for?


## Further Reading

- [Add a condition to a flow](https://learn.microsoft.com/en-us/power-automate/add-condition) — Official walkthrough with screenshots
- [Use expressions in conditions](https://learn.microsoft.com/en-us/power-automate/use-expressions-in-conditions) — Expression reference for Condition fields
- [Power Automate expression reference](https://learn.microsoft.com/en-us/azure/logic-apps/workflow-definition-language-functions-reference) — Full function library (shared with Azure Logic Apps)


---

## Cross-References

<a class="link-card" href="./01_branching_conditions_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Interactive slide deck covering the key concepts with visual examples.</div>
</a>

<a class="link-card" href="../notebooks/01_flow_patterns_simulator.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises and real data.</div>
</a>
