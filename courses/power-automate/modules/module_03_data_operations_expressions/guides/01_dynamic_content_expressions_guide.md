# Dynamic Content and Expressions

> **Reading time:** ~31 min | **Module:** 3 — Data Operations & Expressions | **Prerequisites:** Module 2

## In Brief

Dynamic content and expressions are the mechanism Power Automate uses to move data between steps and compute new values at runtime. Dynamic content lets you select outputs from previous steps via a point-and-click panel; expressions let you write formulas that transform, compare, and format those values before they reach the next action.

<div class="callout-key">
<strong>Key Concept:</strong> Dynamic content and expressions are the mechanism Power Automate uses to move data between steps and compute new values at runtime. Dynamic content lets you select outputs from previous steps via a point-and-click panel; expressions let you write formulas that transform, compare, and format those values before they reach the next action.
</div>


## Learning Objectives

By the end of this guide you will be able to:

<div class="callout-insight">
<strong>Insight:</strong> By the end of this guide you will be able to:

1.
</div>


1. Open the dynamic content panel and insert output tokens from any upstream step
2. Switch to the expression editor and write valid Power Automate expressions
3. Apply string, date/time, logical, type-conversion, and collection functions
4. Nest expressions to compose multi-step transformations
5. Debug a failing expression using the run history

---

## How Data Flows Between Steps

Every action in a flow exposes **outputs** — structured data produced when that step runs. Downstream steps can consume those outputs via tokens. Power Automate resolves every token at runtime by substituting the actual value before executing the next step.

<div class="callout-key">
<strong>Key Point:</strong> Every action in a flow exposes **outputs** — structured data produced when that step runs.
</div>


```
Step A runs
 └─ produces outputs: { "Subject": "Invoice #1042", "Sender": "ap@vendor.com" }

Step B references @{outputs('Step_A')?['Subject']}
 └─ at runtime becomes: "Invoice #1042"
```

This substitution is what the term **dynamic content** means — content whose value is determined dynamically each time the flow runs, not fixed at design time.

---

## The Dynamic Content Panel

### Opening the Panel

<div class="callout-info">
<strong>Info:</strong> ### Opening the Panel

Every text field in an action card that accepts dynamic values shows a lightning-bolt icon when you click into it.
</div>


Every text field in an action card that accepts dynamic values shows a lightning-bolt icon when you click into it.

> **On screen:** Click inside any text field in an action card — for example, the **Subject** field in an **Send an email** action. A small **lightning bolt** icon appears at the right edge of the field. Click it. Alternatively, the **Add dynamic content** link appears directly below many fields.

The panel that slides out on the right is split into two tabs:

| Tab | Purpose |
|-----|---------|
| **Dynamic content** | Click to insert output tokens from previous steps |
| **Expression** | Type or paste expressions using the formula language |

### Navigating the Dynamic Content Tab

The dynamic content tab organises available tokens under the name of the step that produced them.

> **On screen:** Each step from your flow appears as a collapsible heading. Click the heading to expand the list of tokens it exposes. Tokens are labelled with the output name (for example, **Body**, **Subject**, **From**, **Received Time**). Click a token to insert it into the field.

```
Dynamic content panel
├── Trigger: When a new email arrives
│   ├── Subject
│   ├── From
│   ├── Body
│   ├── Received Time
│   └── Attachments
│
├── Get manager (V2)
│   ├── Display Name
│   ├── Mail
│   └── Job Title
│
└── ...
```

Only tokens from steps that are **upstream** of the current step appear in the panel. Power Automate enforces this to prevent circular references.

### Searching for a Token

> **On screen:** At the top of the dynamic content panel is a **Search** box. Type any word to filter tokens by name across all steps. This is faster than scrolling when your flow has many steps.

---

## The Expression Editor

### Switching to Expressions

<div class="callout-warning">
<strong>Warning:</strong> ### Switching to Expressions

> **On screen:** In the dynamic content panel, click the **Expression** tab.
</div>


> **On screen:** In the dynamic content panel, click the **Expression** tab. A text box appears labelled **fx** with a cursor ready for input. Type your expression here. When finished, click **OK** to insert the result as a token in the field.

Expressions use a language called **Workflow Definition Language (WDL)** — a superset of Azure Logic Apps expressions. Functions follow this pattern:

```
functionName(argument1, argument2, ...)
```

Arguments can be:
- String literals: `'hello'` (single quotes only)
- Numbers: `42` or `3.14`
- Booleans: `true` or `false`
- Dynamic content tokens: `triggerBody()?['Subject']`
- Other expressions (nesting)

### Referencing Dynamic Content Inside Expressions

Inside the expression editor, dynamic content tokens are accessed via their internal path rather than clicking:

| What you want | Expression syntax |
|--------------|-------------------|
| Trigger body field | `triggerBody()?['FieldName']` |
| Action output field | `outputs('Action_Name')?['body']?['FieldName']` |
| A variable | `variables('MyVariable')` |
| Items in a loop | `items('Apply_to_each')` |

> **On screen:** When writing an expression that references a dynamic content value, switch to the **Dynamic content** tab mid-expression to click the token — Power Automate inserts the correct internal path automatically, then switch back to **Expression** to continue typing the wrapping function.

---

## String Functions

### `concat()`

<div class="callout-insight">
<strong>Insight:</strong> ### `concat()`

Joins two or more strings together.
</div>


Joins two or more strings together.

```
concat('Hello, ', triggerBody()?['firstName'], '!')
```

Result: `Hello, Priya!`

**When to use:** Building email subjects, file names, or messages from multiple data pieces.

### `substring()`

Extracts a portion of a string.

```
substring('Invoice-2024-001', 8, 4)
```

Result: `2024`

Arguments: `substring(text, startIndex, length)` — index is zero-based.

### `replace()`

Replaces all occurrences of a search string with a replacement.

```
replace('order_status_report.csv', '_', ' ')
```

Result: `order status report.csv`

### `split()`

Splits a string into an array using a delimiter.

```
split('apple,banana,cherry', ',')
```

Result: `["apple", "banana", "cherry"]`

**When to use:** Parsing comma-separated values from emails or form fields before iterating over items.

### `toLower()` and `toUpper()`

Convert string case.

```
toLower('INVOICE #1042')
```

Result: `invoice #1042`

```
toUpper('approved')
```

Result: `APPROVED`

### `trim()`

Removes leading and trailing whitespace.

```
trim('  purchase order  ')
```

Result: `purchase order`

**When to use:** Normalising user-entered data before comparison or storage.

---

## Date and Time Functions

### `utcNow()`

<div class="callout-key">
<strong>Key Point:</strong> ### `utcNow()`

Returns the current date and time in UTC as an ISO 8601 string.
</div>


Returns the current date and time in UTC as an ISO 8601 string.

```
utcNow()
```

Result: `2024-11-15T09:32:00.0000000Z`

`utcNow()` accepts an optional format string:

```
utcNow('yyyy-MM-dd')
```

Result: `2024-11-15`

### `formatDateTime()`

Formats a datetime value using a .NET custom format string.

```
formatDateTime(triggerBody()?['Received'], 'dddd, MMMM d, yyyy')
```

Result: `Friday, November 15, 2024`

Common format codes:

| Code | Meaning | Example |
|------|---------|---------|
| `yyyy` | Four-digit year | 2024 |
| `MM` | Two-digit month | 11 |
| `MMMM` | Full month name | November |
| `dd` | Two-digit day | 15 |
| `HH` | 24-hour hour | 09 |
| `mm` | Minutes | 32 |
| `ss` | Seconds | 00 |

### `addDays()`

Adds (or subtracts) days from a datetime.

```
addDays(utcNow(), 30)
```

Result: A datetime 30 days in the future.

```
addDays(utcNow(), -7, 'yyyy-MM-dd')
```

Result: A formatted date string 7 days in the past.

Arguments: `addDays(timestamp, days, format?)` — `format` is optional.

Other add functions follow the same pattern: `addHours()`, `addMinutes()`, `addSeconds()`, `addMonths()`, `addYears()`.

### `convertTimeZone()`

Converts a datetime value from one time zone to another.

```
convertTimeZone(utcNow(), 'UTC', 'Eastern Standard Time', 'h:mm tt')
```

Result: `5:32 AM` (when UTC time is 09:32)

Arguments: `convertTimeZone(timestamp, sourceZone, targetZone, format?)`

Time zone names use the Windows time zone identifiers (for example, `'Pacific Standard Time'`, `'Central European Standard Time'`).

---

## Logical Functions

### `if()`

Returns one of two values depending on a condition.

```
if(greater(variables('InvoiceAmount'), 10000), 'Requires VP approval', 'Auto-approved')
```

Arguments: `if(condition, valueIfTrue, valueIfFalse)`

### `equals()`

Returns `true` if two values are equal.

```
equals(triggerBody()?['Status'], 'Approved')
```

Useful as the condition argument inside `if()`.

### `and()` and `or()`

Combine multiple boolean expressions.

```
and(equals(variables('Status'), 'Open'), greater(variables('DaysOpen'), 14))
```

Returns `true` only when both conditions are true.

```
or(equals(variables('Priority'), 'High'), equals(variables('Priority'), 'Critical'))
```

Returns `true` when either condition is true.

### `not()`

Inverts a boolean value.

```
not(empty(triggerBody()?['Attachments']))
```

Returns `true` when the Attachments field is not empty.

### `coalesce()`

Returns the first non-null value from a list of arguments. Useful for providing fallback defaults.

```
coalesce(triggerBody()?['ManagerEmail'], variables('DefaultApproverEmail'))
```

Returns the manager email if it exists, otherwise falls back to the default approver variable.

---

## Type Conversion Functions

Power Automate is strongly typed at runtime. When a number comes in as a string from an external system, arithmetic on it fails unless you convert it first.

### `int()`

Converts a value to an integer.

```
int(triggerBody()?['Quantity'])
```

Converts the string `"5"` to the number `5`.

### `float()`

Converts a value to a floating-point number.

```
float(triggerBody()?['UnitPrice'])
```

### `string()`

Converts a value to a string.

```
string(variables('InvoiceNumber'))
```

Use this before passing a number into a string function like `concat()`.

### `bool()`

Converts a string `"true"` or `"false"` to a boolean.

```
bool(triggerBody()?['IsUrgent'])
```

### `json()`

Parses a JSON string into an object.

```
json(body('HTTP_Request'))
```

After this conversion, you can access properties with `?['propertyName']` notation.

### `xml()`

Parses an XML string into an XML object for use with XPath operations.

```
xml(body('Get_file_content'))
```

---

## Collection Functions

### `length()`

Returns the count of items in an array or characters in a string.

```
length(variables('ApproverList'))
```

Returns `3` if the array has three items.

```
length(triggerBody()?['Subject'])
```

Returns the number of characters in the subject line.

### `first()` and `last()`

Return the first or last item from an array.

```
first(variables('ApproverList'))
```

Returns `"alice@contoso.com"` if that is the first entry.

### `contains()`

Returns `true` if a collection contains a value, or a string contains a substring.

```
contains(variables('ApprovedStatuses'), triggerBody()?['Status'])
```

Returns `true` if the status is in the approved list.

```
contains(triggerBody()?['Subject'], 'URGENT')
```

Returns `true` if the subject line contains the word URGENT.

### `empty()`

Returns `true` if a collection or string has no items/characters.

```
empty(triggerBody()?['Attachments'])
```

Use inside a `not()` to check that attachments exist before trying to process them.

---

## Nesting Expressions

Power Automate expressions can be nested to compose multi-step transformations in a single formula. Each inner function call evaluates first and passes its result to the outer function.

### Example 1: Clean and normalise a name

```
trim(toLower(triggerBody()?['FullName']))
```

Evaluation order:
1. `triggerBody()?['FullName']` → `"  Priya Sharma  "`
2. `toLower(...)` → `"  priya sharma  "`
3. `trim(...)` → `"priya sharma"`

### Example 2: Format a date with timezone conversion

```
formatDateTime(convertTimeZone(utcNow(), 'UTC', 'Eastern Standard Time'), 'MMMM d, yyyy h:mm tt')
```

Evaluation order:
1. `utcNow()` → `"2024-11-15T14:00:00Z"`
2. `convertTimeZone(...)` → `"2024-11-15T09:00:00"` (UTC-5)
3. `formatDateTime(...)` → `"November 15, 2024 9:00 AM"`

### Example 3: Conditional greeting based on time of day

```
if(less(int(formatDateTime(utcNow(), 'H')), 12), 'Good morning', 'Good afternoon')
```

Evaluation order:
1. `utcNow()` → current UTC datetime
2. `formatDateTime(..., 'H')` → `"9"` (hour as string)
3. `int(...)` → `9`
4. `less(9, 12)` → `true`
5. `if(true, ...)` → `"Good morning"`

### Practical Limit

Power Automate enforces a maximum expression length. If your expression exceeds roughly 8,000 characters, break it into multiple **Compose** actions and reference those outputs instead (covered in Guide 02).

---

## Adding Expressions: Step-by-Step Walkthrough

This walkthrough builds an expression that formats the current date for use in an email subject.

### Step 1 — Open the field

> **On screen:** In the **Send an email** action card, click inside the **Subject** field.

### Step 2 — Open the expression editor

> **On screen:** Click the **Expression** tab in the panel that appeared on the right.

### Step 3 — Type the expression

> **On screen:** In the `fx` input box, type:
>
> ```
> concat('Weekly Report - ', formatDateTime(utcNow(), 'MMMM d, yyyy'))
> ```
>
> As you type, the editor does not validate in real time — validation happens when you click **OK**.

### Step 4 — Confirm the expression

> **On screen:** Click **OK** below the expression input box. The field now shows a blue token pill reading `concat('Weekly Report - ',...` instead of the raw expression text.

### Step 5 — Verify in the run history

> **On screen:** Save and run the flow. Navigate to the run detail page. Expand the **Send an email** step. Under **Inputs**, the **Subject** field shows the resolved value:
> `Weekly Report - November 15, 2024`

### Correcting Expression Errors

When an expression contains an error (wrong function name, mismatched parentheses, wrong argument type), the flow fails at runtime and the run history shows a red X on the failing step.

> **On screen:** Click the failed step to expand it. The **Error** section shows a message such as:
> `InvalidTemplate. The template language function 'concatt' is not defined.`
>
> Click the pencil icon on the action card to re-open it. Click the blue expression pill to re-enter the expression editor with the current expression pre-filled. Fix the error and click **OK**.

---

## Common Expression Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Using double quotes instead of single quotes for strings | `InvalidTemplate` error | Replace `"hello"` with `'hello'` |
| Wrong action name in `outputs()` | `ActionNotFound` error | Copy the exact action name from the canvas — spaces replaced with underscores |
| Forgetting `?` before `['field']` | Null reference error on optional fields | Use `triggerBody()?['Field']` not `triggerBody()['Field']` |
| Comparing strings to numbers without converting | Type mismatch error | Wrap the string in `int()` or `float()` before arithmetic |
| Mismatched parentheses | `InvalidTemplate` error | Count opening and closing parens — they must match |

---

## Connections


<div class="callout-info">
<strong>Info:</strong> This section maps how this guide connects to the broader course. Use these links to navigate related material.
</div>

- **Builds on:** Module 02 — Triggers and Connectors (understanding action outputs)
- **Leads to:** Guide 02 — Data Operations (Compose, Variables, Select, Filter)
- **Related to:** Module 04 — Conditions and loops (logical expressions used in conditions)

---


## Practice Questions

**Question 1 — Conceptual:** Based on the concepts in this guide, explain in your own words why the core technique matters and when you would choose it over alternatives.

**Question 2 — Application:** Sketch out how you would apply the main concept from this guide to a real-world dataset or problem you have encountered. What would you need to watch out for?


## Further Reading

- [Power Automate expression reference](https://learn.microsoft.com/en-us/azure/logic-apps/workflow-definition-language-functions-reference)
- [String functions reference](https://learn.microsoft.com/en-us/azure/logic-apps/workflow-definition-language-functions-reference#string-functions)
- [Date and time functions reference](https://learn.microsoft.com/en-us/azure/logic-apps/workflow-definition-language-functions-reference#date-and-time-functions)


---

## Cross-References

<a class="link-card" href="./01_dynamic_content_expressions_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Interactive slide deck covering the key concepts with visual examples.</div>
</a>

<a class="link-card" href="../notebooks/01_expression_reference.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises and real data.</div>
</a>
