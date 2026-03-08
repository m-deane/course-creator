# Expression Recipes

Common Power Automate expression patterns. Each recipe includes a description,
the expression, example input, and example output. Paste directly into the
expression editor (the `fx` button in any field).

---

## Date Formatting

### Format a UTC timestamp as a readable date

**When to use:** Display a SharePoint Modified date or a trigger timestamp to a user.

```
formatDateTime(utcNow(), 'MMMM d, yyyy')
```

| Input | Output |
|-------|--------|
| `2026-03-08T14:30:00Z` | `March 8, 2026` |

---

### ISO date only (YYYY-MM-DD)

**When to use:** Storing dates in SharePoint columns or using as a filter key.

```
formatDateTime(utcNow(), 'yyyy-MM-dd')
```

| Input | Output |
|-------|--------|
| `2026-03-08T14:30:00Z` | `2026-03-08` |

---

### Short date and time (12-hour clock)

**When to use:** Notification emails, Teams messages.

```
formatDateTime(utcNow(), 'MM/dd/yyyy hh:mm tt')
```

| Input | Output |
|-------|--------|
| `2026-03-08T14:30:00Z` | `03/08/2026 02:30 PM` |

---

### Add N days to today

**When to use:** Calculating a due date or SLA deadline.

```
formatDateTime(addDays(utcNow(), 5), 'yyyy-MM-dd')
```

| N | Output |
|---|--------|
| `5` | `2026-03-13` |

---

### Subtract N hours from a timestamp

**When to use:** Lookback windows — get items modified in the last 24 hours.

```
formatDateTime(addHours(utcNow(), -24), 'yyyy-MM-ddTHH:mm:ssZ')
```

| Input | Output |
|-------|--------|
| `2026-03-08T14:30:00Z` | `2026-03-07T14:30:00Z` |

---

### Convert UTC to a specific time zone

**When to use:** Showing times in the user's local time zone in emails or cards.

```
convertTimeZone(utcNow(), 'UTC', 'Eastern Standard Time', 'MM/dd/yyyy hh:mm tt')
```

| Input (UTC) | Output (ET) |
|-------------|-------------|
| `2026-03-08T19:00:00Z` | `03/08/2026 02:00 PM` |

Full list of Windows time zone names: https://learn.microsoft.com/windows-hardware/manufacture/desktop/default-time-zones

---

### Day of week (Monday = 1)

**When to use:** Skip weekends in scheduling logic.

```
dayOfWeek(utcNow())
```

| Input | Output |
|-------|--------|
| `2026-03-09` (Monday) | `1` |
| `2026-03-08` (Sunday) | `0` |

---

### Difference between two dates in days

**When to use:** Age of a ticket, overdue calculation.

```
div(sub(ticks(utcNow()), ticks(triggerBody()?['Created'])), 864000000000)
```

Explanation: `ticks()` returns 100-nanosecond intervals; 864000000000 ticks = 1 day.

| Created | Today | Output |
|---------|-------|--------|
| `2026-03-01` | `2026-03-08` | `7` |

---

### First day of the current month

**When to use:** Monthly reports, recurring snapshots.

```
startOfMonth(utcNow(), 'yyyy-MM-dd')
```

| Input | Output |
|-------|--------|
| `2026-03-08` | `2026-03-01` |

---

### Last day of the current month

**When to use:** Month-end closing workflows.

```
formatDateTime(addDays(startOfMonth(addMonths(utcNow(), 1)), -1), 'yyyy-MM-dd')
```

| Input | Output |
|-------|--------|
| `2026-03-08` | `2026-03-31` |

---

## String Manipulation

### Concatenate strings

**When to use:** Building dynamic email subjects or SharePoint item titles.

```
concat('Request #', string(triggerBody()?['ID']), ' — ', triggerBody()?['Title'])
```

| ID | Title | Output |
|----|-------|--------|
| `42` | `Office Supplies` | `Request #42 — Office Supplies` |

---

### Convert to upper case

```
toUpper(triggerBody()?['Status'])
```

| Input | Output |
|-------|--------|
| `approved` | `APPROVED` |

---

### Trim whitespace

**When to use:** Cleaning user-entered data before storing in SharePoint.

```
trim(triggerBody()?['Description'])
```

| Input | Output |
|-------|--------|
| `  hello world  ` | `hello world` |

---

### Extract a substring

**When to use:** Parsing a reference code from a longer string.

```
substring(triggerBody()?['ReferenceCode'], 0, 6)
```

| Input | Start | Length | Output |
|-------|-------|--------|--------|
| `ORD-2026-001` | `0` | `6` | `ORD-20` |

---

### Check if a string contains a keyword

**When to use:** Routing logic based on subject line or description content.

```
contains(toLower(triggerBody()?['Subject']), 'urgent')
```

| Input | Output |
|-------|--------|
| `URGENT: Server down` | `true` |
| `Weekly report` | `false` |

---

### Replace a substring

**When to use:** Sanitising characters, URL encoding.

```
replace(triggerBody()?['FileName'], ' ', '_')
```

| Input | Output |
|-------|--------|
| `My Document.xlsx` | `My_Document.xlsx` |

---

### Split a string into an array

**When to use:** Parsing comma-delimited tags stored in a single text column.

```
split(triggerBody()?['Tags'], ',')
```

| Input | Output |
|-------|--------|
| `finance,procurement,2026` | `["finance","procurement","2026"]` |

---

## Array Operations

### Count items in an array

```
length(body('Get_items')?['value'])
```

| Array length | Output |
|-------------|--------|
| 3 items | `3` |

---

### Filter an array by a property value

**When to use:** Keep only items where Status equals "Active" before looping.

```
filter(body('Get_items')?['value'], item()?['Status'] == 'Active')
```

Note: Use this in a **Select** or **Filter array** data operation action, not inline.

---

### Select (project) specific fields from an array

**When to use:** Reduce a large array to only the fields you need before sending to another system.

In a **Select** data operation action:
- Map input: `body('Get_items')?['value']`
- Map output keys: `ID` → `item()?['ID']`, `Title` → `item()?['Title']`

Result is a new array containing only the mapped fields.

---

### Join an array into a single string

**When to use:** Building a comma-separated list of approver names for an email.

```
join(variables('ApproverNames'), ', ')
```

| Input array | Output |
|------------|--------|
| `["Alice","Bob","Carol"]` | `Alice, Bob, Carol` |

---

### Get the first item from an array

```
first(body('Get_items')?['value'])
```

---

### Get the last item from an array

```
last(body('Get_items')?['value'])
```

---

### Check if an array is empty

**When to use:** Branching logic when a SharePoint query returns no results.

```
empty(body('Get_items')?['value'])
```

| Array | Output |
|-------|--------|
| `[]` | `true` |
| `[{...}]` | `false` |

---

## Null Handling

### Return a default value when a field is null

**When to use:** Optional fields in SharePoint forms that may not be filled in.

```
coalesce(triggerBody()?['ManagerEmail'], 'fallback@contoso.com')
```

| ManagerEmail | Output |
|-------------|--------|
| `null` | `fallback@contoso.com` |
| `alice@contoso.com` | `alice@contoso.com` |

---

### Check if a field is null or empty

```
or(empty(triggerBody()?['Description']), equals(triggerBody()?['Description'], null))
```

---

### Safe navigation through nested JSON

**When to use:** Accessing deeply nested properties where intermediate nodes may be absent.

```
coalesce(triggerBody()?['Metadata']?['Category'], 'Uncategorised')
```

The `?` operator returns null instead of throwing an error when the key is missing.

---

## Type Conversion

### Number to string

```
string(triggerBody()?['Amount'])
```

| Input | Output |
|-------|--------|
| `1500` | `"1500"` |

---

### String to number

```
int(triggerBody()?['QuantityText'])
```

| Input | Output |
|-------|--------|
| `"42"` | `42` |

---

### Boolean to yes/no label

```
if(equals(triggerBody()?['IsUrgent'], true), 'Yes', 'No')
```

| Input | Output |
|-------|--------|
| `true` | `Yes` |
| `false` | `No` |

---

### Parse a JSON string into an object

**When to use:** When an HTTP action returns a JSON body as a raw string.

Use a **Parse JSON** action, or inline:
```
json(body('HTTP')?['rawJsonField'])
```

---

## Conditional Expressions

### Ternary (if/else in one line)

```
if(greater(triggerBody()?['Amount'], 5000), 'Finance approval required', 'Manager approval only')
```

| Amount | Output |
|--------|--------|
| `7500` | `Finance approval required` |
| `2000` | `Manager approval only` |

---

### Nested if (three-way branch)

```
if(
  greater(triggerBody()?['Amount'], 10000),
  'Executive',
  if(
    greater(triggerBody()?['Amount'], 5000),
    'Finance',
    'Manager'
  )
)
```

| Amount | Output |
|--------|--------|
| `15000` | `Executive` |
| `7500` | `Finance` |
| `2000` | `Manager` |

---

### Logical AND

```
and(
  equals(triggerBody()?['Status'], 'Pending'),
  greater(triggerBody()?['Amount'], 1000)
)
```

---

### Logical OR

```
or(
  equals(triggerBody()?['Priority'], 'High'),
  equals(triggerBody()?['IsEscalated'], true)
)
```

---

## Timestamp and Time Zone Patterns

### Current UTC timestamp in ISO 8601

```
utcNow()
```

Output: `2026-03-08T14:30:00.0000000Z`

---

### Unix epoch seconds (for REST APIs that expect epoch time)

```
div(sub(ticks(utcNow()), ticks('1970-01-01T00:00:00Z')), 10000000)
```

| Input | Output |
|-------|--------|
| `2026-03-08T00:00:00Z` | `1741392000` |

---

### Parse an epoch timestamp back to a readable date

```
formatDateTime(addSeconds('1970-01-01T00:00:00Z', triggerBody()?['epochTimestamp']), 'yyyy-MM-dd HH:mm:ss')
```

---

### Check if a date is in the past

```
less(ticks(triggerBody()?['DueDate']), ticks(utcNow()))
```

| DueDate | Output |
|---------|--------|
| `2026-01-01` (past) | `true` |
| `2027-01-01` (future) | `false` |
