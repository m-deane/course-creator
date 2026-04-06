# Excel Integration with Power Automate

> **Reading time:** ~23 min | **Module:** 5 — SharePoint & Excel | **Prerequisites:** Module 2

## In Brief

The Excel Online (Business) connector lets Power Automate read and write structured data in Excel tables stored on SharePoint or OneDrive for Business. You can list rows, add rows, update rows by key, and delete rows — all without opening Excel. This guide covers every table action, data type considerations, and a complete report-generation flow.

<div class="callout-key">

<strong>Key Concept:</strong> The Excel Online (Business) connector lets Power Automate read and write structured data in Excel tables stored on SharePoint or OneDrive for Business. You can list rows, add rows, update rows by key, and delete rows — all without opening Excel.

</div>


## Learning Objectives

By the end of this guide you will be able to:

<div class="callout-insight">

<strong>Insight:</strong> By the end of this guide you will be able to:

1.

</div>


1. Explain the difference between the Excel Online (Business) and Excel Online (OneDrive) connectors
2. Add, list, update, and delete rows in an Excel table from a Power Automate flow
3. Retrieve a specific row using the **Get a row** action and a key column
4. Handle number, date, and text type coercion when writing to Excel
5. Build a flow that generates a weekly summary report in Excel

---

## Prerequisites

- Completed Guide 01 (SharePoint Integration) — the Excel connector shares several patterns with SharePoint
- An Excel workbook stored in SharePoint or OneDrive for Business (not a local file)
- The workbook must contain at least one named **Table** (not just a range)

---

## 1. Connector Overview: Excel Online (Business)

Power Automate provides two Excel connectors. Choose the right one:

<div class="callout-key">

<strong>Key Point:</strong> Power Automate provides two Excel connectors.

</div>


| Connector | File location | Best for |
|-----------|--------------|---------|
| Excel Online (Business) | SharePoint document library or OneDrive for Business (work account) | Enterprise workflows, team-shared workbooks |
| Excel Online (OneDrive) | OneDrive personal (consumer Microsoft account) | Personal automations |

For almost all business automation, use **Excel Online (Business)**.

> **On screen:** When searching for the Excel connector in the action panel, look for the green icon with "Business" in the label. The personal OneDrive version has a lighter icon and no "Business" suffix.

### 1.1 Why the workbook must use a Table

The Excel connector operates exclusively on **named Tables** (Insert → Table in Excel). It cannot read from raw cell ranges or worksheets without a Table. A Table gives every column a programmatic name, which Power Automate uses to map values.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

The following implementation builds on the approach above:

```text
Workbook: Q4_Sales_Report.xlsx
└── Sheet: SalesData
    └── Table: tbl_Sales          ← Power Automate sees this
        ├── Column: OrderDate
        ├── Column: Product
        ├── Column: Quantity
        ├── Column: Revenue
        └── Column: Region
```

</div>

If your workbook does not have a Table, open it in Excel, select the data range including headers, and press **Ctrl+T** (or go to Insert → Table).

---

## 2. Table Actions

### 2.1 List rows present in a table

<div class="callout-info">

<strong>Info:</strong> ### 2.1 List rows present in a table

Retrieves all rows from a named table.

</div>


Retrieves all rows from a named table. This is the primary read action.

> **On screen:** Click **+ New step** → search **Excel Online (Business)** → select **List rows present in a table**.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

The following implementation builds on the approach above:

```text
┌─────────────────────────────────────────────────────────────┐
│  List rows present in a table                           ▲   │
│  ───────────────────────────────────────────────────────    │
│  Location:    [ SharePoint Site - Finance Team     ▼ ]     │
│  Document Library: [ Shared Documents              ▼ ]     │
│  File:        [ Q4_Sales_Report.xlsx               ▼ ]     │
│  Table:       [ tbl_Sales                          ▼ ]     │
│  ▼ Show advanced options                                    │
│  ─────────────────────────────────────────────────────────  │
│  Filter Query:  [ Region eq 'West'                  ]      │
│  Order By:      [ Revenue desc                      ]      │
│  Top Count:     [ 50                                ]      │
│  Skip Count:    [ 0                                 ]      │
└─────────────────────────────────────────────────────────────┘
```

</div>

**Advanced options:**

| Option | Purpose |
|--------|---------|
| Filter Query | OData-style filter — same syntax as SharePoint `Get items` |
| Order By | Column name + `asc` or `desc` |
| Top Count | Maximum rows to return |
| Skip Count | Skip N rows (useful for pagination) |

The action returns a `value` array. Each element is a JSON object whose keys are the column names from the Table header row.

> **Apply to each** is required to process individual rows downstream, just like `Get items` in SharePoint.

### 2.2 Add a row into a table

Appends a new row to the bottom of the table.

> **On screen:** Select **Add a row into a table**. After selecting Location, Library, File, and Table, a field appears for each column in the table.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

The following implementation builds on the approach above:

```text
┌─────────────────────────────────────────────────────────────┐
│  Add a row into a table                                 ▲   │
│  ───────────────────────────────────────────────────────    │
│  Location:    [ SharePoint Site - Finance Team     ▼ ]     │
│  Document Library: [ Shared Documents              ▼ ]     │
│  File:        [ Q4_Sales_Report.xlsx               ▼ ]     │
│  Table:       [ tbl_Sales                          ▼ ]     │
│  ─ Dynamic table columns ─────────────────────────────────  │
│  OrderDate:   [ [utcNow()]                          ]      │
│  Product:     [ [Trigger Product Name]              ]      │
│  Quantity:    [ [Trigger Quantity]                  ]      │
│  Revenue:     [ [Trigger Amount]                    ]      │
│  Region:      [ [Trigger Region]                    ]      │
└─────────────────────────────────────────────────────────────┘
```

</div>

<div class="callout-warning">

<strong>Warning:</strong> Excel tables auto-expand. Power Automate appends to the next available row — you do not need to specify a row number.

</div>

### 2.3 Update a row

Modifies the values in an existing row, identified by its **row ID** (a zero-based integer index assigned by Excel).

> **On screen:** Select **Update a row**. Enter Location, Library, File, Table, and the **Row ID** — an integer retrieved from a previous **List rows** or **Get a row** action.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>


```text
┌─────────────────────────────────────────────────────────────┐
│  Update a row                                           ▲   │
│  ───────────────────────────────────────────────────────    │
│  Location:    [ SharePoint Site - Finance Team     ▼ ]     │
│  Document Library: [ Shared Documents              ▼ ]     │
│  File:        [ Q4_Sales_Report.xlsx               ▼ ]     │
│  Table:       [ tbl_Sales                          ▼ ]     │
│  Row ID:      [ [items('Apply_to_each')?['@row.index']] ]  │
│  Status:      [ Processed                          ]       │
│  ProcessedAt: [ [utcNow()]                         ]       │
└─────────────────────────────────────────────────────────────┘
```

</div>

The `@row.index` token is available in the dynamic content from **List rows present in a table**. It is the row's position in the table, starting at 0.

### 2.4 Delete a row

Removes a row by its row ID.

> **On screen:** Select **Delete a row**. Enter Location, Library, File, Table, and Row ID.

Like the SharePoint **Delete item** action, deletion is permanent. Use a Status column update instead of deletion when you need an audit trail.

### 2.5 Get a row

Retrieves a single row by the value in a designated **Key Column**.

> **On screen:** Select **Get a row**. After selecting the file and table, enter the **Key Column** (e.g., `OrderId`) and the **Key Value** (the specific value to look up in that column).

```text
┌─────────────────────────────────────────────────────────────┐
│  Get a row                                              ▲   │
│  ───────────────────────────────────────────────────────    │
│  Location:     [ SharePoint Site - Finance Team    ▼ ]     │
│  Document Library: [ Shared Documents             ▼ ]     │
│  File:         [ Q4_Sales_Report.xlsx              ▼ ]     │
│  Table:        [ tbl_Sales                         ▼ ]     │
│  Key Column:   [ OrderId                           ]       │
│  Key Value:    [ [Trigger Order ID]                ]       │
└─────────────────────────────────────────────────────────────┘
```

**When to use Get a row vs List rows with a filter:**

| Situation | Use |
|-----------|-----|
| You know the exact key value (e.g., order ID) and need one row | **Get a row** |
| You need all rows matching a condition | **List rows + Filter Query** |
| You need the row ID to later call Update a row | **List rows + Filter Query** → loop → Update a row |

> `Get a row` does not return a row ID. If you need to update the row afterward, use `List rows` with a filter instead, then use the `@row.index` token.

---

## 3. Working with Named Ranges

Named ranges in Excel can be read (but not written) via the Excel connector's **Get a row** action when the range is formatted as a Table. For true named ranges (not Tables), use the **Get a row** action is unavailable — instead, use the **Run script** action (Office Scripts) or the **Send an HTTP request** action with the Excel REST API.

<div class="callout-warning">

<strong>Warning:</strong> Named ranges in Excel can be read (but not written) via the Excel connector's **Get a row** action when the range is formatted as a Table.

</div>


For most automation scenarios, converting named ranges to Tables is the practical solution. Select the range, press **Ctrl+T**, and Power Automate will detect it immediately.

---

## 4. Formatting and Data Type Considerations

Excel stores data in typed cells. Power Automate sends values as strings, and Excel performs type conversion automatically — but only if the destination column's format matches.

<div class="callout-insight">

<strong>Insight:</strong> Excel stores data in typed cells.

</div>


### 4.1 Numbers

Pass numeric values as numbers, not strings. If a column is formatted as Number or Currency in Excel:

```text
Quantity:  5          ← correct (integer)
Revenue:   1299.99    ← correct (decimal)
```

Passing `"1,299.99"` (a string with a comma) will store as text, breaking sum formulas.

**Expression to convert a string to a number:**

```text
float(triggerBody()?['amount'])
```

### 4.2 Dates

Excel stores dates as serial numbers internally. Pass dates in ISO 8601 format and Excel will interpret them correctly if the destination column is formatted as Date:

```text
OrderDate:  2024-03-15
```

Do not include time components for date-only columns — `2024-03-15T14:30:00Z` may appear as a decimal in a Date-formatted column. Use the `formatDateTime` expression to strip the time:

```text
formatDateTime(utcNow(), 'yyyy-MM-dd')
```

### 4.3 Boolean / Yes-No

Excel does not have a native Boolean column type. Use `TRUE` / `FALSE` (uppercase strings) for columns formatted as text, or `1` / `0` for numeric columns.

### 4.4 Column names with spaces

If an Excel table column is named "Order Date" (with a space), the dynamic content token will show it as `Order Date` but the internal key in the JSON response is `Order Date` (space preserved). When referencing it in an expression, quote the key:

```text
items('Apply_to_each')?['Order Date']
```

---

## 5. Build a Report Generation Flow

This flow runs every Monday morning, reads all orders from the previous week in an Excel table, calculates a weekly summary, and writes the summary row to a separate Summary table in the same workbook.

### Architecture overview

```text
Recurrence trigger: every Monday at 7:00 AM
         ↓
List rows (tbl_Orders) filtered by last 7 days
         ↓
Initialize variables: TotalRevenue = 0, OrderCount = 0
         ↓
Apply to each row:
    ├── Add revenue to TotalRevenue variable
    └── Increment OrderCount variable
         ↓
Add a row into tbl_WeeklySummary:
    WeekEnding = formatDateTime(utcNow(), 'yyyy-MM-dd')
    TotalOrders = OrderCount variable
    TotalRevenue = TotalRevenue variable
    AverageOrder = TotalRevenue / OrderCount
         ↓
Send email: "Weekly report updated — [OrderCount] orders, $[TotalRevenue] revenue"
```

### Step 1 — Prepare the workbook

Your Excel workbook needs two tables:

| Table name | Columns |
|------------|---------|
| `tbl_Orders` | OrderDate, Product, Quantity, Revenue, Region |
| `tbl_WeeklySummary` | WeekEnding, TotalOrders, TotalRevenue, AverageOrder |

> **On screen:** Open the workbook in Excel Online → Insert → Table for each range. Name each table via the Table Design tab → Table Name field.

### Step 2 — Create the flow

> **On screen:** In Power Automate: **+ Create** → **Scheduled cloud flow** → Name: `Weekly Sales Report` → Repeat every: `1 Week` → Starting: next Monday → click **Create**.

```text
┌─────────────────────────────────────────────────────────────────┐
│  Recurrence                                                  ▲  │
│  ─────────────────────────────────────────────────────────────  │
│  Interval:   [ 1          ]                                     │
│  Frequency:  [ Week       ]                                     │
│  Time zone:  [ UTC        ]                                     │
│  At these hours:   [ 7    ]                                     │
│  On these days:    [ Monday]                                    │
└─────────────────────────────────────────────────────────────────┘
```

### Step 3 — List orders from the past 7 days

> **On screen:** Add **Excel Online (Business)** → **List rows present in a table** → expand **Show advanced options**.

Filter Query:

```text
OrderDate ge '@{formatDateTime(addDays(utcNow(), -7), 'yyyy-MM-dd')}'
```

> Type the expression directly or use the expression editor (fx icon). The `@{...}` syntax inlines the expression result as a string into the OData filter.

### Step 4 — Initialize variables

> **On screen:** Add **Initialize variable** × 2.

```text
┌─────────────────────────────────────┐
│  Initialize variable                │
│  Name:    TotalRevenue              │
│  Type:    Float                     │
│  Value:   0                         │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  Initialize variable                │
│  Name:    OrderCount                │
│  Type:    Integer                   │
│  Value:   0                         │
└─────────────────────────────────────┘
```

### Step 5 — Loop and accumulate

> **On screen:** Add **Apply to each** → select `value` from the List rows action. Inside the loop, add two **Increment variable** actions.

```text
Apply to each: [value from List rows]
│
├── Increment variable
│   Name:  TotalRevenue
│   Value: [items('Apply_to_each')?['Revenue']]   ← dynamic content token
│
└── Increment variable
    Name:  OrderCount
    Value: 1
```

### Step 6 — Write the summary row

> **On screen:** Outside the Apply to each loop (below it), add **Excel Online (Business)** → **Add a row into a table**.

```text
┌─────────────────────────────────────────────────────────────────┐
│  Add a row into a table                                      ▲  │
│  ─────────────────────────────────────────────────────────────  │
│  Location:         [ SharePoint Site - Finance Team    ▼ ]     │
│  Document Library: [ Shared Documents                  ▼ ]     │
│  File:             [ Q4_Sales_Report.xlsx               ▼ ]    │
│  Table:            [ tbl_WeeklySummary                  ▼ ]    │
│  WeekEnding:       [ formatDateTime(utcNow(),'yyyy-MM-dd') ]   │
│  TotalOrders:      [ [variables('OrderCount')]          ]      │
│  TotalRevenue:     [ [variables('TotalRevenue')]         ]     │
│  AverageOrder:     [ div(variables('TotalRevenue'),      ]     │
│                      variables('OrderCount'))                   │
└─────────────────────────────────────────────────────────────────┘
```

> For `AverageOrder`, use the expression `div(variables('TotalRevenue'), variables('OrderCount'))`. Guard against division-by-zero by wrapping in a Condition: if `OrderCount` is 0, pass 0 for the average.

### Step 7 — Send notification email

> **On screen:** Add **Office 365 Outlook** → **Send an email (V2)**.

```text
To:       [your email]
Subject:  Weekly report updated — @{variables('OrderCount')} orders
Body:
  Week ending: @{formatDateTime(utcNow(), 'MMMM d, yyyy')}
  Total orders: @{variables('OrderCount')}
  Total revenue: $@{variables('TotalRevenue')}
  Average order: $@{div(variables('TotalRevenue'), variables('OrderCount'))}

  The full report is available in Q4_Sales_Report.xlsx.
```

### Step 8 — Save and test

> **On screen:** Click **Save**. To test immediately, click **Test** → **Manually** → **Run flow**. Check the run history to confirm all steps succeed, then open the Excel workbook to verify the new summary row was appended.

---

## 6. Excel vs SharePoint — When to Use Which

A common design question is whether to store data in an Excel table or a SharePoint list. Here is a practical guide:

| Consideration | Excel table | SharePoint list |
|---------------|-------------|-----------------|
| Primary interface | Spreadsheet users who want to work in Excel | Users working in SharePoint / Teams |
| Row limit (practical) | ~1 million rows, but flows slow above 50k | Effectively unlimited with indexing |
| Filtering performance | Good with OData; no indexing required | Better for large datasets with indexed columns |
| Formulas and calculations | Native Excel formulas | Calculated columns, but limited |
| Column types | Text, Number, Date, Boolean | Choice, Person, Lookup, Managed Metadata, Rich Text |
| Versioning | Workbook-level version history | Per-item version history |
| Permissions | Workbook-level only | Per-item, per-column, per-view |
| Real-time collaboration | Multiple editors can conflict | Built for concurrent editing |
| Best for | Reports, financial models, datasets already in Excel | Business records, workflows, approval items |

**Rule of thumb:** If the data is consumed primarily in Excel (charts, formulas, pivot tables), keep it in Excel. If the data drives workflows, approvals, or is queried by multiple systems, use SharePoint.

---


<div class="compare">
<div class="compare-card">
<div class="header before">6. Excel</div>
<div class="body">

See detailed comparison in the table above.

</div>
</div>
<div class="compare-card">
<div class="header after">SharePoint — When to Use Which</div>
<div class="body">

See detailed comparison in the table above.

</div>
</div>
</div>

## 7. Common Pitfalls

- **Action fails with "Table not found":** The table must be a formal Excel Table (Insert → Table), not just a range with headers. Confirm by checking if the Table Design tab appears when you click inside the data.
- **Numbers stored as text:** Ensure numeric columns are formatted as Number in Excel before writing. Use the `float()` or `int()` expression to cast values from dynamic content tokens.
- **Date column shows serial number (e.g., 45365):** The Excel column is not formatted as Date. Format it in Excel, or pass the date as a pre-formatted string and use a Text column instead.
- **List rows returns 0 rows but data exists:** Check the Filter Query carefully — the column name in the filter must exactly match the Table header (case-sensitive).
- **Apply to each iterates over characters, not rows:** You fed a string token (e.g., a single column value) into Apply to each instead of the `value` array from List rows. Always select the `value` array output from the List rows action.
- **Update a row changes the wrong row:** The Row ID (`@row.index`) is the zero-based index within the table at the time of the List rows call. If another flow adds rows concurrently, indices can shift. For stable updates, use a unique key column and **Get a row** to resolve the row first.

## Connections


<div class="callout-info">

<strong>Info:</strong> This section maps how this guide connects to the broader course. Use these links to navigate related material.

</div>

- **Builds on:** Guide 01 (SharePoint Integration) — OData filter syntax is identical
- **Leads to:** Module 06 — Approval flows that update Excel-based tracking sheets
- **Related to:** Module 03 — Variables, expressions, and `formatDateTime`


## Practice Questions

**Question 1 — Conceptual:** Based on the concepts in this guide, explain in your own words why the core technique matters and when you would choose it over alternatives.

**Question 2 — Application:** Sketch out how you would apply the main concept from this guide to a real-world dataset or problem you have encountered. What would you need to watch out for?


## Further Reading

- [Excel Online (Business) connector reference](https://learn.microsoft.com/en-us/connectors/excelonlinebusiness/)
- [Work with Excel tables in Power Automate](https://learn.microsoft.com/en-us/power-automate/excel-connect)
- [Office Scripts with Power Automate](https://learn.microsoft.com/en-us/office/dev/scripts/develop/power-automate-integration)


---

## Cross-References

<a class="link-card" href="./02_excel_integration_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Interactive slide deck covering the key concepts with visual examples.</div>
</a>

<a class="link-card" href="../notebooks/01_sharepoint_graph_api.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises and real data.</div>
</a>
