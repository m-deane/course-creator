# Connector Patterns

Copy-paste configuration patterns for the most frequently used Power Automate connectors.
Each pattern includes the exact settings to enter in the action panel plus an explanation
of why those settings work.

---

## SharePoint

### OData filter: items modified in the last N hours

**Action:** Get items
**Filter Query field:**

```
Modified ge '@{addHours(utcNow(), -24)}'
```

Change `-24` to any negative number of hours. Use single quotes around the expression
because OData expects the date as a string literal.

---

### OData filter: items matching a status column

**Action:** Get items
**Filter Query field:**

```
Status eq 'Approved'
```

For a Choice column named `ApprovalStatus` with a space in the value:

```
ApprovalStatus eq 'In Progress'
```

---

### OData filter: numeric threshold

**Action:** Get items
**Filter Query field:**

```
Amount gt 5000
```

OData operators: `eq` (equals), `ne` (not equals), `gt` (greater than),
`ge` (greater than or equal), `lt` (less than), `le` (less than or equal).

---

### OData filter: combining conditions

**Action:** Get items
**Filter Query field:**

```
Status eq 'Pending' and Amount gt 1000
```

```
Priority eq 'High' or IsEscalated eq true
```

---

### OData filter: text contains (startsWith)

SharePoint REST does not support `contains` natively. Use `startsWith` or retrieve
all items and filter in a **Filter array** data operation.

```
startswith(Title, 'INV-')
```

---

### OData filter: look up by a person column (email)

**Action:** Get items
**Filter Query field:**

```
Author/EMail eq 'alice@contoso.com'
```

For a custom Person column named `AssignedTo`:

```
AssignedTo/EMail eq 'bob@contoso.com'
```

---

### OData filter: items where a lookup column matches

```
Category/Title eq 'Finance'
```

The `/Title` suffix expands the lookup and filters on the display value.

---

### Retrieve only specific columns (reduce payload size)

**Action:** Get items
**Select Columns field:**

```
ID,Title,Status,Amount,Modified,Author/EMail
```

Listing only the columns you need speeds up the action and avoids hitting the
4 MB response size limit on large lists.

---

### Create a file in a document library from HTML content

**Action:** Create file
Setup:
- Site Address: `https://contoso.sharepoint.com/sites/reports`
- Folder Path: `/Reports/2026`
- File Name: `@{formatDateTime(utcNow(), 'yyyy-MM-dd')}_report.htm`
- File Content: output of a previous **Compose** action containing HTML

---

### Update a Person column

**Action:** Update item
- Column internal name: `AssignedTo`
- Value: the user's Azure AD object ID (GUID), not the email address

To look up the ID from email, first use the **Office 365 Users — Get user profile** action.

---

## Outlook

### Filter emails by sender domain

**Action:** When a new email arrives (V3) — Advanced options
**From field:**

```
@contoso.com
```

Uses a suffix match. For an exact sender: `noreply@vendor.com`

---

### Filter emails with attachments only

**Action:** When a new email arrives (V3) — Advanced options
**Has Attachments:** `Yes`

---

### Save all email attachments to SharePoint

**Actions sequence:**

1. Trigger: **When a new email arrives (V3)** — Has Attachments: Yes
2. **Apply to each** — input: `triggerOutputs()?['body/attachments']`
3. Inside loop: **Create file** (SharePoint)
   - File Name: `items('Apply_to_each')?['name']`
   - File Content: `base64ToBinary(items('Apply_to_each')?['contentBytes'])`

---

### Send an email with a dynamic To list from an array variable

**Action:** Send an email (V2)
**To field:**

```
@{join(variables('RecipientEmails'), ';')}
```

Outlook accepts semicolon-delimited addresses in the To field.

---

### Send HTML email with inline table

**Action:** Send an email (V2)
**Body field** — Is HTML: Yes

```html
<table border="1" cellpadding="4" style="border-collapse:collapse">
  <tr><th>ID</th><th>Title</th><th>Status</th></tr>
  <!-- Add rows using an Apply to each + Append to string variable pattern -->
</table>
```

Build the table rows in a string variable using **Append to string variable** inside
an **Apply to each** loop before the send action.

---

## Teams

### Post a message to a channel

**Action:** Post message in a chat or channel
Setup:
- Post as: Flow bot
- Post in: Channel
- Team: select from dropdown
- Channel: select from dropdown
- Message: plain text or HTML

To mention a user: `<at>Alice Smith</at>` (Teams-flavoured HTML only, not standard HTML).

---

### Post an Adaptive Card to a channel (no response needed)

**Action:** Post card in a chat or channel
Setup:
- Post as: Flow bot
- Post in: Channel
- Adaptive Card: paste card JSON (use the templates in `templates/adaptive_card_templates.json`)

---

### Post an Adaptive Card and wait for a response

**Actions sequence:**

1. **Post adaptive card and wait for a response** (Teams)
   - Recipient: email of the person you want to respond
   - Adaptive Card: paste card JSON (must include `Action.Submit` buttons)
   - Update message: `Your response has been recorded. Thank you.`
2. Use `body('Post_adaptive_card_and_wait_for_a_response')?['data']` to access submitted field values

---

### Create a Teams meeting

**Action:** Create a Teams meeting
Setup:
- Subject: `@{triggerBody()?['Title']} — Review Meeting`
- Start time: `@{addHours(utcNow(), 1)}`
- End time: `@{addHours(utcNow(), 2)}`
- Required attendees: semicolon-delimited email list
- Body: meeting agenda in plain text

The action returns a join URL you can include in a notification email.

---

## Excel (Online — Business)

### Get rows from a named table

**Action:** List rows present in a table
Setup:
- Location: OneDrive for Business (or SharePoint)
- Document Library: Documents
- File: select the `.xlsx` file
- Table: select the named table (must be a formal Excel Table, not a range)

Use **$filter** for server-side row filtering:

```
Status eq 'Open'
```

---

### Add a row to a table

**Action:** Add a row into a table
Setup:
- Same location/library/file/table as above
- Row: a JSON object where keys match the table column headers exactly:
  ```json
  {
    "ID": "@{triggerBody()?['ID']}",
    "Title": "@{triggerBody()?['Title']}",
    "Status": "Pending",
    "CreatedDate": "@{formatDateTime(utcNow(), 'yyyy-MM-dd')}"
  }
  ```

---

### Dynamic file path using a date

**Scenario:** Write to a file whose name includes the current month.

In the File field of any Excel action, use:

```
/Reports/@{formatDateTime(utcNow(), 'yyyy-MM')}/monthly_log.xlsx
```

The file and folder must already exist. Power Automate cannot create Excel files.

---

## HTTP Connector

### Call a REST API with Bearer token auth

**Action:** HTTP
Setup:
- Method: GET
- URI: `https://api.example.com/v1/resources`
- Headers:
  ```json
  {
    "Authorization": "Bearer @{body('Get_access_token')?['access_token']}",
    "Accept": "application/json"
  }
  ```

---

### Handle paginated API responses

**Pattern:** Use a **Do Until** loop to iterate through pages.

1. Initialize variable `PageUrl` = first page URL
2. Initialize variable `AllItems` (Array) = `[]`
3. **Do Until** `empty(variables('PageUrl'))`:
   a. **HTTP** GET `variables('PageUrl')`
   b. **Parse JSON** on the response body
   c. **Append to array variable** `AllItems` with `body('Parse_JSON')?['items']`
   d. **Set variable** `PageUrl` = `coalesce(body('Parse_JSON')?['nextPageUrl'], '')`

---

### POST JSON to a webhook

**Action:** HTTP
Setup:
- Method: POST
- URI: `https://hooks.example.com/webhook/abc123`
- Headers:
  ```json
  {
    "Content-Type": "application/json",
    "X-Secret": "@{parameters('WebhookSecret')}"
  }
  ```
- Body:
  ```json
  {
    "event": "item_created",
    "item_id": "@{triggerBody()?['ID']}",
    "timestamp": "@{utcNow()}"
  }
  ```

---

### Handle non-2xx responses from HTTP action

By default the HTTP action marks the flow as failed on any non-2xx response.
To handle errors gracefully:

1. In the HTTP action settings, turn on **Configure run after**.
2. Add a condition action after HTTP:
   - **Run after**: succeeded, failed, skipped, timed out
   - Condition expression:
     ```
     @{outputs('HTTP')?['statusCode']}
     ```
   - Is equal to: `200`
3. In the false branch, log the error or retry.

---

### Authenticate to Azure AD before calling a protected API

**Actions sequence:**

1. **HTTP** (get token):
   - Method: POST
   - URI: `https://login.microsoftonline.com/<TENANT_ID>/oauth2/v2.0/token`
   - Headers: `Content-Type: application/x-www-form-urlencoded`
   - Body:
     ```
     grant_type=client_credentials&client_id=<CLIENT_ID>&client_secret=<CLIENT_SECRET>&scope=<API_SCOPE>
     ```
2. **Parse JSON** on the token response — schema includes `access_token`, `expires_in`
3. Use `body('Parse_JSON')?['access_token']` in subsequent HTTP calls
