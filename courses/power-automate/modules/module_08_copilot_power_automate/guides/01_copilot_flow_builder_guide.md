# Copilot in Power Automate: Building Flows with Natural Language

> **Reading time:** ~18 min | **Module:** 8 — Copilot & Power Automate | **Prerequisites:** Module 4

## In Brief

Copilot in Power Automate lets you describe what you want a flow to do in plain English and watch it build the structure for you. Instead of searching through hundreds of connectors and manually wiring triggers to actions, you type a sentence and Copilot generates a working starting point that you then review and refine.

This guide covers how to access Copilot, how to craft effective prompts, how to use Copilot for expressions and troubleshooting, and where its limits are so you know when to switch to manual editing.

<div class="callout-key">

<strong>Key Concept:</strong> Copilot in Power Automate lets you describe what you want a flow to do in plain English and watch it build the structure for you. Instead of searching through hundreds of connectors and manually wiring triggers to actions, you type a sentence and Copilot generates a working starting point that you then review and refine.

</div>


---

## What Copilot in Power Automate Is

Copilot is an AI assistant embedded directly into the Power Automate interface. It understands natural language descriptions of automation scenarios and translates them into flow structures: triggers, actions, conditions, and connector configurations.

<div class="callout-insight">

<strong>Insight:</strong> Copilot is an AI assistant embedded directly into the Power Automate interface.

</div>


Copilot in Power Automate can:

- **Create flows from scratch** — describe a scenario and it generates a complete flow structure
- **Edit existing flows** — ask it to add a step, change a condition, or restructure logic
- **Explain flows** — ask "what does this flow do?" and it summarizes the logic in plain English
- **Generate expressions** — ask it to write a Power Automate expression for a specific calculation
- **Troubleshoot errors** — paste an error message and ask what caused it and how to fix it

Copilot is not a separate product. It is built into the standard Power Automate interface and available on plans that include Copilot capabilities (Microsoft 365 with Copilot licensing, or Power Platform premium).

---

## Accessing Copilot

### From the Home Page

<div class="callout-key">

<strong>Key Point:</strong> ### From the Home Page

> **On screen:** After signing into `make.powerautomate.com`, you land on the Home page.

</div>


> **On screen:** After signing into `make.powerautomate.com`, you land on the Home page. In the center of the page there is a large text input field labeled "Describe what you'd like to automate" with a lightning bolt icon. This is the primary Copilot entry point.

Type your automation description directly into this field and press Enter or click the arrow button. Copilot processes your description and presents a suggested flow with an explanation of what it created.

### From the Create Page

> **On screen:** Click **+ Create** in the left navigation panel. On the Create page you see three sections: "Start from blank," "Start from a template," and "Start from a description." The third option, "Start from a description," uses Copilot.

1. Click **+ Create** in the left navigation
2. Select **Describe it to design it** under "Start from a description"
3. A text area opens with example prompts shown below it
4. Type your description and press **Generate**

### From Inside the Flow Designer (Copilot Panel)

> **On screen:** When you are inside a flow (either editing an existing one or reviewing a Copilot-generated one), there is a **Copilot** button in the top-right toolbar — a sparkle icon labeled "Copilot." Clicking it opens the Copilot panel on the right side of the designer.

The Copilot panel inside the designer is a chat interface. You can:
- Ask it to add or remove steps
- Ask it to explain what a specific action does
- Ask for help with an expression
- Paste an error message and ask for diagnosis

---

## Creating a Flow Using Copilot: Step by Step

### Step 1: Write Your Prompt

<div class="callout-info">

<strong>Info:</strong> ### Step 1: Write Your Prompt

Navigate to **Home** and type your automation description.

</div>


Navigate to **Home** and type your automation description. Write in plain English as if you are explaining the task to a colleague. Include:
- The trigger (what starts the flow)
- The actions (what happens)
- Any conditions or routing logic

**Example prompt:**
```
When I receive an email with an attachment, save the attachment to OneDrive
and send me a notification in Microsoft Teams
```

> **On screen:** As you type, the text area stays open. There is no live preview yet — Copilot waits until you submit before generating anything.

### Step 2: Review the Generated Flow Structure

> **On screen:** After submitting, the screen transitions to the flow designer. The flow is already populated with steps. At the top you will see a banner: "Copilot created this flow based on your description. Review it before saving." Below the banner, the Copilot panel is open on the right with a chat summary of what was generated.

Copilot generates a flow with:
- The trigger it inferred (in this case: "When a new email arrives (V3)" from the Outlook connector)
- Actions in sequence (in this case: "Create file" from OneDrive, then "Post message in a chat or channel" from Teams)
- Placeholder values in angle brackets that you need to fill in, such as `[Folder Path]` or `[Team Name]`

Read through each step. Click on any action to expand its configuration panel and review the settings.

### Step 3: Fill In the Placeholders

Copilot generates the structure but cannot know your specific folder paths, team names, or email addresses. Placeholders appear as highlighted fields or as text in square brackets.

> **On screen:** Click on the "Create file" action to expand it. You will see fields labeled "Site Address," "Folder Path," and "File Name" — each showing a placeholder value or left blank. Click each field to configure it using the dynamic content picker that appears.

For the email-to-OneDrive example:
- **Site Address**: Select your SharePoint site from the dropdown
- **Folder Path**: Type or browse to your target folder (e.g., `/Email Attachments`)
- **File Name**: Use the dynamic content picker to select `Attachment Name` from the trigger
- **File Content**: Use dynamic content `Attachment Content` from the trigger

### Step 4: Test and Refine

Before saving, use the **Test** button in the top-right of the designer to run the flow against a real trigger.

> **On screen:** Click **Test** in the top-right corner. A side panel opens asking "How would you like to test this flow?" Select **Manually** to trigger the flow yourself, or **Automatically** to trigger it using the last event. Click **Test** to start.

After the test run, each step shows a green check (success) or red X (failure) with the input and output data it processed. Review this output to verify the flow behaved as expected.

---

## Using Copilot to Refine an Existing Flow

Once inside the flow designer with the Copilot panel open, you can modify the flow by describing the change in plain English.

<div class="callout-warning">

<strong>Warning:</strong> Once inside the flow designer with the Copilot panel open, you can modify the flow by describing the change in plain English.

</div>


### Adding a Step

In the Copilot chat panel, type:

```
Add a step that only saves the attachment if it is a PDF file
```

Copilot responds by:
1. Explaining what it will add (a Condition action checking if the attachment name ends with `.pdf`)
2. Inserting the new Condition into the flow between the trigger and the OneDrive action
3. Moving the OneDrive and Teams actions into the "Yes" branch of the condition

> **On screen:** After submitting the prompt, the flow diagram updates in real time. You see a new diamond-shaped Condition block appear, with "Yes" and "No" branches. The "Yes" branch contains the original actions.

### Changing an Action

```
Change the Teams notification to also include the file name and the sender's email address
```

Copilot updates the message body of the Teams action to include the relevant dynamic content fields from the trigger.

### Removing a Step

```
Remove the Teams notification step
```

Copilot removes the action and confirms what it changed.

---

## Using Copilot to Generate Expressions

Power Automate expressions (used in the expression editor) follow a specific syntax for date formatting, string manipulation, conditionals, and data conversion. Writing them manually requires knowing the expression language. Copilot can write them for you.

<div class="callout-insight">

<strong>Insight:</strong> Power Automate expressions (used in the expression editor) follow a specific syntax for date formatting, string manipulation, conditionals, and data conversion.

</div>


Open the Copilot panel and ask:

```
Write an expression that formats the current date as MM/DD/YYYY
```

Copilot responds with:

```
formatDateTime(utcNow(), 'MM/dd/yyyy')
```

And explains: `utcNow()` returns the current UTC timestamp, and `formatDateTime()` applies the format pattern.

To use this expression:
1. Click the field in your flow action where you want the formatted date
2. Switch to **Expression** mode in the dynamic content picker
3. Paste the expression Copilot provided
4. Click **OK**

> **On screen:** In the dynamic content picker, there are two tabs: "Dynamic content" and "Expression." Click **Expression** to switch to the formula bar. Paste the expression text and click **OK** to confirm.

### More Expression Examples You Can Ask For

| What You Ask | What Copilot Returns |
|---|---|
| "Format a number as currency with 2 decimal places" | `formatNumber(variables('Amount'), 'C2')` |
| "Get the first 100 characters of a string" | `substring(triggerBody()?['Body'], 0, 100)` |
| "Check if a string contains the word 'urgent'" | `contains(triggerBody()?['Subject'], 'urgent')` |
| "Convert a date string to a date object" | `parseDateTime(triggerBody()?['DateField'])` |
| "Calculate days between two dates" | `div(sub(ticks(variables('EndDate')), ticks(variables('StartDate'))), 864000000000)` |

---

## Using Copilot to Troubleshoot Errors

When a flow run fails, the error message often contains technical details that are hard to parse. Copilot can interpret them.

### Step 1: Find the Error

> **On screen:** In the flow run history (accessible from the flow detail page by clicking on a run), failed steps show a red X. Click the failed step to expand it. The "Outputs" section shows the raw error response, which looks something like:
> `{"error": {"code": "InvalidTemplate", "message": "Unable to process template language expressions..."}}`

### Step 2: Paste the Error into Copilot

Open the Copilot panel and type:

```
I got this error: {"error": {"code": "InvalidTemplate", "message": "Unable to process
template language expressions in action 'Create_file' inputs at line '0' and column '0':
'The template language expression 'triggerBody()?['Attachments'][0]['ContentBytes']'
cannot be evaluated because array index '0' is out of range."}}

What caused this and how do I fix it?
```

Copilot explains:
- The flow tried to access `Attachments[0]` but the email had no attachments
- The trigger fired even when there were no attachments because no filter was configured
- The fix is to add a Condition before the action that checks `length(triggerBody()?['Attachments'])` is greater than 0

### Step 3: Apply the Fix

Ask Copilot to apply the fix directly:

```
Add a condition at the start of the flow that checks whether the email has at least
one attachment, and only proceed if it does
```

---

## Prompt Engineering Tips for Better Flow Generation

The quality of Copilot's output depends heavily on how clearly you describe the automation. These patterns consistently produce better results:

### Be Specific About the Trigger

Vague:
```
When something happens in SharePoint, notify me
```

Specific:
```
When a new item is added to the SharePoint list called "Project Requests" in the
"Operations" site, send me an email with the item title and the requestor's name
```

### Name Your Systems Explicitly

Include the specific Microsoft 365 service or connector. "Save a file" is ambiguous — "save a file to OneDrive" or "save a file to SharePoint" gives Copilot the connector to use.

### Describe Conditions Explicitly

```
If the approval response is "Approve," update the SharePoint item status to "Approved"
and send a congratulations email. If it is "Reject," update the status to "Rejected"
and send a rejection email with the reason from the approval comments.
```

### Break Complex Flows Into Parts

For flows with many branches or steps, prompt Copilot to build it in sections rather than all at once. Build the main happy path first, then ask Copilot to add error handling, then add notifications.

### Specify What NOT to Do

```
When a new file is added to the "Reports" folder in SharePoint, convert it to PDF
and email it to the team. Do not process files in subfolders.
```

---

## Limitations of Copilot in Power Automate

Copilot is a starting point, not a finished product. Know these limits:

| Limitation | What This Means |
|---|---|
| Cannot access your data | Copilot cannot see your SharePoint lists, email contacts, or folder names — you must fill these in |
| Complex logic may be approximated | Nested conditions, loops, and error handling branches often need manual adjustment |
| Expression syntax may need correction | Copilot-generated expressions sometimes reference wrong field names — always test them |
| Does not configure all action settings | Fields like pagination, retry policies, and advanced options require manual configuration |
| Limited connector coverage | Less common connectors may not be well represented — Copilot may suggest alternatives |
| Iterative prompt follow-up required | Large flows built in one prompt are often imprecise — build iteratively |

**When to use manual editing instead of Copilot:**
- Configuring complex Apply to Each loops with nested conditions
- Setting up custom error handling with Scope and Configure Run After
- Working with uncommon connectors or custom connectors
- Fine-tuning pagination and performance settings
- Building flows with more than 15-20 steps

---

## Summary

Copilot in Power Automate accelerates flow creation by translating natural language descriptions into structured flows. The most effective workflow is:

1. Write a specific, trigger-first description with named systems
2. Let Copilot generate the structure
3. Fill in placeholders and verify dynamic content mappings
4. Test with real data
5. Use the Copilot panel to iterate — add steps, adjust logic, generate expressions
6. Switch to manual editing for complex logic that Copilot approximates

The next guide covers AI Builder actions, which take Copilot-assisted flows further by embedding pre-built AI models directly into your automation logic.


## Practice Questions

**Question 1 — Conceptual:** Based on the concepts in this guide, explain in your own words why the core technique matters and when you would choose it over alternatives.

**Question 2 — Application:** Sketch out how you would apply the main concept from this guide to a real-world dataset or problem you have encountered. What would you need to watch out for?


---

## Cross-References

<a class="link-card" href="./01_copilot_flow_builder_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Interactive slide deck covering the key concepts with visual examples.</div>
</a>

<a class="link-card" href="../notebooks/01_ai_builder_rest_api.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises and real data.</div>
</a>
