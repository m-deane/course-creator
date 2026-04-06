# AI Builder Actions in Power Automate

> **Reading time:** ~16 min | **Module:** 8 — Copilot & Power Automate | **Prerequisites:** Module 4

## In Brief

AI Builder is Microsoft's no-code AI capability embedded in the Power Platform. It provides pre-built and custom AI models that you can drop into Power Automate flows as actions — no data science background required. This guide covers the available AI Builder action types, how to use them in flows, and how Copilot can help you wire them up.

<div class="callout-key">

<strong>Key Concept:</strong> AI Builder is Microsoft's no-code AI capability embedded in the Power Platform. It provides pre-built and custom AI models that you can drop into Power Automate flows as actions — no data science background required.

</div>


---

## What AI Builder Is

AI Builder provides two categories of models:

<div class="callout-insight">

<strong>Insight:</strong> AI Builder provides two categories of models:

**Pre-built models** — Microsoft-trained models ready to use immediately:
- Sentiment analysis
- Entity extraction
- Category classification
- Key phrase...

</div>


**Pre-built models** — Microsoft-trained models ready to use immediately:
- Sentiment analysis
- Entity extraction
- Category classification
- Key phrase extraction
- Language detection
- Receipt processing
- Invoice processing
- Business card reader
- Text recognition (OCR)
- Object detection in images

**Custom models** — Models you train on your own data:
- Custom document processing (forms, contracts)
- Custom object detection
- Custom category classification
- Custom entity extraction

For most Power Automate use cases, pre-built models are the starting point. Custom models are appropriate when pre-built models do not match your specific document layouts or domain vocabulary.

---

## AI Builder Actions in Power Automate

AI Builder actions appear in the Power Automate action picker under the "AI Builder" connector. Each model type has one or more corresponding actions.

<div class="callout-key">

<strong>Key Point:</strong> AI Builder actions appear in the Power Automate action picker under the "AI Builder" connector.

</div>


### Text Generation (GPT-powered)

The **Create text with GPT using a prompt** action sends a text prompt to a GPT model and returns generated text.

Use cases:
- Summarize long email threads
- Draft a reply based on email content
- Classify text by describing the categories in the prompt
- Extract structured data from unstructured text
- Generate subject lines or document titles

> **On screen:** When you add the "Create text with GPT using a prompt" action, you see a prompt field with a text editor. You write your instructions in this field, mixing static text with dynamic content from earlier steps. A token/character counter shows how much of the context window you are using.

**Example prompt for email summarization:**
```
Summarize the following email in 2-3 sentences, focusing on any action items or
deadlines mentioned:

[Body from email trigger]
```

**Licensing note:** GPT-powered actions consume AI Builder credits. Each call costs credits based on the number of tokens processed.

---

### Document Processing: Invoices

The **Extract information from invoices** action parses invoice documents (PDF or image) and returns structured fields.

Fields returned:
- Vendor name, address, phone
- Invoice number, date, due date
- Line items with descriptions and amounts
- Subtotal, tax, total amount

> **On screen:** After adding the action, the "Invoice file content" field expects a base64-encoded file or a file object from a trigger or previous step. If your trigger provides an attachment, use the "Attachment Content" dynamic content from the trigger — it is already in the correct format.

**Document format requirements:**
- Supported: PDF, JPEG, PNG, BMP, TIFF
- Maximum file size: 20 MB
- Best results: clear, machine-generated PDFs (not handwritten or heavily skewed scans)

---

### Document Processing: Receipts

The **Extract information from receipts** action parses receipt images or PDFs and returns:
- Merchant name and address
- Transaction date and time
- Individual line items with prices
- Total, tax, and tip amounts
- Payment method

This action is pre-built for common receipt formats (restaurants, retail, travel). Custom document processing is available if your receipts have unusual layouts.

---

### Sentiment Analysis

The **Analyze positive or negative sentiment in text** action classifies text as Positive, Negative, Neutral, or Mixed and returns a confidence score for each.

> **On screen:** The action has a single input field: "Language" (optional, auto-detected if left blank) and "Text" (required). Connect the text source — email body, form response, survey answer — to this field using dynamic content.

Output fields:
- `Predicted sentiment` — the dominant sentiment (Positive/Negative/Neutral/Mixed)
- `Positive score` — confidence score 0.00 to 1.00
- `Negative score` — confidence score 0.00 to 1.00
- `Neutral score` — confidence score 0.00 to 1.00

**Example use case:** Route support tickets — high negative sentiment score triggers an escalation path; positive sentiment goes to standard queue.

---

### Entity Extraction

The **Extract entities from text** action identifies and labels named entities in unstructured text.

Entity types detected:
- Person (names)
- Organization (company names)
- Location (cities, countries, addresses)
- DateTime (dates, times, durations)
- Quantity (numbers, percentages, measurements)
- Phone number
- Email address
- URL

> **On screen:** The action has "Text" and "Language" inputs. Output is a table (array) of entities, each with a "Type" and "Value" field. You use an Apply to Each loop to process the array and act on individual entities.

**Example use case:** Extract all company names and dates from incoming contract documents, then log them to a SharePoint list for contract tracking.

---

### Category Classification

The **Classify text into categories** action assigns a text input to one of several categories you define.

> **On screen:** The action configuration panel has two sections: the "Text" input field and a "Categories" table where you add the category names you want to classify into. You add rows to the table: each row is a category name. The model returns which category best fits the input text.

**Example categories for IT support tickets:**
- Hardware issue
- Software issue
- Access request
- Network problem
- Other

The model returns the predicted category and a confidence score. Use the predicted category in a condition to route the ticket to the right queue in Teams or send to the right email group.

---

### Object Detection

The **Detect objects in images** action finds and locates objects in image files. This requires a custom-trained model (you train it on your own labeled images) rather than a pre-built general model.

Common use cases:
- Detecting product defects in manufacturing photos
- Identifying items in warehouse photos
- Counting objects in field inspection images

For most Power Automate beginner and intermediate workflows, text-based AI Builder actions (sentiment, entity, GPT) are used more frequently than object detection.

---

## Step-by-Step: Building an Email Processing Flow with AI Builder

This flow processes incoming emails from vendors: it extracts invoice data from attachments, analyzes the email sentiment, and drafts a reply using GPT.

<div class="callout-info">

<strong>Info:</strong> This flow processes incoming emails from vendors: it extracts invoice data from attachments, analyzes the email sentiment, and drafts a reply using GPT.

</div>


### Prerequisites

- AI Builder is licensed in your environment (AI Builder credits available)
- You have an Outlook mailbox
- You have a SharePoint list named "Invoice Log" with columns: Vendor Name, Invoice Number, Invoice Date, Total Amount, Sentiment, Status

### Step 1: Create the Flow

Start from the Copilot prompt on the Home page:


<span class="filename">example.py</span>
</div>
The following implementation builds on the approach above:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```text
When I receive an email with an attachment from a sender whose address contains
"vendor", extract invoice data from the attachment, analyze the email body
sentiment, and create a SharePoint list item with the results
```

</div>

> **On screen:** Copilot generates a flow with: Outlook trigger, a placeholder for invoice processing, a placeholder for sentiment analysis, and a SharePoint create item action. It will likely not include all three AI Builder steps correctly — use this as a starting point and add the AI Builder actions manually.

### Step 2: Configure the Trigger

Expand the Outlook "When a new email arrives (V3)" trigger.

> **On screen:** The trigger configuration panel shows filters. Set "Has Attachments" to "Yes" to ensure the trigger only fires when there is an attachment. In the "From" filter field, enter the vendor email domain if you want to limit to specific senders.

### Step 3: Add Invoice Processing

Click **+ New step** after the trigger. Search for "AI Builder" in the connector search. Select **Extract information from invoices**.

> **On screen:** The action appears with a single "Invoice file content" field. Click the field and in the dynamic content picker, look for "Attachment Content" from the trigger. If the email has multiple attachments, you need an Apply to Each loop around this action — use the trigger's "Attachments" array as the loop input.

For a single-attachment flow (simpler starting point):
- Set "Invoice file content" to `triggerBody()?['Attachments'][0]['ContentBytes']`

For multi-attachment flows:
- Add **Apply to Each** using `triggerBody()?['Attachments']` as the input
- Place the invoice extraction action inside the loop
- Reference `items('Apply_to_each')?['ContentBytes']` as the file content

### Step 4: Add Sentiment Analysis

Add a new step after the invoice extraction. Search for "AI Builder" and select **Analyze positive or negative sentiment in text**.

> **On screen:** Set the "Text" field to the email "Body" from the trigger's dynamic content. Leave "Language" blank for auto-detection. The action outputs "Predicted sentiment" and individual score fields that are available in subsequent steps.

### Step 5: Add GPT Text Generation for Reply Draft

Add another AI Builder action: **Create text with GPT using a prompt**.

In the prompt field, write:


<span class="filename">example.py</span>
</div>
The following implementation builds on the approach above:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```text
You are a professional accounts payable assistant. Write a brief, professional
acknowledgment reply for this vendor email.

Vendor: [Vendor Name from invoice extraction output]
Invoice Number: [Invoice Number from invoice extraction output]
Email Sentiment: [Predicted sentiment from sentiment analysis output]

If sentiment is Negative, add a note acknowledging any concerns. Keep the reply
under 100 words. Do not include a subject line.
```

</div>

> **On screen:** When you click the "Vendor Name" field reference in your prompt, use the expression editor to reference the invoice extraction output. The output field name is typically `vendorName` — find it in dynamic content under the invoice extraction action.

### Step 6: Create the SharePoint List Item

Add the SharePoint **Create item** action. Map fields from previous steps:

| SharePoint Column | Dynamic Content Source |
|---|---|
| Vendor Name | Invoice extraction: Vendor Name |
| Invoice Number | Invoice extraction: Invoice Number |
| Invoice Date | Invoice extraction: Invoice Date |
| Total Amount | Invoice extraction: Invoice Total |
| Sentiment | Sentiment analysis: Predicted sentiment |
| Status | Static text: "Pending Review" |
| Reply Draft | GPT action: Generated text |

### Step 7: Send the Reply Draft for Human Review

Rather than auto-sending the GPT-generated reply, route it for human approval. Add an Outlook **Send an email (V2)** action to notify yourself with the draft:

- **To:** your email address
- **Subject:** `Reply Draft Ready: Invoice [Invoice Number from dynamic content]`
- **Body:** Include the GPT-generated reply text so you can copy, adjust, and send manually

### Step 8: Test the Flow

> **On screen:** Click **Test** and select **Manually**. Send a test email to yourself from a different account with a PDF invoice attached. After the trigger fires, check the run history. Each step shows its input and output — verify that the invoice fields were extracted correctly, the sentiment value looks right, and the GPT reply is coherent.

---

## Copilot-Assisted Editing of Existing Flows

After building a flow manually or with Copilot, you can use the Copilot panel to modify it by describing changes.

**Example modifications you can ask for:**


<span class="filename">example.py</span>
</div>
The following implementation builds on the approach above:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```text
Add error handling so that if the invoice extraction fails, send me an email
with the attachment name and the error message instead of failing the flow
```

</div>


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```text
Change the SharePoint list to also record the email subject line and the
date the email was received
```

</div>

You can also add conditional logic using natural language:

```text
Add a condition: if the invoice total is greater than 10000, also post a
message to the "High Value Invoices" Teams channel
```

After each Copilot edit, verify the change in the designer. Complex modifications (especially those involving error handling with Scope actions) often require manual cleanup.

---

## Best Practices

### Prompt Clarity for GPT Actions

The GPT text generation action produces better results when your prompt:
- States the role ("You are a professional...")
- Specifies the output format and length ("under 100 words," "in bullet points")
- Provides clear context from dynamic content fields
- Tells the model what NOT to do ("Do not include a subject line")

Test prompts by running the flow with representative data. Review the GPT output in the run history before making the flow live.

### Iterating with Copilot

When adding AI Builder steps to a flow, use the Copilot panel to explain what each step returns:

```text
What fields does the "Extract information from invoices" action return?
```

Copilot lists the output schema, helping you identify the correct dynamic content field names for downstream steps.

### Validating Generated Flows

For any flow that writes data (SharePoint, Outlook, Teams), test with non-production data first. Create a test SharePoint list or a test Teams channel. Verify outputs match expectations before pointing the flow at production systems.

---

## AI Builder Licensing and Credits

AI Builder uses a credit system separate from Power Automate licensing.

| Model Type | Credit Consumption |
|---|---|
| Sentiment analysis | 1 credit per 500 characters |
| Entity extraction | 1 credit per 500 characters |
| Invoice processing | 1 credit per document page |
| Receipt processing | 1 credit per document page |
| GPT text generation | Credits based on token count |
| Custom model prediction | 1 credit per prediction |

Credits are allocated at the environment level. Power Apps and Power Automate premium plans include a monthly credit allocation. Environments approaching their credit limit show warnings in the Power Platform admin center.

To check remaining credits:
> **On screen:** Navigate to `admin.powerplatform.microsoft.com`. Select your environment. Under **Resources**, click **AI Builder** to see credit usage and remaining balance.

When credits are exhausted, AI Builder actions fail with a credit-related error. Request additional credits from your organization's Power Platform admin or purchase add-on credit packs through the Microsoft 365 admin center.

---

## Summary

AI Builder actions bring pre-built AI capabilities — invoice parsing, sentiment analysis, entity extraction, text classification, and GPT text generation — directly into Power Automate flows. The typical pattern is:

1. A trigger provides raw data (email, file, form response)
2. AI Builder actions process the data (extract, classify, generate)
3. Conditions route based on AI output (sentiment score, category, extracted fields)
4. Standard actions write results (SharePoint, Teams, Outlook)

Use Copilot to generate the initial flow structure and to modify it conversationally. Use manual editing for the AI Builder action configuration details — field mappings and output references require careful attention to the dynamic content schema.

The next module covers Copilot agents: building autonomous AI workflows that go beyond reactive flows and make decisions across multiple steps.


---

## Cross-References

<a class="link-card" href="./02_ai_builder_actions_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Interactive slide deck covering the key concepts with visual examples.</div>
</a>

<a class="link-card" href="../notebooks/01_ai_builder_rest_api.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises and real data.</div>
</a>
