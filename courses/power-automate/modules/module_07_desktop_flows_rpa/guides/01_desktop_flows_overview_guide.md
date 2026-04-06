# Desktop Flows and RPA Fundamentals

> **Reading time:** ~18 min | **Module:** 7 — Desktop Flows & RPA | **Prerequisites:** Module 1

## In Brief

Desktop flows are the Robotic Process Automation (RPA) engine inside Power Automate. They run on a Windows machine — not in the cloud — and automate any application that has a screen: legacy ERP systems, thick-client desktop apps, terminal emulators, web browsers, and even applications with no API whatsoever.

<div class="callout-key">
<strong>Key Concept:</strong> Desktop flows are the Robotic Process Automation (RPA) engine inside Power Automate. They run on a Windows machine — not in the cloud — and automate any application that has a screen: legacy ERP systems, thick-client desktop apps, terminal emulators, web browsers, and even applications with no API whatsoever.
</div>


## Learning Objectives

By the end of this guide you will be able to:

<div class="callout-insight">
<strong>Insight:</strong> By the end of this guide you will be able to:

1.
</div>


1. Explain what desktop flows are and when to choose them over cloud flows
2. Install and configure Power Automate Desktop on a Windows machine
3. Navigate the Desktop flow designer and identify its panels
4. Categorize desktop flow actions by their action group
5. Record and edit desktop actions using the web and desktop recorders
6. Declare variables and set correct data types in a desktop flow
7. Define input and output variables to pass data between cloud and desktop flows

---

## Cloud Flows vs Desktop Flows: The Core Distinction

Before installing anything, understand the fundamental split.

<div class="callout-key">
<strong>Key Point:</strong> Before installing anything, understand the fundamental split.
</div>


| Dimension | Cloud Flows | Desktop Flows |
|---|---|---|
| **Runs on** | Microsoft cloud servers | Your Windows machine (or a VM) |
| **Requires** | Internet + connector to target service | Power Automate Desktop app installed locally |
| **Automates** | Anything with an API or Microsoft connector | Any visible UI on screen — with or without an API |
| **Trigger** | Event, schedule, or HTTP | Called by a cloud flow (or run manually) |
| **Licensing** | Included with many Microsoft 365 plans | Requires Power Automate per-user plan (attended) or unattended RPA license |
| **Primary use case** | Cloud-to-cloud integration | Legacy systems, thick-client apps, data extraction from UI |

**Decision rule:** If the target application exposes an API or Microsoft connector, use a cloud flow. If automation must interact with pixels on a screen — clicking buttons, reading text from windows, entering data into fields — use a desktop flow.

### When Desktop Flows Are the Right Choice

Desktop flows solve problems that cloud flows cannot:

- **Legacy ERP or terminal apps** — SAP GUI, AS/400 green-screen, or homegrown VB6 applications that predate REST APIs
- **Local Windows applications** — desktop Excel with complex macros, locally-installed PDFs, CAD tools
- **Web applications with no connector** — internal portals, third-party dashboards locked behind authentication that prevents API access
- **File system operations** — bulk renaming, moving, reading files from network shares without SharePoint
- **Multi-application data transfer** — extracting from one app and entering into another, known as "swivel-chair" work

---


<div class="compare">
<div class="compare-card">
<div class="header before">Cloud Flows</div>
<div class="body">

See detailed comparison in the table above.

</div>
</div>
<div class="compare-card">
<div class="header after">Desktop Flows: The Core Distinction</div>
<div class="body">

See detailed comparison in the table above.

</div>
</div>
</div>

## Power Automate Desktop: Installation and Setup

Power Automate Desktop is the Windows application that runs, records, and edits desktop flows. It is free to download; the license controls which execution modes are available.

<div class="callout-info">
<strong>Info:</strong> Power Automate Desktop is the Windows application that runs, records, and edits desktop flows.
</div>


### System Requirements

| Requirement | Minimum |
|---|---|
| Operating system | Windows 10 (version 1903 or later) or Windows 11 |
| Architecture | 64-bit |
| RAM | 2 GB (4 GB recommended) |
| Browser support | Microsoft Edge or Google Chrome for web recording |
| .NET framework | .NET Framework 4.7.2 or later |

> **On screen:** Go to `make.powerautomate.com` → left navigation → **My flows** → **Desktop flows** → **Install Power Automate**. The installer button downloads `Setup.Microsoft.PowerAutomateDesktop.exe`. Run it and accept defaults. The install takes approximately three minutes.

### First Launch

After installation, Power Automate Desktop opens to the **My flows** home screen. Sign in with your Microsoft work or school account — the same account you use at `make.powerautomate.com`.

> **On screen:** The home screen shows a list of desktop flows you own, a **New flow** button in the top-left, and a search bar. The status bar at the bottom confirms which account and environment you are connected to. If you see a different tenant name than expected, click the account avatar and switch environments before creating any flows.

### Registering Your Machine

Your machine must be registered in Power Automate before cloud flows can call desktop flows on it.

> **On screen:** In the Power Automate Desktop app, go to **Settings** (gear icon) → **Machine settings**. Your machine name appears here. To verify registration, open `make.powerautomate.com` → **Monitor** → **Machines**. Your machine should appear in the list with a green **Online** status dot when Power Automate Desktop is running.

---

## The Desktop Flow Designer

The designer is where you build, edit, and test desktop flows. Every desktop flow opens inside this environment.

```
┌───────────────────────────────────────────────────────────────────────┐
│  TOOLBAR  [Run] [Stop] [Step] [Record] [Save] [Flow name...]         │
├────────────────┬──────────────────────────────┬───────────────────────┤
│                │                              │                       │
│  ACTION        │         WORKSPACE            │   VARIABLES           │
│  PANEL         │   (flow steps listed         │   PANEL               │
│                │    top to bottom)            │                       │
│  Search...     │                              │  Input/Output:        │
│                │  1. [Action icon] Step name  │  ├── CustomerID       │
│  ▸ Browser     │  2. [Action icon] Step name  │  └── InvoiceTotal     │
│  ▸ Excel       │  3. [Action icon] Step name  │                       │
│  ▸ File        │  ...                         │  Flow:                │
│  ▸ Folders     │                              │  ├── RowCount         │
│  ▸ System      │  [+ Add action here]         │  └── CurrentRow       │
│  ▸ UI          │                              │                       │
│  ▸ Web         │                              │  Errors:              │
│  ▸ Text        │                              │  (none)               │
│  ▸ Variables   │                              │                       │
│  ...           │                              │                       │
└────────────────┴──────────────────────────────┴───────────────────────┘
```

| Panel | Purpose |
|---|---|
| **Action panel** (left) | Searchable catalog of all available actions grouped by category |
| **Workspace** (center) | The sequential list of steps in your flow — drag to reorder, double-click to edit |
| **Variables panel** (right) | Declares input/output variables and shows flow-scoped variables; inspect values during testing |

### Toolbar Controls

| Control | Shortcut | Effect |
|---|---|---|
| **Run** | F5 | Execute the entire flow from the beginning |
| **Stop** | Shift+F5 | Halt a running flow immediately |
| **Step** | F10 | Execute one action and pause — useful for debugging |
| **Record** | — | Launch the recorder to capture UI interactions |
| **Save** | Ctrl+S | Save changes (auto-save does not exist — save frequently) |

---

## Action Groups

Desktop flow actions are organized into groups. Understanding the groups lets you navigate the catalog quickly and reason about what the flow can do.

### UI Automation

Interacts with elements in Windows desktop applications (not the browser — that is the Web group).

| Action | What It Does |
|---|---|
| Click UI element | Simulates a left, right, or double mouse click |
| Fill text field | Types text into an input field |
| Get UI element attribute | Reads a property (text content, visibility, state) from a UI control |
| Set window state | Minimizes, maximizes, or restores a window |
| Wait for UI element | Pauses flow until an element appears or disappears |
| Select option in drop-down | Selects a value from a combo box or list |

### Excel

Reads from and writes to Excel workbooks without requiring Excel automation libraries.

| Action | What It Does |
|---|---|
| Launch Excel | Opens a workbook or starts a new blank workbook |
| Read from Excel worksheet | Reads a cell, range, named range, or entire sheet into a variable |
| Write to Excel worksheet | Writes a value, list, or data table into a cell or range |
| Get first free row on column | Finds the next empty row for appending data |
| Save Excel | Saves the workbook (equivalent to Ctrl+S) |
| Close Excel | Closes the workbook with optional save |

> **On screen:** In the Action panel, expand **Excel**. Notice that Excel actions work directly against the Excel COM object — they do not require the Excel window to be visible on screen, unlike UI Automation actions. This makes Excel actions faster and more reliable than screen-scraping an Excel window.

### File System

Manages files and folders on local drives and network shares.

| Action | What It Does |
|---|---|
| Get files in folder | Returns a list of file objects matching a path/filter |
| Copy file | Copies a file to a destination |
| Move file | Moves and optionally renames a file |
| Delete file | Sends a file to the recycle bin or permanently deletes |
| Read text from file | Reads full text content into a variable |
| Write text to file | Writes or appends text to a file |

### Web

Controls web browsers (Edge or Chrome) with element-level precision — more reliable than UI Automation for web content.

| Action | What It Does |
|---|---|
| Launch new Edge / Chrome | Opens a browser and navigates to a URL |
| Navigate to web page | Changes the current page in an existing browser instance |
| Click link on web page | Clicks a hyperlink identified by its web element selector |
| Fill text field on web page | Types into an input field identified by CSS or XPath selector |
| Get details of web page element | Reads text, href, or attribute from a web element |
| Wait for web page content | Pauses until an element is present or page text matches |

### Email

Sends and retrieves email using desktop mail clients or direct SMTP/IMAP connections.

| Action | What It Does |
|---|---|
| Send email | Sends via Outlook or SMTP with optional attachments |
| Retrieve email messages | Fetches messages from Outlook or IMAP mailbox |
| Process email messages | Moves or deletes retrieved messages |

> **On screen:** When you expand **Email** in the Action panel, you see two subgroups: **Outlook** (uses the locally-installed Outlook desktop app) and **Email** (uses direct SMTP/IMAP connections without Outlook). Use the **Outlook** subgroup when the machine has Outlook installed; use **Email** for headless or unattended machines.

### System

Performs operating-system-level operations.

| Action | What It Does |
|---|---|
| Run application | Launches an executable or opens a file |
| Get environment variable | Reads a Windows environment variable |
| Set environment variable | Creates or updates an environment variable |
| Get running processes | Returns a list of running processes |
| Wait for process | Pauses until a process starts or stops |
| Log off user | Signs out of the current Windows session |

---

## Recording Desktop Actions

The recorder captures your UI interactions and converts them into a sequence of actions in the workspace. Use recording to bootstrap a flow quickly; expect to edit the result.

### Web Recorder

The web recorder works inside Microsoft Edge or Google Chrome. It captures actions against web elements using CSS selectors and XPath — more robust than pixel coordinates.

**How to start a web recording:**

> **On screen:** In the designer toolbar, click **Record**. A floating **Recorder** toolbar appears. Click **Record** on the toolbar (the red circle button). Switch to your browser. Every click, text entry, and navigation you perform is captured and shown in the recorder's live action list. When finished, click **Done** on the recorder toolbar. The captured actions appear at the current position in the workspace.

**What gets captured:**

| Your Action | Generated Action |
|---|---|
| Clicking a link | Click link on web page (element: anchor[href="..."] |
| Typing in a field | Fill text field on web page (element: input#email) |
| Selecting from dropdown | Select option in drop-down on web page |
| Navigating to a URL | Navigate to web page (URL: https://...) |

### Desktop Recorder

The desktop recorder captures interactions with Windows applications (not browser content). It uses UI element identifiers — window class, control ID, and automation ID — rather than pixel positions.

**How to start a desktop recording:**

> **On screen:** With the target desktop application already open, click **Record** in the designer toolbar. Click **Record** on the floating toolbar. Interact with the target application normally — click buttons, fill fields, select menu items. The recorder lists each captured action in real time. Click **Done** when finished.

**Editing Recorded Actions**

Recordings often require cleanup:

- **Hardcoded values** — Replace literal values (e.g., `"John Smith"`) with variable references (`%CustomerName%`)
- **Selector fragility** — Generated selectors sometimes use volatile attributes (window title, row index). Edit selectors to use stable identifiers (AutomationId, Name)
- **Timing issues** — Add **Wait for UI element** or **Wait for web page content** actions before actions that depend on a page or window loading
- **Missing error handling** — Wrap risky actions (file open, application launch) in a block with **On block error** handling

> **On screen:** To edit a recorded action, double-click it in the workspace. The action's configuration dialog opens. For UI Automation and Web actions, you will see a **selector** field showing the element's identity. Click the selector to open the **UI element picker** where you can refine which attributes are used to identify the element.

---

## Variables and Data Types

Desktop flows use variables to store values between steps. Every variable has a name and a data type determined at assignment.

### Declaring Variables

Variables are created automatically when an action produces output. You can also create variables explicitly using the **Set variable** action.

> **On screen:** In the Variables panel (right side), click **+** to add a new variable. Name it using the `%VariableName%` format — Power Automate Desktop uses `%` delimiters (not `$` or `{}`). Choose a type from the dropdown.

### Data Types

| Type | Examples | Common Source |
|---|---|---|
| **Text** | `"Invoice #1042"`, `"John"` | Filled text fields, file contents, web page text |
| **Number** | `42`, `3.14` | Excel cells, arithmetic results |
| **Boolean** | `True`, `False` | Condition results, UI element visibility checks |
| **Datetime** | `3/8/2026 09:00:00` | File modified dates, email timestamps |
| **List** | `["Alice", "Bob", "Carol"]` | Files in folder, rows from Excel, email messages |
| **Data table** | Rows × columns structure | Read from Excel worksheet, query results |
| **Custom object** | JSON-like named properties | Parsed JSON, web service responses |

### Variable Naming Conventions

Desktop flow variables are referenced as `%VariableName%` inside action fields. Names must:
- Start with a letter
- Contain only letters, numbers, and underscores
- Be unique within the flow

Good names: `%CustomerID%`, `%InvoiceTotal%`, `%FileList%`

Avoid: `%x%`, `%temp%`, `%var1%`

### Working with Lists and Data Tables

Many actions return lists or data tables. Iterate over them with a **For each** loop.

```
For each CurrentItem in %FileList%
    Open Excel: %CurrentItem%
    Read from Excel worksheet → %CellValue%
    Append %CellValue% to %ResultList%
    Close Excel
End
```

> **On screen:** In the Action panel, search for "For each". Drag it into the workspace. In the configuration dialog, set the **Iterate over** field to the list variable and the **Store to** field to the loop-variable name (e.g., `%CurrentItem%`). Actions placed between **For each** and **End** run once for each item in the list.

---

## Input and Output Variables: Cloud ↔ Desktop Integration

Input and output variables are the data bridge between cloud flows and desktop flows.

### Input Variables

Input variables receive values from the calling cloud flow before the desktop flow begins execution.

> **On screen:** In the Variables panel, click the **+** next to **Input/Output** (at the top of the panel, above the **Flow variables** section). Click **Input**. Fill in:
> - **Variable name** — e.g., `CustomerID`
> - **Data type** — choose Text, Number, Boolean, or Datetime
> - **Default value** — used when running the flow manually from the designer (not used in production)
> - **Description** — shown to the cloud flow builder; make it clear and specific

The desktop flow can now reference `%CustomerID%` in any action field.

### Output Variables

Output variables pass results back to the calling cloud flow after the desktop flow completes.

> **On screen:** In the Variables panel, click **+** → **Output**. Enter:
> - **Variable name** — e.g., `InvoiceTotal`
> - **Data type** — the type of the value this variable will hold when the flow finishes
> - **Description** — describe what the value represents

Within the flow, assign the result to `%InvoiceTotal%` using a **Set variable** action or by configuring an action's output field.

### Variable Flow Diagram

```
Cloud Flow
    │
    │  Passes: CustomerID = "C-1042"
    │          InvoiceDate = "2026-03-08"
    ▼
Desktop Flow [runs on Windows machine]
    │
    │  Input variables received:
    │  %CustomerID% = "C-1042"
    │  %InvoiceDate% = "2026-03-08"
    │
    │  [... automation steps ...]
    │
    │  Output variable set:
    │  %InvoiceTotal% = 4850.00
    │
    ▼
Cloud Flow (resumed)
    │
    │  Receives: InvoiceTotal = 4850.00
    │  → proceeds with the rest of the cloud flow
```

### Supported Input/Output Data Types

| Desktop Flow Type | Cloud Flow Type |
|---|---|
| Text | String |
| Number | Float |
| Boolean | Boolean |
| Datetime | String (ISO 8601 format) |
| List of Text | Array of String |

> **On screen:** After saving a desktop flow that has input/output variables, navigate to `make.powerautomate.com`, open a cloud flow, and add the action **Run a flow built with Power Automate Desktop**. The action card will show fields for each input variable you defined and an output section listing each output variable. You connect dynamic content from earlier cloud flow steps into the input fields — no mapping configuration required.

---

## Common Pitfalls

<div class="callout-danger">
<strong>Danger:</strong> The pitfalls below are the most common mistakes practitioners make. Each one can silently degrade your results without obvious errors.
</div>

- **Forgetting to save** — The designer has no auto-save. Use Ctrl+S frequently. An unsaved flow loses changes when the machine restarts.
- **Fragile selectors** — Recording generates selectors based on window title and index position. If the window title changes (e.g., includes the current filename) or a list position shifts, the selector breaks. Always review and stabilize selectors after recording.
- **Hardcoded delays** — Using **Wait** with a fixed number of seconds is brittle. Use **Wait for UI element** or **Wait for web page content** instead, which waits until the condition is met rather than sleeping unconditionally.
- **Data type mismatches** — Passing a Text variable where a Number is expected causes a runtime error. Use **Convert text to number** or **Convert datetime to text** actions to transform types explicitly.
- **Machine offline** — Desktop flows cannot run if Power Automate Desktop is not open and the machine is not online. For unattended execution, ensure the service account keeps the desktop app running.

<div class="callout-warning">
<strong>Warning:</strong> - **Forgetting to save** — The designer has no auto-save.
</div>

---

## Connections to Other Modules

- **Builds on:** Module 00 (flow types overview), Module 01 (cloud flow fundamentals), Module 04 (error handling patterns)
- **Leads to:** Guide 02 in this module (triggering desktop flows from cloud flows, attended vs unattended, machine groups)
- **Related to:** Module 08 (Copilot inside Power Automate Desktop — AI-assisted action generation)

---

## Further Reading

- [Power Automate Desktop documentation](https://learn.microsoft.com/en-us/power-automate/desktop-flows/introduction) — official reference for all actions, selectors, and configuration options
- [Power Automate Desktop action reference](https://learn.microsoft.com/en-us/power-automate/desktop-flows/actions-reference) — complete catalog of every action group and action
- [UI element selectors in desktop flows](https://learn.microsoft.com/en-us/power-automate/desktop-flows/ui-elements) — deep dive into selector syntax and best practices


---

## Cross-References

<a class="link-card" href="./01_desktop_flows_overview_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Interactive slide deck covering the key concepts with visual examples.</div>
</a>

<a class="link-card" href="../notebooks/01_rpa_patterns.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises and real data.</div>
</a>
