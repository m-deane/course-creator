# Adaptive Card Recipes

Complete Adaptive Card JSON recipes for the most common Power Automate use cases.
Each recipe is ready to paste into the **Adaptive Card** field of a Teams action.
Replace static placeholder values with Power Automate dynamic expressions as needed.

Validate and preview any card at https://adaptivecards.io/designer/

---

## Recipe 1: Simple Notification Card

**Use case:** Broadcast a plain-language update to a Teams channel — no response required.

**Power Automate action:** Post card in a chat or channel (no wait)

**Customisation points:**
- `<TITLE>` — main heading
- `<BODY>` — supporting paragraph
- `<LINK_TEXT>` and `<LINK_URL>` — optional call-to-action button (delete `actions` block if not needed)
- Change `"style": "accent"` on the container to `"good"` (green), `"warning"` (amber), or `"attention"` (red)

```json
{
  "type": "AdaptiveCard",
  "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
  "version": "1.4",
  "body": [
    {
      "type": "Container",
      "style": "accent",
      "bleed": true,
      "items": [
        {
          "type": "TextBlock",
          "text": "NOTIFICATION",
          "color": "light",
          "weight": "bolder",
          "size": "small"
        }
      ]
    },
    {
      "type": "Container",
      "spacing": "medium",
      "items": [
        {
          "type": "TextBlock",
          "text": "<TITLE>",
          "weight": "bolder",
          "size": "large",
          "wrap": true
        },
        {
          "type": "TextBlock",
          "text": "<BODY>",
          "wrap": true,
          "spacing": "small"
        },
        {
          "type": "TextBlock",
          "text": "Sent: {{DATE(2026-03-08T14:00:00Z,SHORT)}}",
          "isSubtle": true,
          "size": "small",
          "spacing": "medium"
        }
      ]
    }
  ],
  "actions": [
    {
      "type": "Action.OpenUrl",
      "title": "<LINK_TEXT>",
      "url": "<LINK_URL>"
    }
  ]
}
```

**Tip:** Replace the static date string with a Power Automate expression in the field:
```
{{DATE(@{utcNow()},SHORT)}}
```
Teams renders `{{DATE(...)}}` as a localised date automatically.

---

## Recipe 2: Approval Card with Custom Buttons

**Use case:** Send to an approver in Teams and wait for their decision. The submitted
value is returned to the flow in the `data` property of the response body.

**Power Automate action:** Post adaptive card and wait for a response

**Flow setup:**
1. Add the action and set Recipient to the approver's email.
2. After the action, access the response:
   - Decision: `body('Post_adaptive_card_and_wait_for_a_response')?['data']?['decision']`
   - Comments: `body('Post_adaptive_card_and_wait_for_a_response')?['data']?['approverComments']`

**Customisation points:**
- `<REQUEST_TITLE>` — title of the request being approved
- `<REQUESTER>` — name of the person who submitted
- `<AMOUNT>` — value or quantity (remove the fact row if irrelevant)
- `<DESCRIPTION>` — context paragraph
- `<VIEW_URL>` — deep link to the source item

```json
{
  "type": "AdaptiveCard",
  "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
  "version": "1.4",
  "body": [
    {
      "type": "Container",
      "style": "emphasis",
      "items": [
        {
          "type": "ColumnSet",
          "columns": [
            {
              "type": "Column",
              "width": "stretch",
              "items": [
                {
                  "type": "TextBlock",
                  "text": "Approval Required",
                  "weight": "bolder",
                  "size": "medium"
                },
                {
                  "type": "TextBlock",
                  "text": "<REQUEST_TITLE>",
                  "wrap": true,
                  "spacing": "none",
                  "isSubtle": true
                }
              ]
            }
          ]
        }
      ]
    },
    {
      "type": "Container",
      "spacing": "medium",
      "items": [
        {
          "type": "FactSet",
          "facts": [
            {
              "title": "Requested by",
              "value": "<REQUESTER>"
            },
            {
              "title": "Amount",
              "value": "<AMOUNT>"
            },
            {
              "title": "Description",
              "value": "<DESCRIPTION>"
            }
          ]
        }
      ]
    },
    {
      "type": "Container",
      "spacing": "medium",
      "separator": true,
      "items": [
        {
          "type": "TextBlock",
          "text": "Comments",
          "weight": "bolder"
        },
        {
          "type": "Input.Text",
          "id": "approverComments",
          "placeholder": "Optional — enter any comments for the requestor",
          "isMultiline": true,
          "maxLength": 500
        }
      ]
    }
  ],
  "actions": [
    {
      "type": "Action.Submit",
      "title": "Approve",
      "style": "positive",
      "data": {
        "decision": "Approve"
      }
    },
    {
      "type": "Action.Submit",
      "title": "Reject",
      "style": "destructive",
      "data": {
        "decision": "Reject"
      }
    },
    {
      "type": "Action.OpenUrl",
      "title": "View Details",
      "url": "<VIEW_URL>"
    }
  ]
}
```

---

## Recipe 3: Data Entry Form with Validation

**Use case:** Collect structured information from a user before creating a SharePoint
item, sending an email, or triggering a downstream process.

**Power Automate action:** Post adaptive card and wait for a response

**Accessing form field values after submission:**
```
body('Post_adaptive_card_and_wait_for_a_response')?['data']?['formTitle']
body('Post_adaptive_card_and_wait_for_a_response')?['data']?['formCategory']
body('Post_adaptive_card_and_wait_for_a_response')?['data']?['formPriority']
body('Post_adaptive_card_and_wait_for_a_response')?['data']?['formDueDate']
body('Post_adaptive_card_and_wait_for_a_response')?['data']?['formDescription']
```

**Customisation points:**
- Update choice values in `Input.ChoiceSet` to match your taxonomy
- Add or remove `Input.*` blocks for your schema
- Set `"isRequired": true` on mandatory fields (Teams enforces this client-side)

```json
{
  "type": "AdaptiveCard",
  "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
  "version": "1.4",
  "body": [
    {
      "type": "TextBlock",
      "text": "Submit a Request",
      "weight": "bolder",
      "size": "large"
    },
    {
      "type": "TextBlock",
      "text": "Fill in all required fields and click Submit.",
      "wrap": true,
      "isSubtle": true,
      "spacing": "none"
    },
    {
      "type": "Container",
      "spacing": "medium",
      "items": [
        {
          "type": "TextBlock",
          "text": "Title *",
          "weight": "bolder"
        },
        {
          "type": "Input.Text",
          "id": "formTitle",
          "placeholder": "Enter a short descriptive title",
          "isRequired": true,
          "errorMessage": "Title is required"
        },
        {
          "type": "TextBlock",
          "text": "Category *",
          "weight": "bolder",
          "spacing": "medium"
        },
        {
          "type": "Input.ChoiceSet",
          "id": "formCategory",
          "style": "compact",
          "isRequired": true,
          "errorMessage": "Please select a category",
          "placeholder": "Select a category",
          "choices": [
            {"title": "IT", "value": "IT"},
            {"title": "Finance", "value": "Finance"},
            {"title": "HR", "value": "HR"},
            {"title": "Facilities", "value": "Facilities"},
            {"title": "Other", "value": "Other"}
          ]
        },
        {
          "type": "TextBlock",
          "text": "Priority",
          "weight": "bolder",
          "spacing": "medium"
        },
        {
          "type": "Input.ChoiceSet",
          "id": "formPriority",
          "style": "expanded",
          "value": "Medium",
          "choices": [
            {"title": "Low — within 2 weeks", "value": "Low"},
            {"title": "Medium — within 1 week", "value": "Medium"},
            {"title": "High — within 2 days", "value": "High"}
          ]
        },
        {
          "type": "TextBlock",
          "text": "Due Date",
          "weight": "bolder",
          "spacing": "medium"
        },
        {
          "type": "Input.Date",
          "id": "formDueDate",
          "placeholder": "Select a due date"
        },
        {
          "type": "TextBlock",
          "text": "Notify additional stakeholders",
          "weight": "bolder",
          "spacing": "medium"
        },
        {
          "type": "Input.Text",
          "id": "formCcEmails",
          "placeholder": "email1@contoso.com; email2@contoso.com",
          "isMultiline": false
        },
        {
          "type": "TextBlock",
          "text": "Description",
          "weight": "bolder",
          "spacing": "medium"
        },
        {
          "type": "Input.Text",
          "id": "formDescription",
          "placeholder": "Provide any additional context or attachments references...",
          "isMultiline": true,
          "maxLength": 2000
        }
      ]
    }
  ],
  "actions": [
    {
      "type": "Action.Submit",
      "title": "Submit Request",
      "style": "positive"
    }
  ]
}
```

---

## Recipe 4: Multi-Section Report Card

**Use case:** Post a daily or weekly summary report to a Teams channel. Organises data
into labelled sections with a key metrics row at the top.

**Power Automate action:** Post card in a chat or channel (no wait)

**Customisation points:**
- Replace all `<...>` values with dynamic expressions from your flow
- Adjust `"color"` on metric values: `"good"` (up/positive), `"attention"` (down/negative)
- Add or remove `Container` blocks for additional report sections

```json
{
  "type": "AdaptiveCard",
  "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
  "version": "1.4",
  "body": [
    {
      "type": "Container",
      "style": "accent",
      "bleed": true,
      "items": [
        {
          "type": "ColumnSet",
          "columns": [
            {
              "type": "Column",
              "width": "stretch",
              "items": [
                {
                  "type": "TextBlock",
                  "text": "Weekly Operations Report",
                  "weight": "bolder",
                  "color": "light",
                  "size": "medium"
                },
                {
                  "type": "TextBlock",
                  "text": "Week ending <REPORT_DATE>",
                  "color": "light",
                  "isSubtle": true,
                  "spacing": "none"
                }
              ]
            }
          ]
        }
      ]
    },
    {
      "type": "Container",
      "spacing": "medium",
      "items": [
        {
          "type": "TextBlock",
          "text": "Key Metrics",
          "weight": "bolder",
          "size": "medium"
        },
        {
          "type": "ColumnSet",
          "columns": [
            {
              "type": "Column",
              "width": "stretch",
              "items": [
                {
                  "type": "TextBlock",
                  "text": "Requests Submitted",
                  "isSubtle": true,
                  "size": "small"
                },
                {
                  "type": "TextBlock",
                  "text": "<TOTAL_SUBMITTED>",
                  "weight": "bolder",
                  "size": "extraLarge",
                  "spacing": "none"
                }
              ]
            },
            {
              "type": "Column",
              "width": "stretch",
              "items": [
                {
                  "type": "TextBlock",
                  "text": "Approved",
                  "isSubtle": true,
                  "size": "small"
                },
                {
                  "type": "TextBlock",
                  "text": "<TOTAL_APPROVED>",
                  "weight": "bolder",
                  "size": "extraLarge",
                  "color": "good",
                  "spacing": "none"
                }
              ]
            },
            {
              "type": "Column",
              "width": "stretch",
              "items": [
                {
                  "type": "TextBlock",
                  "text": "Rejected",
                  "isSubtle": true,
                  "size": "small"
                },
                {
                  "type": "TextBlock",
                  "text": "<TOTAL_REJECTED>",
                  "weight": "bolder",
                  "size": "extraLarge",
                  "color": "attention",
                  "spacing": "none"
                }
              ]
            },
            {
              "type": "Column",
              "width": "stretch",
              "items": [
                {
                  "type": "TextBlock",
                  "text": "Pending",
                  "isSubtle": true,
                  "size": "small"
                },
                {
                  "type": "TextBlock",
                  "text": "<TOTAL_PENDING>",
                  "weight": "bolder",
                  "size": "extraLarge",
                  "color": "warning",
                  "spacing": "none"
                }
              ]
            }
          ]
        }
      ]
    },
    {
      "type": "Container",
      "spacing": "medium",
      "separator": true,
      "items": [
        {
          "type": "TextBlock",
          "text": "Highlights",
          "weight": "bolder"
        },
        {
          "type": "TextBlock",
          "text": "<HIGHLIGHTS_TEXT>",
          "wrap": true
        }
      ]
    },
    {
      "type": "Container",
      "spacing": "medium",
      "separator": true,
      "items": [
        {
          "type": "TextBlock",
          "text": "Issues & Risks",
          "weight": "bolder"
        },
        {
          "type": "TextBlock",
          "text": "<ISSUES_TEXT>",
          "wrap": true
        }
      ]
    },
    {
      "type": "Container",
      "spacing": "medium",
      "separator": true,
      "items": [
        {
          "type": "TextBlock",
          "text": "Next Steps",
          "weight": "bolder"
        },
        {
          "type": "TextBlock",
          "text": "<NEXT_STEPS_TEXT>",
          "wrap": true
        }
      ]
    }
  ],
  "actions": [
    {
      "type": "Action.OpenUrl",
      "title": "View Full Report",
      "url": "<FULL_REPORT_URL>"
    }
  ]
}
```

---

## Recipe 5: Image Gallery Card

**Use case:** Display a grid of images with captions — useful for product listings,
document previews, or photo approvals.

**Power Automate action:** Post card in a chat or channel (no wait)

**Customisation points:**
- Replace `<IMAGE_N_URL>` with direct HTTPS image URLs (SharePoint direct links work)
- Replace `<CAPTION_N>` with image labels
- Each column is one image; add or remove columns as needed
- `"pixelWidth"` controls image size — adjust to fit your images

```json
{
  "type": "AdaptiveCard",
  "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
  "version": "1.4",
  "body": [
    {
      "type": "TextBlock",
      "text": "Image Gallery",
      "weight": "bolder",
      "size": "large"
    },
    {
      "type": "TextBlock",
      "text": "<GALLERY_DESCRIPTION>",
      "wrap": true,
      "isSubtle": true,
      "spacing": "none"
    },
    {
      "type": "Container",
      "spacing": "medium",
      "items": [
        {
          "type": "ColumnSet",
          "columns": [
            {
              "type": "Column",
              "width": "stretch",
              "items": [
                {
                  "type": "Image",
                  "url": "<IMAGE_1_URL>",
                  "altText": "<CAPTION_1>",
                  "size": "stretch"
                },
                {
                  "type": "TextBlock",
                  "text": "<CAPTION_1>",
                  "horizontalAlignment": "center",
                  "isSubtle": true,
                  "size": "small"
                }
              ]
            },
            {
              "type": "Column",
              "width": "stretch",
              "items": [
                {
                  "type": "Image",
                  "url": "<IMAGE_2_URL>",
                  "altText": "<CAPTION_2>",
                  "size": "stretch"
                },
                {
                  "type": "TextBlock",
                  "text": "<CAPTION_2>",
                  "horizontalAlignment": "center",
                  "isSubtle": true,
                  "size": "small"
                }
              ]
            },
            {
              "type": "Column",
              "width": "stretch",
              "items": [
                {
                  "type": "Image",
                  "url": "<IMAGE_3_URL>",
                  "altText": "<CAPTION_3>",
                  "size": "stretch"
                },
                {
                  "type": "TextBlock",
                  "text": "<CAPTION_3>",
                  "horizontalAlignment": "center",
                  "isSubtle": true,
                  "size": "small"
                }
              ]
            }
          ]
        },
        {
          "type": "ColumnSet",
          "spacing": "medium",
          "columns": [
            {
              "type": "Column",
              "width": "stretch",
              "items": [
                {
                  "type": "Image",
                  "url": "<IMAGE_4_URL>",
                  "altText": "<CAPTION_4>",
                  "size": "stretch"
                },
                {
                  "type": "TextBlock",
                  "text": "<CAPTION_4>",
                  "horizontalAlignment": "center",
                  "isSubtle": true,
                  "size": "small"
                }
              ]
            },
            {
              "type": "Column",
              "width": "stretch",
              "items": [
                {
                  "type": "Image",
                  "url": "<IMAGE_5_URL>",
                  "altText": "<CAPTION_5>",
                  "size": "stretch"
                },
                {
                  "type": "TextBlock",
                  "text": "<CAPTION_5>",
                  "horizontalAlignment": "center",
                  "isSubtle": true,
                  "size": "small"
                }
              ]
            },
            {
              "type": "Column",
              "width": "stretch",
              "items": [
                {
                  "type": "TextBlock",
                  "text": "",
                  "isSubtle": true
                }
              ]
            }
          ]
        }
      ]
    }
  ],
  "actions": [
    {
      "type": "Action.OpenUrl",
      "title": "View All Images",
      "url": "<GALLERY_URL>"
    }
  ]
}
```

**Note on SharePoint image URLs:** To get a direct (non-redirect) image URL from
SharePoint, use the REST API endpoint:
```
https://<tenant>.sharepoint.com/sites/<site>/_api/web/GetFileByServerRelativeUrl('/sites/<site>/Documents/<file.jpg>')/$value
```
You will need to pass an auth header for private sites.
