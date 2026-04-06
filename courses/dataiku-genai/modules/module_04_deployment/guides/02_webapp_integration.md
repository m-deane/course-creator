# Webapp Integration for Gen AI Applications

> **Reading time:** ~11 min | **Module:** 4 — Deployment | **Prerequisites:** Modules 0-3

## In Brief

Dataiku webapps provide a framework for building interactive user interfaces that integrate LLM capabilities. By combining Dataiku's visual webapp builder with LLM Mesh, you create production-ready Gen AI applications with authentication, governance, and monitoring built-in—without managing infrastructure.

<div class="callout-insight">

<strong>Key Insight:</strong> The best Gen AI applications hide complexity from users. A well-designed webapp transforms complex LLM interactions into simple, intuitive interfaces that non-technical users can leverage. Dataiku webapps handle the infrastructure, security, and deployment so you can focus on user experience and business logic.

</div>

<div class="callout-key">

<strong>Key Concept:</strong> Dataiku webapps provide a framework for building interactive user interfaces that integrate LLM capabilities. By combining Dataiku's visual webapp builder with LLM Mesh, you create production-ready Gen AI applications with authentication, governance, and monitoring built-in—without managing infra...

</div>

## Formal Definition

**Webapp Integration** encompasses:
- **Frontend Framework**: HTML/CSS/JavaScript UI built with Dataiku webapp templates
- **Backend API**: Python endpoints that handle LLM calls and business logic
- **State Management**: Session handling for conversations and user context
- **Authentication**: Integration with Dataiku security groups and permissions
- **Deployment**: One-click publishing to staging and production environments

## Intuitive Explanation

Think of Dataiku webapps like building a custom dashboard in Tableau or PowerBI, but instead of just visualizing data, you're creating interactive Gen AI applications. The webapp builder provides the structure (layout, components, routing), and you fill in the logic (what happens when user clicks submit, how to call LLMs, how to display results). The result is a professional application that looks and feels like a standalone product, but runs securely within Dataiku's governed environment.

## Visual Representation

```

┌─────────────────────────────────────────────────────────────┐
│              Dataiku Webapp Architecture                    │
└─────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────┐
  │         Frontend (Browser)              │
  │  ┌───────────────────────────────────┐  │
  │  │  HTML + JavaScript                │  │
  │  │  • Input forms                    │  │
  │  │  • Display results                │  │
  │  │  • Interactive elements           │  │
  │  └───────────┬───────────────────────┘  │
  └──────────────┼──────────────────────────┘
                 │ AJAX requests
                 ▼
  ┌─────────────────────────────────────────┐
  │         Backend (Python)                │
  │  ┌───────────────────────────────────┐  │
  │  │  Python Flask Endpoints           │  │
  │  │  • Receive user input             │  │
  │  │  • Call LLM Mesh                  │  │
  │  │  • Format responses               │  │
  │  │  • Handle errors                  │  │
  │  └───────────┬───────────────────────┘  │
  └──────────────┼──────────────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────────────┐
  │            LLM Mesh                     │
  │  ┌──────────┐  ┌──────────┐            │
  │  │ Claude   │  │  GPT-4   │            │
  │  └──────────┘  └──────────┘            │
  └─────────────────────────────────────────┘
```

## Code Implementation

### Basic Webapp Structure


<span class="filename">backend.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# backend.py - Dataiku Webapp Backend
import dataiku
from dataiku.llm import LLM
from flask import request, jsonify
import json

# Initialize LLM
llm = LLM("anthropic-claude")

@app.route('/analyze', methods=['POST'])
def analyze_report():
    """
    Endpoint to analyze a commodity report.
    """
    try:
        # Get input from frontend
        data = request.get_json()
        report_text = data.get('report_text', '')
        commodity = data.get('commodity', 'crude_oil')

        # Validate input
        if not report_text:
            return jsonify({
                'status': 'error',
                'message': 'Report text is required'
            }), 400

        # Build prompt
        prompt = f"""Analyze this {commodity} market report:

{report_text}

Return JSON with:
- sentiment: "bullish" | "bearish" | "neutral"
- key_metrics: dict of extracted values
- summary: brief summary
- confidence: 0-1 score"""

        # Call LLM
        response = llm.complete(
            prompt=prompt,
            temperature=0.3,
            max_tokens=800
        )

        # Parse response
        analysis = json.loads(response.text)

        # Return to frontend
        return jsonify({
            'status': 'success',
            'analysis': analysis,
            'tokens_used': response.usage.total_tokens,
            'cost_usd': response.cost
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})
```

</div>
</div>

The frontend HTML connects to the backend endpoints and renders the analysis results:


<span class="filename">frontend.html</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```html
<!-- frontend.html - Dataiku Webapp Frontend -->
<!DOCTYPE html>
<html>
<head>
    <title>Commodity Report Analyzer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .input-section, .output-section {
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        textarea {
            width: 100%;
            height: 300px;
            padding: 10px;
            font-family: monospace;
        }
        button {
            background-color: #007bff;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #0056b3;
        }
        button:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        .result {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-top: 10px;
        }
        .sentiment {
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        }
        .sentiment.bullish { color: #28a745; }
        .sentiment.bearish { color: #dc3545; }
        .sentiment.neutral { color: #ffc107; }
        .metrics {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }
        .metric {
            background: white;
            padding: 10px;
            border-radius: 3px;
            border: 1px solid #dee2e6;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #007bff;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <h1>📊 Commodity Report Analyzer</h1>

    <div class="container">
        <div class="input-section">
            <h2>Input</h2>

            <label for="commodity">Commodity:</label>
            <select id="commodity" style="width: 100%; padding: 5px; margin-bottom: 10px;">
                <option value="crude_oil">Crude Oil</option>
                <option value="natural_gas">Natural Gas</option>
                <option value="gold">Gold</option>
                <option value="copper">Copper</option>
            </select>

            <label for="reportText">Report Text:</label>
            <textarea id="reportText" placeholder="Paste market report here..."></textarea>

            <button id="analyzeBtn" onclick="analyzeReport()">Analyze Report</button>

            <div id="loading" class="loading">
                <div class="spinner"></div>
                <p>Analyzing report...</p>
            </div>
        </div>

        <div class="output-section">
            <h2>Analysis</h2>
            <div id="output">
                <p style="color: #6c757d;">Results will appear here after analysis...</p>
            </div>
        </div>
    </div>

    <script>
        async function analyzeReport() {
            const commodity = document.getElementById('commodity').value;
            const reportText = document.getElementById('reportText').value;

            if (!reportText.trim()) {
                alert('Please enter report text');
                return;
            }

            // Show loading state
            document.getElementById('analyzeBtn').disabled = true;
            document.getElementById('loading').style.display = 'block';
            document.getElementById('output').innerHTML = '';

            try {
                // Call backend API
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        commodity: commodity,
                        report_text: reportText
                    })
                });

                const data = await response.json();

                if (data.status === 'success') {
                    displayResults(data.analysis, data.tokens_used, data.cost_usd);
                } else {
                    displayError(data.message);
                }

            } catch (error) {
                displayError(error.message);
            } finally {
                // Hide loading state
                document.getElementById('analyzeBtn').disabled = false;
                document.getElementById('loading').style.display = 'none';
            }
        }

        function displayResults(analysis, tokens, cost) {
            const sentiment = analysis.sentiment;
            const sentimentClass = sentiment.toLowerCase();

            let metricsHtml = '';
            for (const [key, value] of Object.entries(analysis.key_metrics || {})) {
                metricsHtml += `
                    <div class="metric">
                        <strong>${formatKey(key)}:</strong> ${value}
                    </div>
                `;
            }

            const html = `
                <div class="result">
                    <div class="sentiment ${sentimentClass}">
                        ${sentiment.toUpperCase()}
                    </div>

                    <p><strong>Summary:</strong> ${analysis.summary}</p>

                    <p><strong>Confidence:</strong> ${(analysis.confidence * 100).toFixed(0)}%</p>

                    <h3>Key Metrics</h3>
                    <div class="metrics">
                        ${metricsHtml}
                    </div>

                    <p style="font-size: 12px; color: #6c757d; margin-top: 20px;">
                        Tokens: ${tokens} | Cost: $${cost.toFixed(4)}
                    </p>
                </div>
            `;

            document.getElementById('output').innerHTML = html;
        }

        function displayError(message) {
            document.getElementById('output').innerHTML = `
                <div class="result" style="background-color: #f8d7da; border: 1px solid #f5c6cb;">
                    <strong>Error:</strong> ${message}
                </div>
            `;
        }

        function formatKey(key) {
            return key.split('_').map(word =>
                word.charAt(0).toUpperCase() + word.slice(1)
            ).join(' ');
        }
    </script>
</body>
</html>
```

</div>
</div>

### Chatbot Interface


<span class="filename">backend.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# backend.py - Chatbot Backend
from dataiku.llm import ChatSession

# Store chat sessions by user
chat_sessions = {}

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handle chat messages with conversation history.
    """
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default')
        message = data.get('message', '')

        if not message:
            return jsonify({
                'status': 'error',
                'message': 'Message is required'
            }), 400

        # Get or create chat session for user
        if user_id not in chat_sessions:
            llm = LLM("anthropic-claude")
            session = ChatSession(llm)
            session.set_system_message(
                "You are a commodity market analyst assistant. "
                "Provide concise, data-driven insights."
            )
            chat_sessions[user_id] = session

        session = chat_sessions[user_id]

        # Send message and get response
        response = session.send(message)

        # Return response
        return jsonify({
            'status': 'success',
            'response': response.text,
            'tokens_used': response.usage.total_tokens,
            'cost_usd': response.cost
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/chat/reset', methods=['POST'])
def reset_chat():
    """Reset chat session for user."""
    data = request.get_json()
    user_id = data.get('user_id', 'default')

    if user_id in chat_sessions:
        del chat_sessions[user_id]

    return jsonify({'status': 'success'})

@app.route('/chat/history', methods=['GET'])
def chat_history():
    """Get conversation history."""
    user_id = request.args.get('user_id', 'default')

    if user_id not in chat_sessions:
        return jsonify({'history': []})

    session = chat_sessions[user_id]
    history = [
        {
            'role': msg.role,
            'content': msg.content,
            'timestamp': msg.timestamp
        }
        for msg in session.messages
    ]

    return jsonify({'history': history})
```

</div>
</div>

The chatbot frontend manages conversation state and renders the message history:


<span class="filename">frontend.html</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```html
<!-- frontend.html - Chatbot UI -->
<!DOCTYPE html>
<html>
<head>
    <title>Commodity Market Analyst Chatbot</title>
    <style>
        .chat-container {
            max-width: 800px;
            margin: 0 auto;
            border: 1px solid #ddd;
            border-radius: 10px;
            overflow: hidden;
        }
        .chat-header {
            background-color: #007bff;
            color: white;
            padding: 15px;
            text-align: center;
        }
        .chat-messages {
            height: 500px;
            overflow-y: auto;
            padding: 20px;
            background-color: #f8f9fa;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px 15px;
            border-radius: 10px;
            max-width: 70%;
        }
        .message.user {
            background-color: #007bff;
            color: white;
            margin-left: auto;
        }
        .message.assistant {
            background-color: white;
            border: 1px solid #dee2e6;
        }
        .chat-input {
            display: flex;
            padding: 15px;
            background-color: white;
            border-top: 1px solid #ddd;
        }
        .chat-input input {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin-right: 10px;
        }
        .chat-input button {
            padding: 10px 20px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h2>💬 Commodity Market Analyst</h2>
            <button onclick="resetChat()" style="background: transparent; border: 1px solid white; color: white; padding: 5px 10px; border-radius: 3px; cursor: pointer;">
                Reset Chat
            </button>
        </div>

        <div id="chatMessages" class="chat-messages">
            <div class="message assistant">
                Hello! I'm your commodity market analyst assistant. Ask me about oil, gas, metals, or other commodities.
            </div>
        </div>

        <div class="chat-input">
            <input
                type="text"
                id="messageInput"
                placeholder="Ask about commodity markets..."
                onkeypress="if(event.key==='Enter') sendMessage()"
            />
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        const userId = 'user_' + Math.random().toString(36).substr(2, 9);

        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();

            if (!message) return;

            // Display user message
            addMessage('user', message);
            input.value = '';

            try {
                // Call backend
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        user_id: userId,
                        message: message
                    })
                });

                const data = await response.json();

                if (data.status === 'success') {
                    addMessage('assistant', data.response);
                } else {
                    addMessage('assistant', 'Error: ' + data.message);
                }

            } catch (error) {
                addMessage('assistant', 'Error: ' + error.message);
            }
        }

        function addMessage(role, content) {
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            messageDiv.textContent = content;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        async function resetChat() {
            await fetch('/chat/reset', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({user_id: userId})
            });

            document.getElementById('chatMessages').innerHTML = `
                <div class="message assistant">
                    Chat reset. How can I help you?
                </div>
            `;
        }
    </script>
</body>
</html>
```

</div>
</div>

### Streaming Responses


<span class="filename">backend.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# backend.py - Streaming Responses
from flask import Response, stream_with_context
import json

@app.route('/stream-analysis', methods=['POST'])
def stream_analysis():
    """
    Stream LLM response as it generates.
    """
    data = request.get_json()
    report_text = data.get('report_text', '')

    def generate():
        """Generator function for streaming."""
        try:
            llm = LLM("anthropic-claude")

            prompt = f"Analyze this report:\n\n{report_text}"

            # Stream response
            for chunk in llm.complete_stream(prompt, max_tokens=1000):
                # Send chunk to frontend
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk.text})}\n\n"

            # Send completion signal
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream'
    )
```

</div>
</div>

The frontend JavaScript reads from the event stream and updates the UI incrementally:


<span class="filename">frontend.js</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```javascript
// Frontend - Handle streaming
async function streamAnalysis() {
    const reportText = document.getElementById('reportText').value;

    const response = await fetch('/stream-analysis', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({report_text: reportText})
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let output = document.getElementById('output');
    output.innerHTML = '';

    while (true) {
        const {value, done} = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = JSON.parse(line.substring(6));

                if (data.type === 'chunk') {
                    output.innerHTML += data.content;
                } else if (data.type === 'error') {
                    console.error('Stream error:', data.message);
                }
            }
        }
    }
}
```

</div>
</div>

## Common Pitfalls

**Pitfall 1: No Session Management**
- Storing conversation history in memory doesn't scale or persist across restarts
- Use Dataiku datasets or external storage for session persistence
- Implement session timeouts to prevent memory leaks

**Pitfall 2: Blocking UI During LLM Calls**
- Synchronous LLM calls freeze the interface
- Always use async/await patterns in frontend
- Show loading states and progress indicators

**Pitfall 3: No Error Handling in Frontend**
- Network errors and LLM failures break the user experience
- Display user-friendly error messages
- Implement retry logic and fallback behavior

**Pitfall 4: Exposing Sensitive Data**
- Returning raw LLM responses can leak prompts or internal logic
- Sanitize outputs before displaying to users
- Implement proper authentication and authorization

**Pitfall 5: Not Monitoring Usage**
- Webapps without usage tracking make optimization impossible
- Log every LLM call with user, prompt, tokens, and cost
- Create usage dashboards for monitoring and cost allocation

## Connections

<div class="callout-info">

<strong>How this connects to the rest of the course:</strong>

</div>

**Builds on:**
- LLM Mesh configuration (Module 0)
- Custom model wrappers (Module 3.2)
- Pipeline integration (Module 3.3)

**Leads to:**
- Production deployment strategies (Module 4.3)
- Usage monitoring and optimization
- Multi-user application patterns

**Related to:**
- Web application architecture
- Frontend/backend separation
- User experience design

## Practice Problems

1. **Basic Webapp**
   - Create a simple webapp that accepts text input and returns LLM analysis
   - Include loading states and error handling
   - Display token usage and cost to users

2. **Chat Interface**
   - Build a chatbot webapp with conversation history
   - Implement session management for multiple users
   - Add a "clear history" button

3. **Batch Analysis Tool**
   - Create a webapp where users upload CSV files with multiple texts
   - Process all rows with LLM in background
   - Show progress bar and download results when complete

4. **Comparison Tool**
   - Build a side-by-side comparison tool
   - Users submit same prompt to two different models
   - Display results, tokens, costs, and latency for comparison

5. **Streaming Chat**
   - Implement streaming responses for better UX
   - Display tokens as they arrive (like ChatGPT)
   - Allow users to stop generation mid-stream

## Further Reading

- **Dataiku Documentation**: [Webapp Framework](https://doc.dataiku.com/dss/latest/webapps/index.html) - Official webapp development guide

- **Flask Documentation**: [Quickstart](https://flask.palletsprojects.com/quickstart/) - Flask basics for Python backend

- **MDN Web Docs**: [Fetch API](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API) - Modern JavaScript API calls

- **Blog Post**: "Building Production-Ready LLM Webapps" - Best practices from real deployments (representative of industry patterns)

- **Research**: "User Experience Patterns for Generative AI Applications" - Emerging UX design principles for Gen AI (representative of current UX research)


## Resources

<a class="link-card" href="../notebooks/01_api_setup.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises for this topic.</div>
</a>
