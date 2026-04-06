# Building Commodity Trading Agents

> **Reading time:** ~6 min | **Module:** Module 6: Production | **Prerequisites:** Modules 0-5

<div class="callout-key">

**Key Concept Summary:** Commodity trading agents autonomously gather data, analyze markets, and generate recommendations. This guide covers building agents that handle the full research-to-signal pipeline.

</div>

## Introduction

Commodity trading agents autonomously gather data, analyze markets, and generate recommendations. This guide covers building agents that handle the full research-to-signal pipeline.

<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## Agent Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Commodity Agent                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Memory    │  │   Tools     │  │   Actions   │         │
│  │             │  │             │  │             │         │
│  │ - Context   │  │ - API calls │  │ - Research  │         │
│  │ - History   │  │ - Database  │  │ - Analysis  │         │
│  │ - State     │  │ - Web fetch │  │ - Report    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                           │                                  │
│                     ┌─────▼─────┐                           │
│                     │    LLM    │                           │
│                     │  (Brain)  │                           │
│                     └───────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

### Agent Types

| Agent Type | Purpose | Data Sources |
|------------|---------|--------------|
| **Report Monitor** | Parse government reports | EIA, USDA, IEA |
| **News Analyst** | Sentiment analysis | Reuters, Bloomberg |
| **Balance Tracker** | Supply/demand modeling | Multiple agencies |
| **Signal Generator** | Trading signals | All above |

## Building a Report Monitor Agent

### Core Agent Class


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">class.py</span>
</div>

```python
from anthropic import Anthropic
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from datetime import datetime
import json

@dataclass
class AgentMemory:
    """Agent memory store."""
    context: List[Dict] = field(default_factory=list)
    reports_processed: List[str] = field(default_factory=list)
    extracted_data: List[Dict] = field(default_factory=list)
    max_context: int = 20

    def add_context(self, role: str, content: str):
        """Add to conversation context."""
        self.context.append({"role": role, "content": content})
        if len(self.context) > self.max_context:
            # Keep system message and recent context
            self.context = self.context[:1] + self.context[-(self.max_context-1):]

    def get_recent_extractions(self, n: int = 5) -> List[Dict]:
        """Get most recent extracted data."""
        return self.extracted_data[-n:]

class CommodityAgent:
    """
    Base commodity research agent.
    """

    def __init__(
        self,
        name: str,
        commodity: str,
        system_prompt: str,
        tools: Dict[str, Callable]
    ):
        self.name = name
        self.commodity = commodity
        self.client = Anthropic()
        self.memory = AgentMemory()
        self.tools = tools

        # Initialize system prompt
        self.memory.add_context("system", system_prompt)

    def think(self, observation: str) -> str:
        """
        Process observation and decide action.
        """
        self.memory.add_context("user", observation)

        # Build tool descriptions
        tool_desc = "\n".join([
            f"- {name}: {func.__doc__}"
            for name, func in self.tools.items()
        ])

        prompt = f"""
You are analyzing {self.commodity} markets.

Available tools:
{tool_desc}

Recent context:
{json.dumps(self.memory.get_recent_extractions(), indent=2)}

Based on the observation, decide what action to take.
Return JSON:
{{
  "reasoning": "your thought process",
  "action": "tool_name or 'respond'",
  "action_input": "input for the tool or response text"
}}
"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.content[0].text

    def act(self, action: str, action_input: str) -> str:
        """
        Execute action using tools.
        """
        if action == 'respond':
            return action_input

        if action in self.tools:
            result = self.tools[action](action_input)
            return str(result)

        return f"Unknown action: {action}"

    def run(self, task: str, max_steps: int = 5) -> str:
        """
        Run agent on a task.
        """
        observation = task
        final_response = ""

        for step in range(max_steps):
            # Think
            thought = self.think(observation)

            try:
                decision = json.loads(thought)
            except json.JSONDecodeError:
                decision = {"action": "respond", "action_input": thought}

            # Act
            result = self.act(
                decision.get('action', 'respond'),
                decision.get('action_input', '')
            )

            # Check if done
            if decision.get('action') == 'respond':
                final_response = result
                break

            # Update observation for next step
            observation = f"Tool result: {result}"

        self.memory.add_context("assistant", final_response)
        return final_response
```

</div>
</div>

### EIA Report Agent


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">create_eia_agent.py</span>
</div>

```python
import requests
from datetime import datetime, timedelta

def create_eia_agent():
    """Create agent for monitoring EIA reports."""

    def fetch_eia_data(series_id: str) -> dict:
        """Fetch data from EIA API. Input: EIA series ID"""
        api_key = os.environ.get('EIA_API_KEY')
        url = f"https://api.eia.gov/v2/petroleum/sum/sndw/data"
        params = {
            'api_key': api_key,
            'frequency': 'weekly',
            'data[0]': 'value',
            'length': 10
        }
        response = requests.get(url, params=params)
        return response.json()

    def parse_report_text(text: str) -> dict:
        """Parse EIA report text to extract key metrics. Input: report text"""
        client = Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"""Extract key metrics from this EIA report:

{text}

Return JSON with crude_stocks, gasoline_stocks, distillate_stocks, and weekly changes."""
            }]
        )
        return response.content[0].text

    def compare_to_expectations(data: str) -> dict:
        """Compare extracted data to market expectations. Input: JSON data string"""
        # In practice, fetch from consensus data provider
        expectations = {
            "crude_change": -2.0,
            "gasoline_change": 1.0,
            "distillate_change": -0.5
        }
        return {"expectations": expectations, "actual": json.loads(data)}

    system_prompt = """You are an EIA report analysis agent specialized in petroleum markets.
Your job is to:
1. Fetch the latest EIA data
2. Extract key inventory metrics
3. Compare to expectations
4. Generate a trading-relevant summary

Focus on surprises vs expectations and their price implications."""

    tools = {
        "fetch_eia_data": fetch_eia_data,
        "parse_report_text": parse_report_text,
        "compare_to_expectations": compare_to_expectations
    }

    return CommodityAgent(
        name="EIA_Monitor",
        commodity="crude_oil",
        system_prompt=system_prompt,
        tools=tools
    )
```

</div>
</div>

## Multi-Agent Systems

### Agent Orchestration


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agentorchestrator.py</span>
</div>

```python
class AgentOrchestrator:
    """
    Coordinate multiple commodity agents.
    """

    def __init__(self):
        self.agents: Dict[str, CommodityAgent] = {}
        self.results: Dict[str, List] = {}

    def register_agent(self, name: str, agent: CommodityAgent):
        """Register an agent."""
        self.agents[name] = agent
        self.results[name] = []

    def run_all(self, task: str) -> Dict[str, str]:
        """
        Run all agents on a task.
        """
        outputs = {}
        for name, agent in self.agents.items():
            try:
                result = agent.run(task)
                outputs[name] = result
                self.results[name].append({
                    'timestamp': datetime.now().isoformat(),
                    'task': task,
                    'result': result
                })
            except Exception as e:
                outputs[name] = f"Error: {str(e)}"

        return outputs

    def synthesize_results(self, outputs: Dict[str, str]) -> str:
        """
        Synthesize results from multiple agents.
        """
        client = Anthropic()

        prompt = f"""Synthesize these commodity analysis results from multiple agents:

{json.dumps(outputs, indent=2)}

Provide:
1. Key findings across all sources
2. Areas of agreement/disagreement
3. Overall market assessment
4. Recommended action"""

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text
```

</div>
</div>

### Example Multi-Agent Setup


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">create_commodity_analysis_system.py</span>
</div>

```python
def create_commodity_analysis_system():
    """
    Create multi-agent commodity analysis system.
    """
    orchestrator = AgentOrchestrator()

    # Register specialized agents
    orchestrator.register_agent("eia_monitor", create_eia_agent())

    # Add more agents...
    # orchestrator.register_agent("news_analyst", create_news_agent())
    # orchestrator.register_agent("balance_tracker", create_balance_agent())

    return orchestrator

# Usage
system = create_commodity_analysis_system()
results = system.run_all("Analyze this week's petroleum market developments")
synthesis = system.synthesize_results(results)
print(synthesis)
```

</div>
</div>

## Scheduled Agent Execution

### Event-Driven Triggers


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agentscheduler.py</span>

```python
from datetime import datetime, time
import schedule

class AgentScheduler:
    """
    Schedule agent execution around market events.
    """

    def __init__(self, orchestrator: AgentOrchestrator):
        self.orchestrator = orchestrator

    def schedule_eia_report(self):
        """Schedule analysis around EIA report release."""
        # EIA WPSR releases Wednesday 10:30 AM ET
        schedule.every().wednesday.at("10:35").do(
            self._run_task,
            "Analyze newly released EIA Weekly Petroleum Status Report"
        )

    def schedule_wasde(self):
        """Schedule analysis around WASDE release."""
        # WASDE releases around 12:00 ET on designated days
        schedule.every().day.at("12:05").do(
            self._check_wasde_release
        )

    def _run_task(self, task: str):
        """Execute task through orchestrator."""
        results = self.orchestrator.run_all(task)
        synthesis = self.orchestrator.synthesize_results(results)

        # Store or send results
        self._store_results(task, synthesis)

    def _check_wasde_release(self):
        """Check if WASDE was released today."""
        # Logic to check USDA calendar
        pass

    def _store_results(self, task: str, results: str):
        """Store analysis results."""
        # Store to database, send alerts, etc.
        pass

    def run(self):
        """Run the scheduler."""
        while True:
            schedule.run_pending()
            time.sleep(60)
```


<div class="callout-insight">

**Insight:** Understanding building commodity trading agents is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.




## Key Takeaways

1. **Agents automate research** - tools + LLM reasoning = autonomous analysis

2. **Specialized agents** - build focused agents for specific data sources

3. **Multi-agent systems** - orchestrate multiple agents for comprehensive analysis

4. **Event-driven** - schedule agents around market-moving releases

5. **Memory matters** - maintain context for coherent ongoing analysis

---

## Conceptual Practice Questions

1. What tasks in commodity research are well-suited for agentic AI workflows?

2. How does a multi-agent architecture improve upon single-prompt commodity analysis?

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.


---



## Cross-References

<a class="link-card" href="./01_commodity_agents_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_pipeline_build.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_production_deployment.md">
  <div class="link-card-title">01 Production Deployment</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_monitoring.md">
  <div class="link-card-title">02 Monitoring</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

