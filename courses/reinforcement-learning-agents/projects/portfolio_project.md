# Portfolio Project: Train a Domain-Specific Agent with RL

## Overview

Build and train a custom agent that masters a domain-specific tool environment through reinforcement learning. You'll create your own MCP server, design reward functions, and train a model using ART/GRPO.

**Estimated Time:** 15-20 hours
**Deliverable:** Working trained agent + evaluation report

---

## Project Options

Choose one domain (or propose your own):

### Option A: Customer Support Agent
- **Environment:** MCP server wrapping a ticketing system database
- **Tools:** search_tickets, get_ticket_details, get_customer_history, suggest_resolution
- **Goal:** Agent learns to diagnose issues and suggest resolutions
- **Evaluation:** Resolution accuracy, number of tool calls, customer satisfaction proxy

### Option B: Code Review Agent
- **Environment:** MCP server wrapping a Git repository
- **Tools:** list_files, read_file, search_code, get_diff, check_style
- **Goal:** Agent learns to identify bugs and style issues in code diffs
- **Evaluation:** Bug detection rate, false positive rate, review quality

### Option C: Research Assistant Agent
- **Environment:** MCP server wrapping a document/paper database
- **Tools:** search_papers, get_abstract, get_citations, get_full_text, compare_methods
- **Goal:** Agent learns to answer research questions by finding and synthesizing papers
- **Evaluation:** Answer accuracy, source relevance, synthesis quality

### Option D: Data Analysis Agent
- **Environment:** MCP server wrapping a multi-table analytics database
- **Tools:** list_datasets, describe_columns, run_query, compute_statistics, create_chart
- **Goal:** Agent learns to answer business questions from raw data
- **Evaluation:** Query correctness, insight quality, analysis completeness

---

## Milestones

### Milestone 1: Environment Setup (3-4 hours)
- [ ] Create SQLite database with realistic data (minimum 3 tables, 100+ rows)
- [ ] Build MCP server with 3-5 tools using FastMCP
- [ ] Test all tools manually
- [ ] Document the database schema and tool capabilities

**Deliverable:** Working MCP server + database

### Milestone 2: Scenario Generation (2-3 hours)
- [ ] Generate 20+ training scenarios using ART's auto-generation
- [ ] Review and curate scenarios for difficulty distribution
- [ ] Create 10 held-out evaluation scenarios with known correct answers
- [ ] Categorize scenarios by complexity (single-tool, multi-tool, multi-step)

**Deliverable:** Scenario set in JSON format

### Milestone 3: Reward Design (3-4 hours)
- [ ] Implement programmatic reward function for your domain
- [ ] Configure RULER for subjective quality scoring
- [ ] Build hybrid reward combining programmatic + RULER scores
- [ ] Validate rewards on sample trajectories (sanity check)

**Deliverable:** Reward function module with tests

### Milestone 4: Training (3-4 hours)
- [ ] Configure ART with Qwen2.5-3B (or larger if GPU allows)
- [ ] Run initial training (10 steps) to verify pipeline works
- [ ] Run full training (50-100 steps)
- [ ] Monitor reward curves and trajectory quality
- [ ] Save checkpoints at intervals

**Deliverable:** Trained model checkpoint + training logs

### Milestone 5: Evaluation & Report (3-5 hours)
- [ ] Evaluate trained model on held-out scenarios
- [ ] Compare against base model (before training)
- [ ] Measure: accuracy, average turns, error rate, latency
- [ ] Analyze failure cases — what does the model still get wrong?
- [ ] Write evaluation report

**Deliverable:** Evaluation report with quantitative results

---

## Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Environment Quality | 20% | Realistic data, well-designed tools, proper MCP implementation |
| Reward Design | 20% | Thoughtful reward function that captures task success |
| Training Execution | 20% | Complete training run with evidence of improvement |
| Evaluation Rigor | 25% | Quantitative comparison, held-out test set, failure analysis |
| Code Quality | 15% | Clean, documented, reproducible code |

---

## Starter Code

Use these templates from the course:

```python
# 1. MCP Server: templates/mcp_database_server.py
# 2. Reward Functions: templates/reward_functions.py
# 3. Training Config: templates/art_training_config.py
# 4. Training Loop: recipes/04_training_loop.py
```

---

## Tips

1. **Start small:** Get the full pipeline working with a tiny database before scaling
2. **Reward engineering matters:** Spend time validating your reward function on edge cases
3. **Log everything:** Save trajectories from early and late training for comparison
4. **Ablate:** Try different group sizes, learning rates, and training steps
5. **Fail analysis is gold:** Understanding *why* the model fails teaches more than celebrating wins
