# Module 8: Bandits for LLM Prompt Routing — Eliminate the Bad Prompt Tax

## Overview

Prompt engineering is expensive because the learning loop is slow. You test prompts manually, wait for user feedback, iterate, and hope you found the best approach. Meanwhile, your system keeps using suboptimal prompts that cost you money in tokens, time, and user trust. This is the "bad prompt tax."

Multi-armed bandits solve this. Instead of manually A/B testing prompts, let your system learn which prompt template works best for each type of request — while still serving real users. Apply the same explore-exploit framework from commodity trading to LLM prompt selection, and watch your system improve without manual intervention.

This module teaches you to build prompt routing bandits for commodity trading research assistants, report analysis systems, and trading signal generation. You'll learn to design prompt arms, define reward functions that don't train hallucinations, and implement contextual routing that adapts to task type, commodity sector, and data availability.

**Why this matters for commodity trading:** Your research assistant needs different prompts for extracting EIA inventory data versus generating fundamental analysis versus producing trading signals. Manual prompt engineering doesn't scale across hundreds of commodity reports and market conditions. Bandits let the system learn which prompt works best for each scenario — and adapt as markets change.

## Learning Objectives

After completing this module, you will be able to:

- **Design prompt arms and reward functions** for LLM systems that balance quality, accuracy, and guardrails
- **Implement prompt routing bandits** using Thompson Sampling and contextual bandits for commodity analysis tasks
- **Build contextual prompt routing** based on task type, commodity sector, data availability, and market conditions
- **Reduce the "bad prompt tax"** in production LLM pipelines with adaptive prompt selection
- **Deploy production-grade prompt routers** with hallucination detection, cost controls, and performance monitoring
- **Connect bandit concepts** from previous modules to modern GenAI applications

## Module Contents

### Concept Guides
- **01_prompt_routing_fundamentals.md** — Why prompts are a perfect bandit problem and how to design prompt arms
- **02_reward_design_llm.md** — Defining rewards that don't train confident hallucinations
- **03_contextual_prompt_routing.md** — Routing based on task type, commodity sector, and data availability
- **04_commodity_research_assistant.md** — Case studies of prompt routing for commodity analysis systems
- **cheatsheet.md** — Quick reference for prompt routing architecture and design patterns

### Interactive Notebooks
- **01_prompt_routing_bandit.ipynb** — Build a Thompson Sampling prompt router for commodity analysis (15 min)
- **02_reward_function_design.ipynb** — Design reward functions that prevent hallucinations (15 min)
- **03_contextual_commodity_router.ipynb** — Build context-aware prompt routing for commodity research (15 min)

### Self-Check Exercises
- **exercises.py** — Practice implementing prompt routers and reward functions (ungraded)

### Supporting Resources
- **additional_readings.md** — Research papers, blog posts, and connections to GenAI for Commodities course
- **figures/** — Architecture diagrams and visual assets

## Completion Criteria

You're ready to deploy prompt routing bandits when you can:

1. ✅ Design 5-8 prompt arms for a specific commodity trading LLM task
2. ✅ Define composite reward functions with primary metrics and guardrails
3. ✅ Implement contextual prompt routing based on task features
4. ✅ Explain why "user satisfaction only" rewards train hallucinations
5. ✅ Build a production-ready prompt router with monitoring and failsafes
6. ✅ Connect this to your work in Modules 2 (Thompson Sampling) and 3 (Contextual Bandits)

## Time Estimate

- **Quick path** (just notebooks): 45 minutes
- **Full path** (guides + notebooks + exercises): 3-4 hours
- **Deep dive** (everything + readings + implementation): 6-8 hours

## Real-World Impact

After completing this module, you'll understand how to:

- Build commodity research assistants that improve without manual prompt tuning
- Design EIA/USDA report processors that adapt to different report structures
- Create multi-commodity trading signal systems that learn optimal prompts per sector
- Reduce LLM costs by routing to simpler prompts when they work just as well
- Prevent hallucinations through reward design instead of post-hoc filtering

## Connection to Other Modules

- **Module 2 (Bayesian Bandits):** Thompson Sampling powers the prompt routing engine
- **Module 3 (Contextual Bandits):** Context features (task type, commodity) determine routing
- **Module 5 (Commodity Trading):** Same explore-exploit logic, applied to prompt selection
- **GenAI for Commodities Course:** Connects directly to RAG systems, agent design, and production deployment

## What's Next?

This is the final module of the course. You've now learned:
1. The explore-exploit tradeoff and regret minimization (Modules 0-1)
2. Bayesian updating with Thompson Sampling (Module 2)
3. Contextual decision-making with features (Module 3)
4. Applications to content, growth, and commodity trading (Modules 4-5)
5. Advanced topics and production systems (Modules 6-7)
6. Modern GenAI applications with prompt routing (Module 8)

**Next steps:**
- Apply bandits to your own commodity trading systems
- Build portfolio projects using the templates from this course
- Explore the GenAI for Commodities course for deeper LLM applications
- Deploy production bandit systems with monitoring and guardrails

---

*"The best prompt engineering is the kind you don't have to do. Let the system learn while it runs."*
