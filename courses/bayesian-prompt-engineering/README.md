# Bayesian Prompt Engineering: Conditional Probability for AI Mastery

> *Every bad AI answer is a tax you pay for making the model guess your conditions.*

## Course Overview

A practical-first course teaching prompt engineering through the lens of conditional probability and Bayesian reasoning. Based on the core insight that **"your prompt is evidence that defines the posterior,"** this course transforms prompt writing from a bag of hacks into a rigorous probability skill.

Students learn to specify conditions that collapse AI models into the right "world" — producing precise, operational answers from LLMs, AI agents, multi-agent workflows, and Claude specifically.

## Target Audience

- AI practitioners and developers building with LLMs
- Product managers using AI tools daily
- Analysts and professionals who interact with AI but get inconsistent results
- Anyone who wants to move beyond "prompt tips" to first-principles understanding

## Core Thesis

Prompt engineering is not a writing skill — it's a **probability skill**. When you specify the right conditions (evidence), you control the posterior distribution the model reasons inside.

## Prerequisites

- Basic Python programming
- Familiarity with using AI chatbots (ChatGPT, Claude, etc.)
- No statistics or probability background required (we build it from scratch)
- An Anthropic API key for hands-on exercises

## Modules

| Module | Topic | Key Skill |
|--------|-------|-----------|
| 0 | **Foundations: How Language Models Actually Work** | Understand autoregressive generation as conditional probability |
| 1 | **The Bayesian Frame: Prompts as Evidence** | See prompts as evidence that shapes P(A\|C) |
| 2 | **Switch Variables: The Conditions That Actually Matter** | Identify the few conditions that flip solution branches |
| 3 | **The Condition Stack Framework** | Build 6-layer condition stacks for any domain |
| 4 | **Conditional Trees: When One Answer Is Wrong** | Prompt for decision trees instead of single verdicts |
| 5 | **Agents and Workflows** | Apply Bayesian conditioning to agents and multi-step systems |
| 6 | **Common Probability Mistakes** | Diagnose and fix the 6 most common prompting errors |
| 7 | **Production Patterns at Scale** | Build production prompt pipelines with dynamic condition injection |

## Capstone Project

Build a complete Bayesian prompt system for a real-world domain — including condition stack templates, switch variable catalogs, decision tree prompts, and a working Claude agent that applies them automatically.

## Setup

```bash
pip install anthropic matplotlib numpy ipywidgets plotly
```

Set your API key:
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

## Philosophy

- **Working code first**, theory contextually
- **Visual-first** — diagram before text, always
- **15-minute max** for any single notebook
- **Copy-paste ready** — all code works in your own projects
- **Portfolio over grades** — build reusable prompt systems, not quiz answers
