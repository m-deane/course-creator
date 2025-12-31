# Module 0: LLM Mesh Foundations

## Overview

Set up and configure Dataiku's LLM Mesh for enterprise Gen AI applications. Connect to multiple providers and understand the abstraction layer.

**Time Estimate:** 4-6 hours

## Learning Objectives

By completing this module, you will:
1. Configure LLM Mesh connections
2. Set up multiple provider endpoints
3. Understand cost tracking and governance
4. Make your first LLM Mesh calls

## Contents

### Guides
- `01_llm_mesh_architecture.md` - How LLM Mesh works
- `02_provider_setup.md` - Configuring Anthropic, OpenAI
- `03_governance.md` - Cost tracking and access control

### Notebooks
- `01_first_connection.ipynb` - Initial LLM Mesh setup
- `02_provider_comparison.ipynb` - Testing different providers

## Key Concepts

### LLM Mesh Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Dataiku DSS                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │                  LLM Mesh                       │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐         │   │
│  │  │ Claude  │  │  GPT-4  │  │  Azure  │         │   │
│  │  │Connection│  │Connection│  │Connection│         │   │
│  │  └─────────┘  └─────────┘  └─────────┘         │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│  ┌──────────────────────┼──────────────────────────┐   │
│  │       Projects using LLM Mesh                   │   │
│  │  ┌────────┐  ┌────────┐  ┌────────┐            │   │
│  │  │Project1│  │Project2│  │Project3│            │   │
│  │  └────────┘  └────────┘  └────────┘            │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Key Features

| Feature | Description |
|---------|-------------|
| Provider Abstraction | Single API for all providers |
| Cost Tracking | Centralized usage monitoring |
| Access Control | Role-based LLM access |
| Fallback | Automatic provider failover |

### Connection Types

- Direct API (Anthropic, OpenAI)
- Azure OpenAI
- Google Vertex AI
- Custom endpoints

## Prerequisites

- Dataiku DSS 12.0+
- Admin access for LLM Mesh config
- Provider API keys
