# Prompt Routing Bandits

```
┌─────────────────────────────────────────────────┐
│       BANDIT PROMPT ROUTING                      │
│       (Eliminate the Bad Prompt Tax)             │
├─────────────────────────────────────────────────┤
│                                                  │
│  User Request                                    │
│       │                                          │
│       ▼                                          │
│  ┌──────────┐    Context features:               │
│  │  BANDIT   │◄── task_type, commodity,           │
│  │  ROUTER   │    data_availability, urgency     │
│  └──────────┘                                    │
│       │                                          │
│       ├──► Arm 1: STRUCTURED_EXTRACTION          │
│       │    "Parse EIA report into JSON"          │
│       │                                          │
│       ├──► Arm 2: EVIDENCE_ONLY (RAG-safe)       │
│       │    "Answer using ONLY retrieved sources" │
│       │                                          │
│       ├──► Arm 3: ANALYTICAL                     │
│       │    "Supply/demand + fundamentals"        │
│       │                                          │
│       ├──► Arm 4: SIGNAL_GENERATION              │
│       │    "Long/short/neutral + confidence"     │
│       │                                          │
│       └──► Arm 5: SCENARIO_ANALYSIS              │
│            "Bull/base/bear with probabilities"   │
│                                                  │
│       │                                          │
│       ▼                                          │
│  ┌──────────┐                                    │
│  │   LLM    │ → Response                         │
│  └──────────┘                                    │
│       │                                          │
│       ▼                                          │
│  ┌──────────┐    Reward = quality_score           │
│  │  REWARD   │    - hallucination_penalty        │
│  │  SIGNAL   │    + citation_bonus               │
│  └──────────┘    (cost/latency guardrails)       │
│       │                                          │
│       └──► Update bandit beliefs                 │
│                                                  │
├─────────────────────────────────────────────────┤
│ TL;DR: Stop guessing which prompt works. Let a  │
│ bandit route requests to the best prompt while   │
│ still serving users. Learn while you ship.       │
├─────────────────────────────────────────────────┤
│ Code (< 15 lines):                               │
│                                                  │
│   prompts = ["structured", "evidence",           │
│              "analytical", "signal", "scenario"] │
│   alphas = np.ones(5)                            │
│   betas = np.ones(5)                             │
│   for request in requests:                       │
│       samples = np.random.beta(alphas, betas)    │
│       arm = np.argmax(samples)                   │
│       response = llm(prompts[arm], request)      │
│       quality = evaluate(response)               │
│       alphas[arm] += quality                     │
│       betas[arm] += 1 - quality                  │
│                                                  │
├─────────────────────────────────────────────────┤
│ Common Pitfall: Rewarding "sounds confident"     │
│ trains hallucinations. Always include factual    │
│ accuracy guardrails, especially for commodity    │
│ data where wrong numbers have real consequences. │
└─────────────────────────────────────────────────┘
```
