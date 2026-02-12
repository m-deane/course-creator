# Thompson Sampling

```
┌─────────────────────────────────────────────────┐
│           THOMPSON SAMPLING                      │
├─────────────────────────────────────────────────┤
│                                                  │
│  Step 1: SAMPLE from each arm's belief           │
│                                                  │
│  Arm A: Beta(12,8)  → sample: 0.58              │
│          ╭──╮                                    │
│       ╭──╯  ╰──╮                                │
│    ───╯        ╰───                              │
│                                                  │
│  Arm B: Beta(3,2)   → sample: 0.71              │
│       ╭────────╮                                 │
│    ───╯        ╰───                              │
│    (wide = uncertain)                            │
│                                                  │
│  Arm C: Beta(25,30) → sample: 0.44              │
│            ╭╮                                    │
│         ╭──╯╰──╮                                │
│    ─────╯      ╰─────                           │
│    (narrow = confident)                          │
│                                                  │
│  Step 2: PICK the highest sample → Arm B (0.71) │
│                                                  │
│  Step 3: OBSERVE reward (success/failure)        │
│                                                  │
│  Step 4: UPDATE belief                           │
│    Success: Beta(α+1, β)                         │
│    Failure: Beta(α, β+1)                         │
│                                                  │
│  ──► Repeat ──►                                  │
│                                                  │
├─────────────────────────────────────────────────┤
│ TL;DR: Ask each option to "make its best case." │
│ Uncertain options sometimes win, so they get     │
│ tested. Confident losers rarely win samples.     │
├─────────────────────────────────────────────────┤
│ Code (< 15 lines):                               │
│                                                  │
│   import numpy as np                             │
│   alphas = np.ones(K)  # prior successes         │
│   betas = np.ones(K)   # prior failures          │
│   for t in range(1000):                          │
│       samples = np.random.beta(alphas, betas)    │
│       arm = np.argmax(samples)                   │
│       reward = pull(arm)  # 0 or 1               │
│       alphas[arm] += reward                      │
│       betas[arm] += 1 - reward                   │
│                                                  │
├─────────────────────────────────────────────────┤
│ Common Pitfall: Using strong priors (e.g.,       │
│ Beta(100,100)) slows learning. Start with        │
│ Beta(1,1) unless you have real prior knowledge.  │
└─────────────────────────────────────────────────┘
```
