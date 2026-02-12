# Portfolio Projects & Course Resources

## Start Here

This directory contains three hands-on portfolio projects and comprehensive course resources for the Multi-Armed Bandits & A/B Testing course.

**New to bandits?** Start with Project 1.
**Want production code?** Jump to Project 3.
**Need quick reference?** Check the cheat sheet.

---

## Portfolio Projects

### Project 1: Content Strategy Optimizer (Beginner)
📁 `/projects/project_1_beginner/`

**What you'll build:** A bandit system that optimizes content publishing strategy over 52 weeks using Thompson Sampling.

**Time:** 3-4 hours  
**Difficulty:** Beginner  
**Prerequisites:** Python, basic probability

**Files:**
- `README.md` - Complete project guide
- `starter_code.py` - Working skeleton with TODOs
- `solution.py` - Reference implementation

**You'll learn:**
- Thompson Sampling for Beta-Bernoulli bandits
- Designing repeatable arms (topic × format)
- Reward functions (read ratio vs vanity metrics)
- Arm retirement for evolving strategies

**Start:** Read `README.md` then open `starter_code.py`

---

### Project 2: Commodity Allocation Engine (Intermediate)
📁 `/projects/project_2_intermediate/`

**What you'll build:** A two-wallet commodity allocator (80% core, 20% bandit sleeve) with real market data and safety guardrails.

**Time:** 4-5 hours  
**Difficulty:** Intermediate  
**Prerequisites:** Project 1, understanding of returns/volatility

**Files:**
- `README.md` - Comprehensive guide
- `starter_code.py` - Two-wallet framework with TODOs
- `solution.py` - Complete implementation with benchmarks

**You'll learn:**
- Two-wallet framework (stable core + adaptive bandit)
- Risk-adjusted rewards (Sharpe ratio)
- Safety guardrails (position limits, tilt speed, minimum allocation)
- Normal-Normal conjugacy for continuous rewards
- Backtesting on real commodity data (yfinance)

**Start:** Read `README.md` then open `starter_code.py`

---

### Project 3: Production Regime-Aware Trading Allocator (Advanced)
📁 `/projects/project_3_advanced/`

**What you'll build:** Institutional-grade commodity allocator with contextual bandits, regime detection, comprehensive guardrails, monitoring, and deployment guide.

**Time:** 6-8 hours  
**Difficulty:** Advanced  
**Prerequisites:** Projects 1 & 2, production system experience helpful

**Files:**
- `README.md` - System architecture and guide
- `starter_code.py` - Production skeleton (feature pipeline, LinUCB, guardrails, monitoring)
- `solution.py` - Complete production implementation
- `deploy.md` - Deployment guide (cron, Airflow, Lambda)

**You'll learn:**
- LinUCB (contextual bandit with linear models)
- Feature engineering for regime detection
- Production guardrail systems (5+ types)
- Monitoring, logging, and alerting
- Offline evaluation and stress testing
- Deployment patterns and scheduling

**Start:** Read `README.md` then review `starter_code.py` structure

---

## Course Resources

### Setup Guide
📄 `/resources/setup.md`

Complete environment setup with verification script.

**Contents:**
- Conda and pip installation
- Package versions (tested)
- Data source configuration (yfinance, FRED)
- Jupyter Lab setup
- Troubleshooting guide
- Google Colab alternative

**Use:** Run before starting projects to verify your environment.

---

### Glossary
📄 `/resources/glossary.md`

70+ terms with concise, practical definitions.

**Sections:**
- Core bandit concepts
- Classic algorithms
- Bayesian bandits
- Contextual bandits
- Non-stationary bandits
- Commodity-specific terms
- Trading applications
- Risk-adjusted rewards
- Theoretical concepts
- Production deployment

**Use:** Bookmark and refer to when encountering unfamiliar terms.

---

### Cheat Sheet
📄 `/resources/cheat_sheet.md`

One-page reference card with formulas, decision trees, and code snippets.

**Contents:**
- Algorithm comparison table
- Key formulas (UCB1, Thompson, LinUCB, regret, Sharpe)
- Decision tree: "Which algorithm should I use?"
- Reward design checklist
- Guardrail checklist
- Commodity allocation quick-start (6 steps)
- Code snippets (< 5 lines each)
- Common pitfalls
- Parameter recommendations

**Use:** Print and keep handy while working through projects.

---

## Visual Concept Guides

60-second visual concept cards for quick understanding.

### Explore-Exploit Tradeoff
📄 `/concepts/visual_guides/explore_exploit_tradeoff.md`

**Covers:** The fundamental tension in bandits, spectrum from random to greedy, Thompson Sampling as balanced approach.

**Format:** ASCII diagram + code + common pitfall

---

### Thompson Sampling
📄 `/concepts/visual_guides/thompson_sampling_visual.md`

**Covers:** Posterior sampling, automatic exploration-exploitation balance, evolution from wide to narrow posteriors.

**Format:** ASCII diagram of Beta distributions + code + common pitfall

---

### Two-Wallet Framework
📄 `/concepts/visual_guides/two_wallet_framework.md`

**Covers:** Portfolio structure, core stability + bandit adaptation, allocation combination.

**Format:** ASCII portfolio diagram + code + common pitfall

---

## Deep Dives (Optional Theory)

For learners who want deeper theoretical understanding.

### Regret Theory
📄 `/concepts/deep_dives/01_regret_theory.md`

**Covers:**
- Cumulative vs simple regret
- Lai-Robbins lower bound (Ω(log T) is optimal)
- UCB regret bound (O(log T) upper bound)
- When regret analysis matters in practice
- Commodity trading context

**Depth:** Formal definitions + intuition + practical implications

---

### Bayesian Bandits Theory
📄 `/concepts/deep_dives/02_bayesian_bandits_theory.md`

**Covers:**
- Bayesian updating (prior → posterior)
- Conjugate pairs (Beta-Bernoulli, Normal-Normal, Gamma-Poisson)
- Gittins index (theoretically optimal solution)
- Why Thompson Sampling works so well
- Connection to Bayesian Commodity Forecasting course

**Depth:** Formal framework + Gittins theorem + Bayesian decision theory

---

## Learning Paths

### Path 1: Quick Projects (12-15 hours)
1. Setup verification (`resources/setup.md`)
2. Project 1 (3-4 hours)
3. Project 2 (4-5 hours)
4. Project 3 (6-8 hours)

### Path 2: Theory-First (20-25 hours)
1. Setup + cheat sheet
2. Visual guides (all 3)
3. Deep dive: Regret theory
4. Project 1
5. Deep dive: Bayesian bandits
6. Project 2
7. Project 3 + deploy guide

### Path 3: Production Focus (15-18 hours)
1. Setup
2. Cheat sheet (bookmark)
3. Project 1 (quick completion)
4. Project 2 (focus on guardrails)
5. Project 3 (full implementation)
6. Deploy guide (study deployment patterns)

---

## Quick Reference

**Need help with:**
- Environment setup → `resources/setup.md`
- Term definition → `resources/glossary.md`
- Formula or algorithm → `resources/cheat_sheet.md`
- Quick concept → `concepts/visual_guides/`
- Deep theory → `concepts/deep_dives/`

**Want to build:**
- Simple bandit → Project 1
- Trading allocator → Project 2
- Production system → Project 3

**Looking for:**
- Code examples → All starter_code.py files
- Complete solutions → All solution.py files
- Deployment guide → `projects/project_3_advanced/deploy.md`

---

## File Structure

```
/multi-armed-bandits-ab-testing/
├── PROJECTS_AND_RESOURCES.md (this file)
│
├── projects/
│   ├── PROJECT_SUMMARY.md
│   ├── project_1_beginner/
│   │   ├── README.md
│   │   ├── starter_code.py
│   │   └── solution.py
│   ├── project_2_intermediate/
│   │   ├── README.md
│   │   ├── starter_code.py
│   │   └── solution.py
│   └── project_3_advanced/
│       ├── README.md
│       ├── starter_code.py
│       ├── solution.py
│       └── deploy.md
│
├── resources/
│   ├── setup.md
│   ├── glossary.md
│   └── cheat_sheet.md
│
└── concepts/
    ├── visual_guides/
    │   ├── explore_exploit_tradeoff.md
    │   ├── thompson_sampling_visual.md
    │   └── two_wallet_framework.md
    └── deep_dives/
        ├── 01_regret_theory.md
        └── 02_bayesian_bandits_theory.md
```

---

## Key Principles

All materials follow the course's practical-first philosophy:

✅ **Working code in 2 minutes** - Every project has runnable starter code  
✅ **Real datasets** - yfinance commodity data with synthetic fallback  
✅ **Copy-paste ready** - All code works in your own projects  
✅ **Visual-first** - ASCII diagrams before equations  
✅ **Portfolio over grades** - Build real systems, not pass tests  

---

**Ready to start?** Go to `resources/setup.md` to verify your environment, then jump into Project 1!
