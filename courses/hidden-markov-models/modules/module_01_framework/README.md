# Module 1: HMM Framework

## Overview

Define the complete Hidden Markov Model framework: states, observations, transition probabilities, and emission distributions.

**Time Estimate:** 6-8 hours

## Learning Objectives

By completing this module, you will:
1. Define all HMM components formally
2. Understand emission distributions
3. Specify HMM parameters
4. Simulate from HMMs

## Contents

### Guides
- `01_hmm_components.md` - States, transitions, emissions
- `02_emission_distributions.md` - Discrete and continuous
- `03_hmm_simulation.md` - Generating from HMMs

### Notebooks
- `01_hmm_definition.ipynb` - Defining HMMs
- `02_simulation.ipynb` - Generating synthetic data

## Key Concepts

### HMM Definition

An HMM $\lambda = (\pi, A, B)$ consists of:

- **States**: $S = \{s_1, ..., s_K\}$
- **Initial distribution**: $\pi_i = P(q_1 = s_i)$
- **Transition matrix**: $A_{ij} = P(q_{t+1} = s_j | q_t = s_i)$
- **Emission distribution**: $B_i(o) = P(o_t | q_t = s_i)$

### The Three Problems

| Problem | Question | Use |
|---------|----------|-----|
| Evaluation | $P(O \| \lambda)$ | Model comparison |
| Decoding | Best state sequence | Regime identification |
| Learning | Best $\lambda$ | Parameter estimation |

### Emission Types

```python
# Discrete emissions
B = [
    [0.3, 0.5, 0.2],  # State 1: P(symbol 1,2,3)
    [0.1, 0.2, 0.7],  # State 2: P(symbol 1,2,3)
]

# Gaussian emissions
class GaussianHMM:
    means = [0.05, -0.03]      # State means
    variances = [0.01, 0.04]   # State variances
```

### Simulation

```python
def simulate_hmm(hmm, n_steps):
    states = []
    observations = []

    # Initial state
    state = np.random.choice(K, p=hmm.pi)

    for t in range(n_steps):
        states.append(state)
        observations.append(hmm.emit(state))
        state = np.random.choice(K, p=hmm.A[state])

    return states, observations
```

## Prerequisites

- Module 0 completed
- Probability distributions
